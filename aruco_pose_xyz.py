# -*- coding: utf-8 -*-
"""Detecta marcadores ArUco, estima pose 3D e desenha os eixos X, Y e Z.

Exemplos:
    python aruco_pose_xyz.py --camera 0 --marker-size-mm 50
    python aruco_pose_xyz.py --video teste.mp4 --marker-size-mm 50
    python aruco_pose_xyz.py --camera 0 --marker-size-mm 50 --calibration camera_calibration.npz

O arquivo NPZ de calibracao deve conter:
    camera_matrix: matriz intrinseca 3x3
    dist_coeffs: coeficientes de distorcao
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np


ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deteccao ArUco com posicao XYZ e eixos 3D."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=str, default=None,
                        help="Caminho do video. Se omitido, usa webcam.")
    source.add_argument("--camera", type=int, default=0,
                        help="Indice da webcam (padrao: 0).")
    parser.add_argument("--marker-size-mm", type=float, required=True,
                        help="Tamanho externo real do marcador, em mm.")
    parser.add_argument("--dictionary", choices=sorted(ARUCO_DICTIONARIES),
                        default="DICT_5X5_250",
                        help="Dicionario ArUco (padrao: DICT_5X5_250).")
    parser.add_argument("--calibration", type=str, default=None,
                        help="Arquivo NPZ com camera_matrix e dist_coeffs.")
    parser.add_argument("--axis-length-mm", type=float, default=None,
                        help="Comprimento visual dos eixos. Padrao: metade do marcador.")
    parser.add_argument("--csv", type=str, default=None,
                        help="CSV opcional para registrar poses por frame.")
    parser.add_argument("--width", type=int, default=None,
                        help="Largura solicitada para a webcam.")
    parser.add_argument("--height", type=int, default=None,
                        help="Altura solicitada para a webcam.")
    return parser.parse_args()


def open_capture(args):
    if args.video:
        capture = cv2.VideoCapture(args.video)
    else:
        capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(args.camera)
        if args.width:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        if args.height:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        raise RuntimeError("Nao foi possivel abrir a fonte de video.")
    return capture


def load_camera_calibration(path, frame_width, frame_height):
    """Carrega calibracao ou cria uma aproximacao para demonstracao."""
    if path:
        data = np.load(path)
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data["dist_coeffs"], dtype=np.float64)
        return camera_matrix, dist_coeffs, True

    # Aproximacao pinhole. Permite executar, mas XYZ nao e metrologico.
    focal = float(max(frame_width, frame_height))
    camera_matrix = np.array([
        [focal, 0.0, frame_width / 2.0],
        [0.0, focal, frame_height / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs, False


def marker_object_points(marker_size_mm):
    """Quatro cantos do marcador no sistema local do marcador.

    X aponta para a direita, Y aponta para cima e Z sai do plano.
    A ordem corresponde aos cantos detectados pelo ArUco:
    superior esquerdo, superior direito, inferior direito, inferior esquerdo.
    """
    half = marker_size_mm / 2.0
    return np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float32)


def rotation_matrix_to_euler_degrees(rotation_matrix):
    """Converte matriz de rotacao em roll, pitch e yaw, em graus."""
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = np.arctan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def estimate_marker_pose(corners, object_points, camera_matrix, dist_coeffs):
    image_points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    roll, pitch, yaw = rotation_matrix_to_euler_degrees(rotation_matrix)
    return rvec, tvec, rotation_matrix, (roll, pitch, yaw)


def draw_pose_information(frame, marker_id, corners, rvec, tvec, euler,
                          camera_matrix, dist_coeffs, axis_length_mm):
    cv2.drawFrameAxes(
        frame, camera_matrix, dist_coeffs, rvec, tvec,
        axis_length_mm, 3
    )
    points = np.asarray(corners).reshape(4, 2)
    center = points.mean(axis=0).astype(int)
    x_mm, y_mm, z_mm = tvec.reshape(3)
    roll, pitch, yaw = euler

    lines = [
        f"ID {marker_id}",
        f"X={x_mm:+.1f} mm  Y={y_mm:+.1f} mm  Z={z_mm:+.1f} mm",
        f"roll={roll:+.1f}  pitch={pitch:+.1f}  yaw={yaw:+.1f} deg",
    ]
    x_text = int(center[0] + 12)
    y_text = int(center[1] - 40)
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (x_text, y_text + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                    3, cv2.LINE_AA)
        cv2.putText(frame, text, (x_text, y_text + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20),
                    1, cv2.LINE_AA)


def create_detector(dictionary_name):
    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICTIONARIES[dictionary_name]
    )
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def main():
    args = parse_args()
    if args.marker_size_mm <= 0:
        raise ValueError("--marker-size-mm deve ser maior que zero.")

    capture = open_capture(args)
    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError("A fonte abriu, mas nao entregou o primeiro frame.")

    height, width = frame.shape[:2]
    camera_matrix, dist_coeffs, calibrated = load_camera_calibration(
        args.calibration, width, height
    )
    detector = create_detector(args.dictionary)
    object_points = marker_object_points(args.marker_size_mm)
    axis_length_mm = args.axis_length_mm or args.marker_size_mm * 0.5

    csv_file = None
    csv_writer = None
    if args.csv:
        parent = os.path.dirname(os.path.abspath(args.csv))
        os.makedirs(parent, exist_ok=True)
        csv_file = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp_s", "frame", "marker_id",
            "x_mm", "y_mm", "z_mm",
            "roll_deg", "pitch_deg", "yaw_deg",
        ])

    if calibrated:
        print(f"[OK] Calibracao carregada: {args.calibration}")
    else:
        print("[AVISO] Usando intrinsecos aproximados.")
        print("[AVISO] Os valores XYZ nao devem ser usados como medicao precisa.")
    print("Eixos OpenCV: X=direita, Y=baixo na camera, Z=para frente da camera.")
    print("Pressione Q ou ESC para sair.")

    frame_index = 0
    start_time = time.perf_counter()

    try:
        while True:
            if frame_index > 0:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
            frame_index += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = detector.detectMarkers(gray)

            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(
                    frame,
                    corners,
                    ids
                )
            
                for marker_corners, marker_id_value in zip(
                    corners,
                    ids
                ):
                    marker_id = int(
                        np.asarray(
                            marker_id_value
                        ).reshape(-1)[0]
                    )
            
                    pose = estimate_marker_pose(
                        marker_corners,
                        object_points,
                        camera_matrix,
                        dist_coeffs
                    )
            
                    if pose is None:
                        continue
            
                    (
                        rvec,
                        tvec,
                        rotation_matrix,
                        euler
                    ) = pose
            
                    draw_pose_information(
                        frame,
                        marker_id,
                        marker_corners,
                        rvec,
                        tvec,
                        euler,
                        camera_matrix,
                        dist_coeffs,
                        axis_length_mm
                    )
            
                    if csv_writer is not None:
                        x_mm, y_mm, z_mm = tvec.reshape(3)
                        roll, pitch, yaw = euler
            
                        csv_writer.writerow([
                            time.perf_counter() - start_time,
                            frame_index,
                            marker_id,
                            float(x_mm),
                            float(y_mm),
                            float(z_mm),
                            float(roll),
                            float(pitch),
                            float(yaw)
                        ])
            else:
                cv2.putText(frame, "Nenhum ArUco detectado", (15, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255),
                            2, cv2.LINE_AA)

            status = "CALIBRADA" if calibrated else "INTRINSECOS APROXIMADOS"
            cv2.putText(frame, status, (15, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if calibrated else (0, 165, 255),
                        2, cv2.LINE_AA)
            cv2.imshow("ArUco - Pose XYZ", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
