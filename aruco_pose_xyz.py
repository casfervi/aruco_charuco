# -*- coding: utf-8 -*-
"""Detecta marcadores ArUco, estima pose 3D e desenha os eixos X, Y e Z.

Exemplos:
    python aruco_pose_xyz.py --camera 0 --marker-size-mm 50
    python aruco_pose_xyz.py --video teste.mp4 --marker-size-mm 50
    python aruco_pose_xyz.py --video teste.mp4 --marker-size-mm 50 --scale 1.0
    python aruco_pose_xyz.py --camera 0 --marker-size-mm 50 --calibration camera_calibration.npz

O arquivo NPZ de calibracao deve conter:
    camera_matrix: matriz intrinseca 3x3
    dist_coeffs: coeficientes de distorcao

Controles no modo --video:
    barra "Frame"        arrastar para ir a qualquer ponto do video
    ESPACO               pausa / continua
    D ou seta direita    avanca 1 frame (pausa)
    A ou seta esquerda   volta 1 frame (pausa)
    L                    avanca ~1 segundo
    J                    volta ~1 segundo
    Home / End           vai ao inicio / ao fim
    Q ou ESC             sair
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

# Codigos de teclas especiais retornados por cv2.waitKeyEx
# (Windows / GTK no Linux).
KEYS_LEFT = {2424832, 65361}
KEYS_RIGHT = {2555904, 65363}
KEYS_HOME = {2359296, 65360}
KEYS_END = {2293760, 65367}

TRACKBAR_NAME = "Frame"


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
    parser.add_argument("--scale", type=float, default=None,
                        help="Escala de exibicao. 1.0 = tamanho original. "
                             "Padrao: ajusta para caber em 1600x900 sem ampliar. "
                             "Nao afeta a deteccao, so a janela.")
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
                          camera_matrix, dist_coeffs, axis_length_mm,
                          ui_scale=1.0):
    """ui_scale > 1 engrossa linhas e texto para continuarem legiveis
    quando o frame em resolucao total e reduzido na exibicao."""
    cv2.drawFrameAxes(
        frame, camera_matrix, dist_coeffs, rvec, tvec,
        axis_length_mm, max(1, int(round(3 * ui_scale)))
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
    font_scale = 0.55 * ui_scale
    x_text = int(center[0] + 12 * ui_scale)
    y_text = int(center[1] - 40 * ui_scale)
    line_step = int(22 * ui_scale)
    thick_outer = max(1, int(round(3 * ui_scale)))
    thick_inner = max(1, int(round(1 * ui_scale)))
    for index, text in enumerate(lines):
        position = (x_text, y_text + index * line_step)
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thick_outer, cv2.LINE_AA)
        cv2.putText(frame, text, position,
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 20, 20),
                    thick_inner, cv2.LINE_AA)


def create_detector(dictionary_name):
    dictionary = cv2.aruco.getPredefinedDictionary(
        ARUCO_DICTIONARIES[dictionary_name]
    )
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def annotate_frame(raw_frame, detector, object_points, camera_matrix,
                   dist_coeffs, axis_length_mm, ui_scale=1.0):
    """Detecta ArUcos, desenha poses e devolve (imagem, lista de poses)."""
    frame = raw_frame.copy()
    poses = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for marker_corners, marker_id_value in zip(corners, ids):
            marker_id = int(np.asarray(marker_id_value).reshape(-1)[0])
            pose = estimate_marker_pose(
                marker_corners, object_points, camera_matrix, dist_coeffs
            )
            if pose is None:
                continue
            rvec, tvec, _rotation_matrix, euler = pose
            draw_pose_information(
                frame, marker_id, marker_corners, rvec, tvec, euler,
                camera_matrix, dist_coeffs, axis_length_mm, ui_scale
            )
            poses.append((marker_id, tvec.reshape(3), euler))
    else:
        cv2.putText(frame, "Nenhum ArUco detectado",
                    (int(15 * ui_scale), int(35 * ui_scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8 * ui_scale, (0, 0, 255),
                    max(1, int(round(2 * ui_scale))), cv2.LINE_AA)
    return frame, poses


def draw_status(frame, calibrated, is_video, index, total, paused):
    height = frame.shape[0]
    status = "CALIBRADA" if calibrated else "INTRINSECOS APROXIMADOS"
    cv2.putText(frame, status, (15, height - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 0) if calibrated else (0, 165, 255),
                2, cv2.LINE_AA)
    if is_video:
        position = f"frame {index + 1}/{total}" if total > 0 else f"frame {index + 1}"
        text = f"{position}  {'PAUSADO' if paused else 'REPRODUZINDO'}"
        cv2.putText(frame, text, (15, height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                    3, cv2.LINE_AA)
        cv2.putText(frame, text, (15, height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20),
                    1, cv2.LINE_AA)


def main():
    args = parse_args()
    if args.marker_size_mm <= 0:
        raise ValueError("--marker-size-mm deve ser maior que zero.")

    capture = open_capture(args)
    ok, raw_frame = capture.read()
    if not ok or raw_frame is None:
        capture.release()
        raise RuntimeError("A fonte abriu, mas nao entregou o primeiro frame.")

    is_video = bool(args.video)
    height, width = raw_frame.shape[:2]
    if args.scale is not None and args.scale <= 0:
        raise ValueError("--scale deve ser maior que zero.")
    # Escala so da exibicao: a deteccao sempre usa o frame em resolucao total.
    display_scale = args.scale if args.scale else min(
        1.0, 1600.0 / width, 900.0 / height
    )
    ui_scale = max(1.0, 1.0 / display_scale)
    camera_matrix, dist_coeffs, calibrated = load_camera_calibration(
        args.calibration, width, height
    )
    detector = create_detector(args.dictionary)
    object_points = marker_object_points(args.marker_size_mm)
    axis_length_mm = args.axis_length_mm or args.marker_size_mm * 0.5

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 0
    fps = capture.get(cv2.CAP_PROP_FPS) if is_video else 0.0
    if not fps or fps != fps or fps <= 0:
        fps = 30.0
    jump_frames = max(1, int(round(fps)))  # ~1 segundo

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
    print(f"Fonte: {width}x{height} px | exibicao a {display_scale:.2f}x "
          "(deteccao sempre em resolucao total).")
    if is_video:
        print("Video: barra 'Frame' | ESPACO pausa | A/D (ou setas) +-1 frame | "
              "J/L +-1 s | Home/End | Q ou ESC sai.")
    else:
        print("Pressione Q ou ESC para sair.")

    window_name = "ArUco - Pose XYZ"
    # AUTOSIZE: a janela mostra a imagem pixel a pixel, sem reamostragem do
    # backend. A reducao (se houver) e feita por nos com INTER_AREA.
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Estado do player (so faz sentido em video)
    state = {"index": 0, "seek": None}

    if is_video and total > 1:
        def on_trackbar(value):
            # Ignora o callback disparado pelo nosso proprio setTrackbarPos
            if value != state["index"]:
                state["seek"] = value

        cv2.createTrackbar(TRACKBAR_NAME, window_name, 0, total - 1, on_trackbar)

    index = 0
    paused = False
    need_process = True
    display = None
    logged_frames = set()  # evita linhas duplicadas no CSV ao revisitar frames
    start_time = time.perf_counter()

    try:
        while True:
            if need_process:
                display, poses = annotate_frame(
                    raw_frame, detector, object_points, camera_matrix,
                    dist_coeffs, axis_length_mm, ui_scale
                )
                if csv_writer is not None and index not in logged_frames:
                    logged_frames.add(index)
                    timestamp = (index / fps) if is_video \
                        else (time.perf_counter() - start_time)
                    for marker_id, tvec, euler in poses:
                        csv_writer.writerow([
                            timestamp, index + 1, marker_id,
                            float(tvec[0]), float(tvec[1]), float(tvec[2]),
                            float(euler[0]), float(euler[1]), float(euler[2]),
                        ])
                need_process = False

            if abs(display_scale - 1.0) > 1e-6:
                interpolation = (cv2.INTER_AREA if display_scale < 1.0
                                 else cv2.INTER_CUBIC)
                shown = cv2.resize(display, None, fx=display_scale,
                                   fy=display_scale,
                                   interpolation=interpolation)
            else:
                shown = display.copy()
            draw_status(shown, calibrated, is_video, index, total, paused)
            cv2.imshow(window_name, shown)

            if is_video and total > 1:
                state["index"] = index
                cv2.setTrackbarPos(TRACKBAR_NAME, window_name, index)

            key_ex = cv2.waitKeyEx(30 if paused else 1)
            key = key_ex & 0xFF if key_ex != -1 else -1

            if key in (ord("q"), ord("Q"), 27):
                break

            target = None  # indice absoluto do proximo frame a mostrar

            if is_video:
                if key == ord(" "):
                    paused = not paused
                elif key in (ord("d"), ord("D")) or key_ex in KEYS_RIGHT:
                    paused = True
                    target = index + 1
                elif key in (ord("a"), ord("A")) or key_ex in KEYS_LEFT:
                    paused = True
                    target = index - 1
                elif key in (ord("l"), ord("L")):
                    target = index + jump_frames
                elif key in (ord("j"), ord("J")):
                    target = index - jump_frames
                elif key_ex in KEYS_HOME:
                    target = 0
                elif key_ex in KEYS_END and total > 0:
                    target = total - 1

                if state["seek"] is not None:  # barra de tempo
                    target = state["seek"]
                    state["seek"] = None

            if target is None and not paused:
                target = index + 1  # reproducao normal

            if target is None:
                continue

            if total > 0:
                target = max(0, min(target, total - 1))
            else:
                target = max(0, target)

            if target == index:
                # Chegou ao fim do video: fica pausado no ultimo frame.
                if is_video and not paused and total > 0 and index >= total - 1:
                    paused = True
                continue

            if target == index + 1:
                ok, new_frame = capture.read()  # leitura sequencial (rapida)
            else:
                capture.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, new_frame = capture.read()

            if ok and new_frame is not None:
                raw_frame = new_frame
                index = target
                need_process = True
            else:
                if is_video:
                    # Fim inesperado (contagem de frames imprecisa): pausa
                    # e deixa o usuario voltar.
                    paused = True
                    total = index + 1
                    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                else:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
