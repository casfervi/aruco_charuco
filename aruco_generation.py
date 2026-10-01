import argparse
import os

import cv2 as cv
from cv2 import aruco


DICTIONARIES = {
    "DICT_4X4_50": aruco.DICT_4X4_50,
    "DICT_4X4_100": aruco.DICT_4X4_100,
    "DICT_4X4_250": aruco.DICT_4X4_250,
    "DICT_5X5_50": aruco.DICT_5X5_50,
    "DICT_5X5_100": aruco.DICT_5X5_100,
    "DICT_5X5_250": aruco.DICT_5X5_250,
    "DICT_6X6_50": aruco.DICT_6X6_50,
    "DICT_6X6_100": aruco.DICT_6X6_100,
    "DICT_6X6_250": aruco.DICT_6X6_250,
    "DICT_7X7_50": aruco.DICT_7X7_50,
    "DICT_7X7_100": aruco.DICT_7X7_100,
    "DICT_7X7_250": aruco.DICT_7X7_250,
    "DICT_ARUCO_ORIGINAL": aruco.DICT_ARUCO_ORIGINAL,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera imagens de marcadores ArUco."
    )

    parser.add_argument(
        "--num-markers",
        type=int,
        default=4,
        help="Quantidade de marcadores que serao gerados (padrao: 4).",
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="ID inicial dos marcadores (padrao: 0).",
    )

    parser.add_argument(
        "--marker-size",
        type=int,
        default=400,
        help="Tamanho da imagem de cada marcador em pixels (padrao: 400).",
    )

    parser.add_argument(
        "--dictionary",
        choices=sorted(DICTIONARIES),
        default="DICT_5X5_250",
        help="Dicionario ArUco utilizado (padrao: DICT_5X5_250).",
    )

    parser.add_argument(
        "--output-dir",
        default="markers",
        help="Pasta de saida das imagens (padrao: markers).",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra cada marcador gerado em uma janela.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.num_markers <= 0:
        raise ValueError("--num-markers deve ser maior que zero.")

    if args.start_id < 0:
        raise ValueError("--start-id nao pode ser negativo.")

    if args.marker_size <= 0:
        raise ValueError("--marker-size deve ser maior que zero.")

    dictionary_id = DICTIONARIES[args.dictionary]
    marker_dict = aruco.getPredefinedDictionary(dictionary_id)

    dictionary_capacity = marker_dict.bytesList.shape[0]
    final_id = args.start_id + args.num_markers - 1

    if final_id >= dictionary_capacity:
        raise ValueError(
            f"O dicionario {args.dictionary} possui IDs de 0 a "
            f"{dictionary_capacity - 1}. O ID final solicitado seria {final_id}."
        )

    os.makedirs(args.output_dir, exist_ok=True)

    generated_count = 0

    for marker_id in range(args.start_id, final_id + 1):
        marker_image = aruco.generateImageMarker(
            marker_dict,
            marker_id,
            args.marker_size,
        )

        output_path = os.path.join(
            args.output_dir,
            f"marker_{marker_id}_{args.dictionary}.png",
        )

        if not cv.imwrite(output_path, marker_image):
            raise RuntimeError(f"Nao foi possivel salvar: {output_path}")

        generated_count += 1
        print(f"[OK] ID {marker_id}: {output_path}")

        if args.show:
            cv.imshow(f"Marker {marker_id}", marker_image)
            key = cv.waitKey(0) & 0xFF

            if key == 27:
                print("[INFO] Geracao interrompida pela tecla ESC.")
                break

    cv.destroyAllWindows()

    print("")
    print(f"Marcadores gerados: {generated_count}")
    print(f"Dicionario: {args.dictionary}")
    print(f"Pasta de saida: {args.output_dir}")


if __name__ == "__main__":
    main()
