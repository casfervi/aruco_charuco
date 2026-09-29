# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:59:31 2026

@author: vinicius.ferreira
"""

import argparse

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera um tabuleiro ChArUco."
    )

    parser.add_argument(
        "--squares-x",
        type=int,
        default=7,
        help="Quantidade de quadrados na horizontal."
    )

    parser.add_argument(
        "--squares-y",
        type=int,
        default=10,
        help="Quantidade de quadrados na vertical."
    )

    parser.add_argument(
        "--square-mm",
        type=float,
        default=30.0,
        help="Lado real de cada quadrado, em mm."
    )

    parser.add_argument(
        "--marker-mm",
        type=float,
        default=22.0,
        help="Lado real do marcador ArUco, em mm."
    )

    parser.add_argument(
        "--pixels-per-mm",
        type=float,
        default=10.0,
        help="Resolucao da imagem gerada."
    )

    parser.add_argument(
        "--output",
        default="markers/charuco_board.png",
        help="Arquivo PNG de saida."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.marker_mm >= args.square_mm:
        raise ValueError(
            "--marker-mm deve ser menor que --square-mm."
        )

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_5X5_250
    )

    board = cv2.aruco.CharucoBoard(
        (
            args.squares_x,
            args.squares_y
        ),
        args.square_mm,
        args.marker_mm,
        dictionary
    )

    board_width_mm = (
        args.squares_x *
        args.square_mm
    )

    board_height_mm = (
        args.squares_y *
        args.square_mm
    )

    image_width = int(round(
        board_width_mm *
        args.pixels_per_mm
    ))

    image_height = int(round(
        board_height_mm *
        args.pixels_per_mm
    ))

    margin_pixels = int(round(
        10 * args.pixels_per_mm
    ))

    board_image = board.generateImage(
        (
            image_width,
            image_height
        ),
        marginSize=margin_pixels,
        borderBits=1
    )

    success = cv2.imwrite(
        args.output,
        board_image
    )

    if not success:
        raise RuntimeError(
            f"Nao foi possivel salvar: {args.output}"
        )

    print("")
    print("[OK] Tabuleiro gerado")
    print(f"Arquivo: {args.output}")
    print(
        f"Quadrados: "
        f"{args.squares_x} x {args.squares_y}"
    )
    print(
        f"Quadrado: {args.square_mm:.2f} mm"
    )
    print(
        f"Marcador: {args.marker_mm:.2f} mm"
    )
    print(
        f"Area fisica: "
        f"{board_width_mm:.2f} x "
        f"{board_height_mm:.2f} mm"
    )
    print(
        f"Imagem: "
        f"{image_width} x {image_height} px"
    )


if __name__ == "__main__":
    main()