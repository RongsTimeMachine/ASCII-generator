"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse

import cv2
import numpy as np


def get_args() -> argparse.Namespace:
    """Gets command line arguments for the image to ASCII conversion.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image to ASCII")
    parser.add_argument("--input", type=str, default="data/input.jpg", help="path to input image")
    parser.add_argument("--output", type=str, default="data/output.txt", help="path to output text file")
    parser.add_argument("--mode", type=str, default="complex", choices=["simple", "complex"], help="mode for character selection")
    parser.add_argument("--num_cols", type=int, default=150, help="number of characters for output's width")
    return parser.parse_args()




def main(opt):
    """
    Converts an image to an ASCII art representation and saves it to a text file.

    Args:
        opt: Command line arguments with options for input, output, mode, and num_cols.
    """
    char_list = get_char_list(opt.mode)
    num_chars = len(char_list)
    num_cols = opt.num_cols

    image = cv2.imread(opt.input, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape

    cell_width = width / num_cols
    cell_height = 2 * cell_width
    num_rows = int(height / cell_height)

    if num_cols > width or num_rows > height:
        cell_width = 6
        cell_height = 12
        num_cols = int(width / cell_width)
        num_rows = int(height / cell_height)

    with open(opt.output, 'w') as output_file:
        for i in range(0, height, int(cell_height)):
            for j in range(0, width, int(cell_width)):
                avg_brightness = np.mean(image[i:i + int(cell_height), j:j + int(cell_width)])
                char_index = min(int(avg_brightness * num_chars / 255), num_chars - 1)
                output_file.write(char_list[char_index])
            output_file.write("\n")


def get_char_list(mode: str) -> str:
    """Returns a character list based on the specified mode.

    Args:
        mode (str): The mode for character selection.

    Returns:
        str: The character list.
    """
    if mode == "simple":
        chars = '@%#*+=-:. '
    else:
        chars = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    return chars


if __name__ == '__main__':
    opt = get_args()
    main(opt)
