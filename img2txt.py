"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse

import cv2
import numpy as np


def get_args():
    """
    Gets command line arguments for the image to ASCII conversion.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser("Image to ASCII")
    parser.add_argument("--input", type=str, default="data/input.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="data/output.txt", help="Path to output text file")
    parser.add_argument("--mode", type=str, default="complex", choices=["simple", "complex"], help="10 or 70 different characters")
    parser.add_argument("--num_cols", type=int, default=150, help="number of character for output's width")
    args = parser.parse_args()
    return args


def main(opt):
    """
    Converts an image to an ASCII art representation and saves it to a text file.

    Args:
        opt: Command line arguments with options for input, output, mode, and num_cols.
    """
    # Define character list based on mode
    if opt.mode == "simple":
        CHAR_LIST = '@%#*+=-:. '
    else:
        CHAR_LIST = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
    
    num_chars = len(CHAR_LIST)
    num_cols = opt.num_cols

    # Read and convert the input image to grayscale
    image = cv2.imread(opt.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape

    # Calculate dimensions for ASCII cells
    cell_width = width / opt.num_cols
    cell_height = 2 * cell_width
    num_rows = int(height / cell_height)

    # Adjust dimensions if too many columns or rows
    if num_cols > width or num_rows > height:
        print("Too many columns or rows. Use default setting")
        cell_width = 6
        cell_height = 12
        num_cols = int(width / cell_width)
        num_rows = int(height / cell_height)

    # Open the output file for writing
    output_file = open(opt.output, 'w')

    # Process each cell in the image to determine the best matching ASCII character
    for i in range(num_rows):
        for j in range(num_cols):
            # Extract the part of the image corresponding to the current cell
            partial_image = image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                                  int(j * cell_width):min(int((j + 1) * cell_width), width)]
            # Calculate the average brightness of the cell
            avg_brightness = np.mean(partial_image)
            # Determine the character index based on brightness
            char_index = min(int(avg_brightness * num_chars / 255), num_chars - 1)
            # Write the corresponding character to the output file
            output_file.write(CHAR_LIST[char_index])
        # Write a newline after each row
        output_file.write("\n")

    # Close the output file
    output_file.close()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
