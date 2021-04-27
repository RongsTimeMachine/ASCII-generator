"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from utils import get_data


def get_args():
    """
    Parses command line arguments for the image to ASCII conversion.

    Returns:
        argparse.Namespace: A namespace containing the parsed arguments.
            - input (str): Path to input image.
            - output (str): Path to output text file.
            - language (str): Language for character retrieval.
            - mode (str): Mode for character selection.
            - background (str): Background color choice.
            - num_cols (int): Number of characters for output's width.
    """
    parser = argparse.ArgumentParser(description="Image to ASCII")

    # Input image path argument
    parser.add_argument("--input", type=str, default="data/input.jpg", help="Path to input image")

    # Output text file path argument
    parser.add_argument("--output", type=str, default="data/output.jpg", help="Path to output text file")

    # Language option for character retrieval
    parser.add_argument("--language", type=str, default="english", help="Language for character retrieval")

    # Mode option for character selection
    parser.add_argument("--mode", type=str, default="standard", help="Mode for character selection")

    # Background color option
    parser.add_argument("--background", type=str, default="black", choices=["black", "white"], help="Background color")

    # Number of columns for output's width
    parser.add_argument("--num_cols", type=int, default=300, help="Number of characters for output's width")

    args = parser.parse_args()
    return args


def main(opt):
    """
    Converts an image to an ASCII art representation.

    Args:
        opt: Command line arguments with options for input, output, language, mode, background, num_cols, and scale.
    """
    if opt.background == "white":
        # White background
        bg_code = 255
    else:
        # Black background
        bg_code = 0

    # Retrieve character list, font, sample character, and scale based on language and mode
    char_list, font, sample_character, scale = get_data(opt.language, opt.mode)
    num_chars = len(char_list)
    num_cols = opt.num_cols

    # Read and convert the input image to grayscale
    image = cv2.imread(opt.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of the image
    height, width = image.shape

    # Calculate the dimensions for ASCII cells
    cell_width = width / opt.num_cols
    cell_height = scale * cell_width
    num_rows = int(height / cell_height)

    # If too many columns or rows, use default settings
    if num_cols > width or num_rows > height:
        print("Too many columns or rows. Use default setting")
        cell_width = 6
        cell_height = 12
        num_cols = int(width / cell_width)
        num_rows = int(height / cell_height)

    # Get the dimensions of the characters
    char_width, char_height = font.getsize(sample_character)

    # Calculate the output image dimensions
    out_width = char_width * num_cols
    out_height = scale * char_height * num_rows

    # Create an image with the background color
    out_image = Image.new("L", (out_width, out_height), bg_code)
    draw = ImageDraw.Draw(out_image)

    # Iterate through the rows and columns of the image
    for i in range(num_rows):
        # Create a line of characters for the current row
        line = "".join([char_list[min(int(np.mean(image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                                                  int(j * cell_width):min(int((j + 1) * cell_width),
                                                                          width)]) / 255 * num_chars), num_chars - 1)]
                        for j in
                        range(num_cols)]) + "\n"

        # Draw the line of characters on the output image
        draw.text((0, i * char_height), line, fill=255 - bg_code, font=font)

    # Get the bounding box of the text
    if opt.background == "white":
        cropped_image = ImageOps.invert(out_image).getbbox()
    else:
        cropped_image = out_image.getbbox()

    # Crop the output image to the bounding box of the text
    out_image = out_image.crop(cropped_image)

    # Save the output image
    out_image.save(opt.output)


if __name__ == '__main__':
    opt = get_args()
    main(opt)
