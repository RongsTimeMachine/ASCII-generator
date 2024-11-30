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
    Parses command line arguments for the image to ASCII conversion with color.

    Returns:
        argparse.Namespace: A namespace containing the parsed arguments.
            - input (str): Path to input image.
            - output (str): Path to output text file.
            - language (str): Language for character retrieval.
            - mode (str): Mode for character selection.
            - background (str): Background color choice.
            - num_cols (int): Number of characters for output's width.
            - scale (int): Upsize output.
    """
    parser = argparse.ArgumentParser(description="Image to ASCII with color")
    # Input image path argument
    parser.add_argument("--input", type=str, default="data/input.jpg", help="Path to input image")
    if not parser.parse_args().input:
        raise ValueError("Input image path is required.")
    # Output text file path argument
    parser.add_argument("--output", type=str, default="data/output.jpg", help="Path to output text file")
    if not parser.parse_args().output:
        raise ValueError("Output text file path is required.")
    # Language option for character retrieval
    parser.add_argument("--language", type=str, default="english", help="Language for character retrieval")
    # Mode option for character selection
    parser.add_argument("--mode", type=str, default="standard", help="Mode for character selection")
    # Background color option
    parser.add_argument("--background", type=str, default="black", choices=["black", "white"], help="background's color")
    # Number of columns for output's width
    parser.add_argument("--num_cols", type=int, default=300, help="number of character for output's width")
    if parser.parse_args().num_cols <= 0:
        raise ValueError("Number of columns must be greater than 0.")
    # Upsize output
    parser.add_argument("--scale", type=int, default=2, help="upsize output")
    if parser.parse_args().scale <= 0:
        raise ValueError("Upsize output must be greater than 0.")
    args = parser.parse_args()
    return args


def main(opt):
    """
    Converts an image to an ASCII art representation with color.

    Args:
        opt: Command line arguments with options for input, output, language, mode, background, num_cols, and scale.
    """
    # Set background color
    bg_code = (255, 255, 255) if opt.background == "white" else (0, 0, 0)
    
    # Retrieve character list, font, sample character, and scale based on language and mode
    try:
        char_list, font, sample_character, scale = get_data(opt.language, opt.mode)
    except Exception as e:
        raise ValueError(f"Error retrieving data: {e}")
    
    num_chars = len(char_list)
    num_cols = opt.num_cols

    # Read the input image
    image = cv2.imread(opt.input, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open or find the image '{opt.input}'")
    
    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Calculate dimensions for ASCII cells
    cell_width = width / num_cols
    cell_height = scale * cell_width
    num_rows = int(height / cell_height)

    # Adjust dimensions if too many columns or rows
    if num_cols > width or num_rows > height:
        print("Too many columns or rows. Using default setting")
        cell_width = 6
        cell_height = 12
        num_cols = int(width / cell_width)
        num_rows = int(height / cell_height)

    # Calculate output image dimensions
    try:
        x0, y0, x1, y1 = font.getbbox(sample_character)
    except Exception as e:
        raise ValueError(f"Error getting character bounding box: {e}")

    char_width = x1 - x0
    char_height = y1 - y0
    out_width = char_width * num_cols
    out_height = scale * char_height * num_rows

    # Create a new image with the background code
    out_image = Image.new("RGB", (out_width, out_height), bg_code)
    draw = ImageDraw.Draw(out_image)

    # Draw ASCII characters on the output image
    for i in range(num_rows):
        for j in range(num_cols):
            # Extract the part of the image corresponding to the current cell
            partial_image = image[int(i * cell_height):min(int((i + 1) * cell_height), height),
                                  int(j * cell_width):min(int((j + 1) * cell_width), width), :]

            # Calculate the average color of the cell
            partial_avg_color = np.sum(np.sum(partial_image, axis=0), axis=0) / (cell_height * cell_width)
            partial_avg_color = tuple(partial_avg_color.astype(np.int32).tolist())

            # Select the character based on the brightness of the cell
            char = char_list[min(int(np.mean(partial_image) * num_chars / 255), num_chars - 1)]

            # Draw the character on the output image
            draw.text((j * char_width, i * char_height), char, fill=partial_avg_color, font=font)

    # Crop the output image to remove excess background
    if opt.background == "white":
        cropped_image = ImageOps.invert(out_image).getbbox()
    else:
        cropped_image = out_image.getbbox()
    
    if cropped_image:
        out_image = out_image.crop(cropped_image)

    # Save the output image
    try:
        out_image.save(opt.output)
    except Exception as e:
        raise IOError(f"Error saving the output image: {e}")


if __name__ == '__main__':
    opt = get_args()
    main(opt)
