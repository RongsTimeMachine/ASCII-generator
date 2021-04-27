import numpy as np
from PIL import Image, ImageFont, ImageDraw, ImageOps


def sort_chars(char_list, font, language):
    """
    Sorts characters in char_list based on their brightness when rendered in a specific language.

    Args:
        char_list (str): String of characters to be sorted.
        font (ImageFont): Font used to render characters.
        language (str): Language of the characters which affects their size.

    Returns:
        str: Sorted string of characters based on brightness.
    """
    # Determine character dimensions based on language
    if language == "chinese":
        char_width, char_height = font.getsize("制")
    elif language == "korean":
        char_width, char_height = font.getsize("ㅊ")
    elif language == "japanese":
        char_width, char_height = font.getsize("あ")
    elif language in ["english", "german", "french", "spanish", "italian", "portuguese", "polish"]:
        char_width, char_height = font.getsize("A")
    elif language == "russian":
        char_width, char_height = font.getsize("A")

    # Limit the number of characters to sort
    num_chars = min(len(char_list), 100)

    # Create an image to render characters
    out_width = char_width * len(char_list)
    out_height = char_height
    out_image = Image.new("L", (out_width, out_height), 255)
    draw = ImageDraw.Draw(out_image)
    draw.text((0, 0), char_list, fill=0, font=font)

    # Crop the image to the bounding box of the text
    cropped_image = ImageOps.invert(out_image).getbbox()
    out_image = out_image.crop(cropped_image)

    # Calculate brightness of each character
    brightness = [np.mean(np.array(out_image)[:, 10 * i:10 * (i + 1)]) for i in range(len(char_list))]

    # Sort characters by brightness
    char_list = list(char_list)
    zipped_lists = zip(brightness, char_list)
    zipped_lists = sorted(zipped_lists)

    # Build the result string based on sorted brightness
    result = ""
    counter = 0
    incremental_step = (zipped_lists[-1][0] - zipped_lists[0][0]) / num_chars
    current_value = zipped_lists[0][0]

    for value, char in zipped_lists:
        if value >= current_value:
            result += char
            counter += 1
            current_value += incremental_step
        if counter == num_chars:
            break

    # Ensure the last character in the sorted list is included
    if result[-1] != zipped_lists[-1][1]:
        result += zipped_lists[-1][1]

    return result


def get_data(language, mode):
    """
    Retrieves the character list, font, sample character, and scale based on the specified language and mode.

    Args:
        language (str): The language to be used for character retrieval.
        mode (str): The mode to select specific characters from the language set.

    Returns:
        tuple: A tuple containing the character list (str), font (ImageFont), sample character (str), and scale (int).
               Returns (None, None, None, None) if the language or mode is invalid.
    """
    try:
        # Select the appropriate character set, font, sample character, and scale based on language
        if language == "general":
            from alphabets import GENERAL as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "A"
            scale = 2
        elif language == "english":
            from alphabets import ENGLISH as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "A"
            scale = 2
        elif language == "german":
            from alphabets import GERMAN as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "A"
            scale = 2
        elif language == "french":
            from alphabets import FRENCH as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "A"
            scale = 2
        elif language == "italian":
            from alphabets import ITALIAN as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "A"
            scale = 2
        elif language == "polish":
            from alphabets import POLISH as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "A"
            scale = 2
        elif language == "portuguese":
            from alphabets import PORTUGUESE as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "A"
            scale = 2
        elif language == "spanish":
            from alphabets import SPANISH as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "A"
            scale = 2
        elif language == "russian":
            from alphabets import RUSSIAN as character
            font = ImageFont.truetype("fonts/DejaVuSansMono-Bold.ttf", size=20)
            sample_character = "Ш"
            scale = 2
        elif language == "chinese":
            from alphabets import CHINESE as character
            font = ImageFont.truetype("fonts/simsun.ttc", size=10)
            sample_character = "制"
            scale = 1
        elif language == "korean":
            from alphabets import KOREAN as character
            font = ImageFont.truetype("fonts/arial-unicode.ttf", size=10)
            sample_character = "ㅊ"
            scale = 1
        elif language == "japanese":
            from alphabets import JAPANESE as character
            font = ImageFont.truetype("fonts/arial-unicode.ttf", size=10)
            sample_character = "お"
            scale = 1
        else:
            print("Invalid language")
            return None, None, None, None

        # Retrieve character list based on mode
        if len(character) > 1:
            char_list = character[mode]
        else:
            char_list = character["standard"]

        # Sort characters by brightness if not general language
        if language != "general":
            char_list = sort_chars(char_list, font, language)

        return char_list, font, sample_character, scale

    except KeyError:
        # Handle invalid mode
        print("Invalid mode for {}".format(language))
        return None, None, None, None
