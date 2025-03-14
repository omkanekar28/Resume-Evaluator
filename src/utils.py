import json
import pyfiglet    

def get_json_data(filepath: str) -> dict:
    """
    Returns the data present in given JSON file as dict.
    """
    with open(filepath, 'r') as file:
        return json.load(file)

def fancy_print(text: str) -> None:
    """
    Uses pyfiglet library to print given text in a fancy manner.
    """
    ascii_art = pyfiglet.figlet_format(text, font='slant')
    print(ascii_art)