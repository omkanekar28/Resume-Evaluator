import pyfiglet

def fancy_print(text: str, font: str = 'slant') -> None:
    """
    Prints the given text in a fancy format using pyfiglet library.
    """
    pyfiglet.print_figlet(text=text, font=font)