def fancy_print(text: str, padding_spaces: int = 6) -> None:
    """
    Prints the given text in a decorative box.
    """
    border_length = len(text) + padding_spaces  # Adjust for padding and borders
    print('*' * border_length)
    surrounding_empty_spaces = int((padding_spaces - 2) / 2)
    print(f"*{' ' * surrounding_empty_spaces}{text}{' ' * surrounding_empty_spaces}*")
    print('*' * border_length)