def fancy_print(text: str) -> None:
    """
    Prints the given text in a decorative box.
    """
    border_length = len(text) + 6  # Adjust for padding and borders
    print('*' * border_length)
    print(f"*  {text}  *")
    print('*' * border_length)
