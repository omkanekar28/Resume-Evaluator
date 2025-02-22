import os 
import pytesseract
import cv2
from typing import Tuple


class Preprocessor:
    """
    Handles preprocessing of various input types (text, file paths, images) 
    for the resume evaluator. Detects input type, validates it, and extracts text accordingly.
    """

    def __init__(self, input_str: str, minimum_character_count: int = 300) -> None:
        """
        Validates and initialises the input and its type (File/Text).
        """
        self.minimum_character_count = minimum_character_count
        is_valid, input_type = self.validate_input(input_str=input_str)

        # VALIDATING THE INPUT
        if not is_valid:
            raise ValueError("Incorrect filepath provided or the resume description text is too short!")
        
        self.input = input_str
        self.input_type = input_type

    def __call__(self, input_str: str, input_type: str):
        """
        Uses the input-type relevant function to return the text present inside.
        """
        if input_type == 'Text':
            return input_str
        
        pass

    def get_file_type(self, file_path: str):
        """
        Detects the file type (pdf, doc, image, other).
        """
        pass

    def image_to_text(self, image_path: str) -> str:
        """
        Uses pytesseract to detect and return the text that is present in the given image.
        """
        image = cv2.imread(image_path)

        # PREPROCESS FOR BETTER OCR ACCURACY
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        text = pytesseract.image_to_string(processed_image)
        # print("Extracted Text:\n", text)
        return text

    def validate_input(self, input_str: str) -> Tuple[bool, str]:
        """
        Checks the type of a given input and if it is valid or not.

        Returns: (is_valid: True/False, input_type: Text/File)
        """
        # FILE
        if os.path.isfile(input_str):
            return True, "File"
        
        # TEXT
        if len(input_str.strip()) >= self.minimum_character_count:
            return True, "Text"
        
        # EITHER INCORRECT FILEPATH OR TOO SHORT OF A RESUME DESCRIPTION
        return False, "Invalid"
