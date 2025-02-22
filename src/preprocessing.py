import os 
import pytesseract
import cv2
from typing import Tuple
from utils import fancy_print


class Preprocessor:
    """
    Handles preprocessing of various input types (text, file paths, images) 
    for the resume evaluator. Detects input type, validates it, and extracts text accordingly.
    """

    def __init__(self, minimum_character_count: int = 300) -> None:
        """
        Initialises constants to be used in preprocessing.
        """
        self.minimum_character_count = minimum_character_count
        self.file_extensions = {
            'Image': ('.jpg', '.jpeg', '.png', '.webp'),
            'Docx': ('.docx',),
            'PDF': ('.pdf',)
        }

    def __call__(self, input_str: str) -> str:
        """
        Validates and initialises the input and its type (File/Text). 
        Uses the input-type relevant function to return the text present inside.
        """
        fancy_print("Performing input preprocessing...")
        is_valid, input_type = self.validate_input(input_str=input_str)

        # VALIDATING THE INPUT
        if not is_valid:
            raise ValueError("Incorrect filepath provided or the resume description text is too short!")
        
        self.input = input_str
        self.input_type = input_type
        print(f"Detected input type: {self.input_type}")

        if self.input_type == 'Text':
            return self.input
        
        is_supported, self.file_type = self.validate_file_type(file_path=self.input)

        if not is_supported:
            raise TypeError(f"Unsupported file format: {self.file_type}!")

        print(f"Detected file type: {self.file_type}")
        if self.file_type == 'PDF':
            return self.pdf_to_text(pdf_path=self.input)
        
        if self.file_type == 'Image':
            return self.image_to_text(image_path=self.input)
        
        if self.file_type == 'Docx':
            return self.docx_to_text(wordx_path=self.input)

    def validate_file_type(self, file_path: str) -> Tuple[bool, str]:
        """
        Checks the type of file and if it is valid or not.

        Returns: (is_valid: True/False, file_type: PDF/Image/Docx/Other)
        """
        _, extension = os.path.splitext(file_path.lower())

        for file_type, extensions in self.file_extensions.items():
            if extension in extensions:
                return True, file_type

        return False, extension

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
    
    def pdf_to_text(self, pdf_path: str) -> str:
        """
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
        
        print("Performing OCR using PyTesseract...")
        text = pytesseract.image_to_string(processed_image)
        # print("Extracted Text:\n", text)
        return text
        
    def docx_to_text(self, wordx_path: str) -> str:
        """
        """
        pass


if __name__ == '__main__':
    ##############################
    # USE BELOW CODE FOR TESTING #
    ##############################
    input = """"""
    preprocessor = Preprocessor()
    print(preprocessor(input_str=input))