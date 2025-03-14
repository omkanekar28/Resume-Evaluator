import os
import pandas as pd
from preprocessing import Preprocessor


class ResumeFormatter:
    """
    Class that handles operations related to formatting the given input resume files into 
    structured excel format.
    """

    def __init__(self, input_dir: str, output_store_path: str) -> None:
        """
        Initialises directories and classes that will be used during resume formatting.
        """
        self.input_dir = input_dir
        self.preprocessor = Preprocessor()
        self.output_dict = {
            "filename": [],
            "text": [],
        }
        self.output_store_path = output_store_path

    def save_current_output_dict(self) -> None:
        """
        Saves current version of the output as an excel file in the provided directory.
        """
        output_df = pd.DataFrame(self.output_dict)
        output_df.to_excel(self.output_store_path, index=False)
    
    def __call__(self) -> None:
        """
        Iterates through the files, extracts text and stores the results as 
        an excel file in the specified output directory.
        """
        for count, filename in enumerate(os.listdir(self.input_dir)):
            try:
                print(f"\n\nProcessing file {count + 1} out of {len(os.listdir(self.input_dir))}...\n\n")
                
                filepath = os.path.join(self.input_dir, filename)
                text = self.preprocessor(input_str=filepath)
                
                self.output_dict['filename'].append(filename)
                self.output_dict['text'].append(text)
                self.save_current_output_dict()

                print(f"file {count + 1} processed successfully")
            except Exception as e:
                print(f"An error occured while trying to process file {count + 1}: {str(e)}")
                continue


if __name__ == '__main__':
    INPUT_DIR = "/home/om/code/Resume-Evaluator/data/resumes"
    OUTPUT_STORE_PATH = "files_dataset.xlsx"
    resume_formatter = ResumeFormatter(
        input_dir=INPUT_DIR,
        output_store_path=OUTPUT_STORE_PATH
    )
    resume_formatter()
