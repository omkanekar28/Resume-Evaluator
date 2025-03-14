import pandas as pd
from models import GGUFModel
from prompts import get_label_generation_prompt


class DatasetCompleterAutomatic:
    """
    Uses a base model to get and save resume-jd matching scores (label) 
    on the incomplete dataset.
    """

    def __init__(self, dataset_path: str, output_path: str, gguf_model_path: str, system_prompt: str, context_window_size) -> None:
        """
        Initialises the parameters needed for dataset completion.
        """
        self.model_handler = GGUFModel(
            gguf_model_path=gguf_model_path, 
            system_prompt=system_prompt, 
            context_window_size=context_window_size
        )
        self.dataset = pd.read_excel(dataset_path)
        self.output_path = output_path

    def fill_rows(self) -> None:
        """
        Uses the specified model to predict and validate the output and 
        store it in the excel file.
        """
        for index, row in self.dataset.iterrows():
            try:
                print(f"\n\nProcessing row {index + 1} out of {len(self.dataset)}...\n\n")
                instruction_prompt = get_label_generation_prompt(resume=row['Resume'], jd=row['JD'])
                response = self.model_handler.perform_inference(instruction_prompt=instruction_prompt)
                # TODO: Validation followed by saving of excel.
            except Exception as e:
                print(f"Skipping row {index+1}: {str(e)}")
                continue


if __name__ == '__main__':
    dataset_completer = DatasetCompleterAutomatic()
    dataset_completer.fill_rows()
