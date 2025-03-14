import os
import time
import json
import pandas as pd
from models import GGUFModel
from prompts import get_label_generation_system_prompt, get_label_generation_instruction_prompt


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
        
        self.output_store_path = output_path

        if os.path.exists(self.output_store_path):
            self.output_dict = pd.read_excel(self.output_store_path).to_dict(orient="list")
            self.starting_index = len(self.output_dict['JD'])
            print(f"Resuming from row {self.starting_index + 1}...")
        else:
            self.output_dict = {
                'JD': [],
                'Resume': [],
                'Response': []
            }
            self.starting_index = 0

    def save_current_output_dict(self) -> None:
        """
        Saves current version of the output as an excel file in the provided directory.
        """
        output_df = pd.DataFrame(self.output_dict)
        output_df.to_excel(self.output_store_path, index=False)

    def __call__(self) -> None:
        """
        Uses the specified model to predict and validate the output and 
        store it in the excel file.
        """
        for index, row in self.dataset.iloc[self.starting_index:].iterrows():
            try:
                print(f"\n\nProcessing row {index + 1} out of {len(self.dataset)}...\n\n")
                resume = row['Resume']
                jd = row['JD']
                instruction_prompt = get_label_generation_instruction_prompt(resume=resume, jd=jd)
                inference_start_time = time.time()
                response = self.model_handler.perform_inference(instruction_prompt=instruction_prompt)
                print(f"Inference time taken: {(time.time() - inference_start_time):.2f} seconds")
                print(response)
                try:
                    parsed_response = json.loads(response)
                    
                    required_keys = ["match_score", "summary", "skill_match", "experience_match"]
                    if not all(key in parsed_response for key in required_keys):
                        raise ValueError(f"Response JSON missing required fields! Response: {parsed_response}")
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response!")
                
                self.output_dict['JD'].append(jd)
                self.output_dict['Resume'].append(resume)
                self.output_dict['Response'].append(response)
                self.save_current_output_dict()
            except Exception as e:
                print(f"Skipping row {index+1}: {str(e)}")
                continue


if __name__ == '__main__':
    dataset_completer = DatasetCompleterAutomatic(
        dataset_path="/home/om/code/Resume-Evaluator/data/dataset_without_labels.xlsx",
        output_path="dataset.xlsx",
        gguf_model_path="",
        system_prompt=get_label_generation_system_prompt(),
        context_window_size=8000
    )
    dataset_completer()
