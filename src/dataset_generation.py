import os
import time
import json
import numpy as np
import pandas as pd
from models import (
    GGUFModel, 
    GoogleGenaiModel
)
from prompts import get_label_generation_system_prompt, get_label_generation_instruction_prompt


class DatasetCompleterAutomatic:
    """
    Uses a base model to get and save resume-jd matching scores (label) 
    on the incomplete dataset.
    """

    def __init__(
        self, 
        dataset_path: str, 
        dataset_shuffle_seed: int, 
        output_path: str, 
        model_ckpt: str, 
        system_prompt: str, 
        failure_sleep_time: int = 60,
        **kwargs
    ) -> None:
        """
        Initialises the parameters needed for dataset completion.
        """
        self.model_handler = GoogleGenaiModel(
            model_name=model_ckpt,
            system_prompt=system_prompt,
            api_key=kwargs.get('api_key', None), 
            include_thoughts=kwargs.get('include_thoughts', None)
        )
        self.dataset = pd.read_excel(dataset_path)
        self.shuffle_seed = dataset_shuffle_seed
        self.failure_sleep_time = failure_sleep_time

        if self.shuffle_seed is not None:
            print(f"Shuffling dataset with seed: {self.shuffle_seed}")
            # Set random seed for reproducibility
            np.random.seed(self.shuffle_seed)
            # Shuffle the dataset
            self.dataset = self.dataset.sample(frac=1, random_state=self.shuffle_seed).reset_index(drop=True)
            print(f"Dataset shuffled. Total rows: {len(self.dataset)}")
        else:
            print("No shuffle seed provided. Processing dataset in original order.")

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
            attempt = 0
            while True:
                try:
                    attempt += 1
                    print(f"\n\nProcessing row {index + 1} out of {len(self.dataset)}... (Attempt {attempt})\n\n")
                    resume = row['Resume']
                    jd = row['JD']
                    instruction_prompt = get_label_generation_instruction_prompt(resume=resume, jd=jd)
                    inference_start_time = time.time()
                    response = self.model_handler.perform_inference(instruction_prompt=instruction_prompt)
                    print(f"Inference time taken: {(time.time() - inference_start_time):.2f} seconds")
                    print(response)
                    try:
                        response_json_str = response[response.index('{'):response.rindex('}') + 1]
                        parsed_response = json.loads(response_json_str)
                        
                        required_keys = {"summary", "match_score", "skill_match", "experience_match", "education_match", "responsibility_match", "final_assessment"}
                        if not all(key in parsed_response for key in required_keys):
                            raise ValueError(f"Response JSON missing required fields! Response: {parsed_response}")
                    except json.JSONDecodeError:
                        raise Exception(f"Invalid JSON response!")
                    
                    self.output_dict['JD'].append(jd)
                    self.output_dict['Resume'].append(resume)
                    self.output_dict['Response'].append(response)
                    self.save_current_output_dict()
                    break  # Success, move to next row

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Row {index + 1} failed (Attempt {attempt}): {str(e)}")
                    print(f"Retrying in {self.failure_sleep_time} seconds...")
                    time.sleep(self.failure_sleep_time)


if __name__ == '__main__':
    print("********************************************************************************************")
    print("*                                    DATASET GENERATION                                    *")
    print("********************************************************************************************")
    dataset_completer = DatasetCompleterAutomatic(
        dataset_path=r"path\to\Resume-Evaluator\data\dataset_without_labels(Data Scientist).xlsx",
        dataset_shuffle_seed=None,  # Set to None for no shuffling
        output_path=r"path\to\Resume-Evaluator\data\dataset_complete(Data Scientist).xlsx",
        model_ckpt="gemma-4-26b-a4b-it",
        system_prompt=get_label_generation_system_prompt(),
        context_window_size=8000, 
        api_key="", 
        include_thoughts=True, 
        failure_sleep_time=60
    )
    dataset_completer()
