import os
import json
import pandas as pd
from prompts import get_label_generation_instruction_prompt, get_label_generation_system_prompt

input_df = pd.read_excel(r"C:\Users\Om\code\Resume-Evaluator\data\dataset_complete(DataScience-WebDesigning).xlsx")
print(f"Loaded dataset from {r"C:\Users\Om\code\Resume-Evaluator\data\dataset_complete(DataScience-WebDesigning).xlsx"} with shape {input_df.shape}")
output_data = []
os.makedirs(os.path.dirname(r"C:\Users\Om\code\Resume-Evaluator\data\dataset_complete(DataScience-WebDesigning).json"), exist_ok=True)

for idx, row in input_df.iterrows():
    try:
        print(f"Processing row {idx + 1} / {len(input_df)} ...")
        system_prompt = get_label_generation_system_prompt()
        user_prompt = get_label_generation_instruction_prompt(
            resume=row["Resume"],
            jd=row["JD"]
        )
        assistant_prompt = row["Response"]

        output_data.append({
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_prompt}
            ]
        })
    except Exception as e:
        print(f"Error processing row {idx + 1}: {e}. Skipping ...")

print(f"Saving converted dataset to {r"C:\Users\Om\code\Resume-Evaluator\data\dataset_complete(DataScience-WebDesigning).json"} ...")
with open(r"C:\Users\Om\code\Resume-Evaluator\data\dataset_complete(DataScience-WebDesigning).json", "w") as f:
    json.dump(output_data, f, indent=4)
print(f"Dataset successfully saved to {r"C:\Users\Om\code\Resume-Evaluator\data\dataset_complete(DataScience-WebDesigning).json"} with {len(output_data)} conversations!")