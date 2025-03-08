import time
import pandas as pd

df = pd.read_csv("/home/om/code/Resume-Evaluator/data/Developer-Resume-Text.csv")

print(len(df))

df = df.dropna()
df = df.drop_duplicates()

print(len(df))
df.to_excel('text_dataset.xlsx', index=False)