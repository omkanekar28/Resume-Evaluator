import pandas as pd


class DuplicateRemover:
    """
    Class that handles detection and removal of duplicate rows from an Excel file.
    """

    def __init__(self, filepath: str) -> None:
        """
        Initialises the filepath to the target Excel file.
        """
        self.filepath = filepath

    def __call__(self) -> None:
        """
        Reads the Excel file, detects and removes duplicate rows,
        then overwrites the original file.
        """
        try:
            df = pd.read_excel(self.filepath)
            original_count = len(df)

            duplicate_count = df.duplicated().sum()
            print(f"\n{duplicate_count} duplicate(s) detected.\n")

            df_cleaned = df.drop_duplicates()
            df_cleaned.to_excel(self.filepath, index=False)

            print(f"Done. {original_count - len(df_cleaned)} row(s) removed. File saved to: {self.filepath}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    FILEPATH = r"dataset_without_labels.xlsx"
    remover = DuplicateRemover(filepath=FILEPATH)
    remover()