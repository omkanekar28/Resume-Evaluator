import pandas as pd


class JDResumeCombiner:
    """
    Class for handling the combining of all resumes and JDs to form the dataset.
    """

    def __init__(self, jd_excel_filepath: str, resume_excel_filepath: str, output_store_path: str) -> None:
        """
        Initialises the JDs, Resumes and relevant directories.
        """
        self.jd_excel_filepath = jd_excel_filepath
        self.resume_excel_filepath = resume_excel_filepath

        self.jds = pd.read_excel(self.jd_excel_filepath).to_dict()
        self.resumes = pd.read_excel(self.resume_excel_filepath).to_dict()

        self.output_dict = {
            'JD': [],
            'Resume': []
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
        Combines every JD-Resume combination as one row and finally stores everything 
        in an excel file.
        """
        for jd in self.jds['Job Description'].values():
            for resume in self.resumes['Resume'].values():
                self.output_dict['JD'].append(jd)
                self.output_dict['Resume'].append(resume)
        self.save_current_output_dict()


if __name__ == '__main__':
    JD_EXCEL_FILEPATH = "/home/om/code/Resume-Evaluator/data/JDs.xlsx"
    RESUME_EXCEL_FILEPATH = "/home/om/code/Resume-Evaluator/data/resumes.xlsx"
    OUTPUT_STORE_PATH = "dataset_without_labels.xlsx"
    combiner = JDResumeCombiner(
        jd_excel_filepath=JD_EXCEL_FILEPATH,
        resume_excel_filepath=RESUME_EXCEL_FILEPATH,
        output_store_path=OUTPUT_STORE_PATH
    )
    combiner()