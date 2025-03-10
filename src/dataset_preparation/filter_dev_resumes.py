import os
import random
import pymupdf4llm
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from utils import fancy_print


class ResumeClassifier:
    """
    Class to handle all operations related to the filtering of resumes.
    """

    def __init__(self, resume_dir: str, output_dir: str, embedding_model: str = "all-mpnet-base-v2") -> None:
        """
        Initialises the sentence transformer model and other constants needed for filtering.
        """
        print(f"Initialising {embedding_model} model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.categories = {
            "Data Science": "Working with data, building machine learning models, deep learning, artificial intelligence, data visualization, statistics, and data analysis.",
            "Web Designing": "Creating website layouts, UI/UX design, working with HTML, CSS, JavaScript, Figma, Adobe XD, and responsive design principles.",
            "Java Developer": "Developing applications using Java, Spring Boot, Hibernate, Microservices, backend development, and software engineering principles.",
            "DevOps Engineer": "Handling CI/CD pipelines, Docker, Kubernetes, cloud infrastructure (AWS, Azure, GCP), automation, and system reliability.",
            "Database": "Working with SQL, MySQL, PostgreSQL, MongoDB, database administration, query optimization, and data modeling.",
            "Testing": "Manual and automated testing, writing test cases, using tools like JIRA, Selenium, TestNG, performance testing, and software quality assurance."
        }
        print(f"Computing embeddings for pre-defined categories...")
        self.category_embeddings = {key: self.embedding_model.encode(value, convert_to_tensor=True) for key, value in self.categories.items()}
        self.output_dict = {
            "Filename": [],
            "Text": []
        }
        self.resume_dir = resume_dir
        self.output_dir = output_dir

    def save_current_output(self) -> None:
        """
        Saves current version of the output dataframe as excel file.
        """
        output_df = pd.DataFrame(self.output_dict)
        output_df.to_excel(self.output_dir, index=False)

    def filter_resumes(self, similarity_threshold: float = 0.5) -> pd.DataFrame:
        """
        Iterates through the unfiltered resumes and only saves those ones which are related to 
        software development roles.
        """
        files = os.listdir(self.resume_dir)
        random.shuffle(files)
        for index, file in enumerate(files):
            try:
                print(f"\n\nProcessing file {index + 1} out of {len(os.listdir(self.resume_dir))}\n\n")
                filepath = os.path.join(self.resume_dir, file)
                filesize = os.path.getsize(filepath)

                # SKIP LARGE FILES
                if filesize > 500000:
                    raise OverflowError("File size exceeding the threshold!")

                # GETTING THE RESUME TEXT
                text = pymupdf4llm.to_markdown(doc=filepath)
                print(text)
                highest_category_score = 0

                # COMPARING WITH PRE-DEFINED CATEGORIES
                for category, description_embedding in self.category_embeddings.items():
                    resume_text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)
                    category_score = util.cos_sim(resume_text_embedding, description_embedding).item()
                    print(f"{category}: {category_score}")
                    if category_score > highest_category_score:
                        highest_category_score = category_score

                if highest_category_score >= similarity_threshold:
                    self.output_dict['Filename'].append(file)
                    self.output_dict['Text'].append(text)
                
                self.save_current_output()
            except Exception as e:
                print(f"Skipping file {index + 1} due to the following error: {str(e)}")
                continue


if __name__ == '__main__':
    fancy_print("Starting Resume Filtering")
    resume_dir = "/home/om/code/Resume-Evaluator/data/ResumesPDF"
    output_dir = "pdf_dataset.xlsx"
    resume_classifier = ResumeClassifier(
        resume_dir=resume_dir,
        output_dir=output_dir
    )
    resume_classifier.filter_resumes()