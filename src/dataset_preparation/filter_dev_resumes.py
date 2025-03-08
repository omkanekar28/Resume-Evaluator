import os
import pymupdf4llm
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class ResumeClassifier:
    """
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """
        """
        print(f"Initialising {embedding_model} model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.categories = {
            "Data Science": "Working with data, building machine learning models, deep learning, artificial intelligence, data visualization, statistics, and data analysis.",
            "Web Designing": "Creating website layouts, UI/UX design, working with HTML, CSS, JavaScript, Figma, Adobe XD, and responsive design principles.",
            "Java Developer": "Developing applications using Java, Spring Boot, Hibernate, Microservices, backend development, and software engineering principles.",
            "SAP Developer": "Working with SAP modules, ABAP programming, SAP HANA, enterprise resource planning (ERP) solutions, and SAP integrations.",
            "Automation Testing": "Developing test scripts, using Selenium, Cypress, Playwright, automated QA, test automation frameworks, and performance testing.",
            "Python Developer": "Building applications using Python, Flask, Django, FastAPI, scripting, backend development, and data processing.",
            "DevOps Engineer": "Handling CI/CD pipelines, Docker, Kubernetes, cloud infrastructure (AWS, Azure, GCP), automation, and system reliability.",
            "Network Security Engineer": "Implementing firewalls, network security protocols, ethical hacking, intrusion detection, cybersecurity, and VPN management.",
            "Database": "Working with SQL, MySQL, PostgreSQL, MongoDB, database administration, query optimization, and data modeling.",
            "Hadoop": "Big data processing using Hadoop, Spark, HDFS, MapReduce, data pipelines, and distributed computing.",
            "ETL Developer": "Extracting, transforming, and loading data, working with ETL tools like Informatica, Talend, DataStage, and SQL-based data pipelines.",
            "Dotnet Developer": "Developing applications using .NET framework, C#, ASP.NET, MVC, Web API, and Windows applications.",
            "Testing": "Manual and automated testing, writing test cases, using tools like JIRA, Selenium, TestNG, performance testing, and software quality assurance."
        }
        print(f"Computing embeddings for pre-defined categories...")
        self.category_embeddings = {key: self.embedding_model.encode(value, convert_to_tensor=True) for key, value in self.categories.items()}

    def filter_resumes(self, resume_dir: str) -> pd.DataFrame:
        """
        """
        for index, file in enumerate(os.listdir(resume_dir)):
            print(f"\n\nProcessing file {index + 1} out of {len(os.listdir(resume_dir))}\n\n")
            filepath = os.path.join(resume_dir, file)

            # GETTING THE RESUME TEXT
            text = pymupdf4llm.to_markdown(doc=filepath)
            print(text)

            # COMPARING WITH PRE-DEFINED CATEGORIES
            for category, description_embedding in self.category_embeddings.items():
                resume_text_embedding = self.embedding_model.encode(text, convert_to_tensor=True)
                category_score = util.cos_sim(resume_text_embedding, description_embedding).item()
                print(f"{category}: {category_score}")

        # TODO: Write logic to take resumes with respectable score in atleast one of the categories 
        #       (Would be better to take some irrelevant ones too for generalisation of data).


if __name__ == '__main__':
    resume_dir = "/home/om/code/Resume-Evaluator/data/ResumesPDF"
    resume_classifier = ResumeClassifier()
    resume_classifier.filter_resumes(resume_dir=resume_dir)