# Resume Evaluator (Project on Hold)

🚨 **Status: On Hold** 🚨  
This project is currently on hold because labeling 20,000+ samples is extremely time-consuming. However, the dataset and training pipeline are available for further use and contribution.

## 📌 Project Overview
This is an AI-powered **Resume Evaluator** that aims to analyze resumes and match them with job descriptions. The goal was to fine-tune an LLM-based model to classify resumes based on job relevance.

## 📂 Dataset
- **Source:**
  - **Job Descriptions:** Scraped from **LinkedIn, Wellfound, and Glassdoor**
  - **Resumes:** Extracted from Kaggle datasets
- **Size:** ~20,000 rows (unlabeled)
- **Challenges:** Requires manual/LLM-assisted labeling, which is resource-intensive

## 🏗️ Project Components
- **Data Collection & Preprocessing**: Extracting resumes and job descriptions, cleaning text
- **Embedding Generation**: Using sentence-transformers for cosine similarity comparison
- **Training Pipeline**: Script for fine-tuning a model (currently incomplete due to compute constraints)
