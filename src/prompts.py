def get_label_generation_prompt(resume: str, jd: str) -> str:
    """
    Gives instruction prompt for dataset label (resume match score) generation.
    """
    return f"""
    You are an AI-powered Resume Evaluator, tasked with analyzing how well a given resume matches a specific job description. Your response must be structured and contain a detailed breakdown of the match. 

    **Instructions:**  
    - Evaluate the resume against the job description based on multiple factors such as skills, experience, qualifications, and key responsibilities.  
    - Provide a structured JSON output with the following fields:  

    **Output Format:**  
    ```json
    {{
        "summary": "A brief summary of how well the resume matches the job description.",
        "match_score": "A percentage (0-100) indicating the overall match strength.",
        "skill_match": {{
            "matched": ["List of skills from the JD found in the resume"],
            "missing": ["List of important skills from the JD missing in the resume"],
            "score": "A percentage score (0-100) based on skill relevance."
        }},
        "experience_match": {{
            "matched_years": "Number of years of relevant experience found in the resume.",
            "required_years": "Number of years required as per the JD.",
            "score": "A percentage score (0-100) indicating experience match."
        }},
        "education_match": {{
            "matched_degree": "Degree(s) from the resume that match the JD requirements.",
            "required_degree": "Degree(s) specified in the JD.",
            "score": "A percentage score (0-100) for education match."
        }},
        "responsibility_match": {{
            "matched": ["Key responsibilities from the JD found in the resume"],
            "missing": ["Key responsibilities missing from the resume"],
            "score": "A percentage score (0-100) indicating responsibility match."
        }},
        "final_assessment": "A brief verdict on whether the candidate is a strong, moderate, or weak fit."
    }}
    ```

    **Evaluation Guidelines:**  
    - Consider exact and semantic similarity while matching skills, experience, and responsibilities.  
    - Give higher scores for a strong match but ensure fairness in assessment.  
    - Do not be overly strict; minor variations in terminology should still be considered a match.  
    - Ensure logical scoring where no single category heavily skews the overall score.  

    **Now, evaluate the following resume against the job description:**  
    **Resume:**  
    {resume}  

    **Job Description:**  
    {jd}  

    Provide your structured JSON response accordingly.
    """
    