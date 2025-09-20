import re
from parsers.cleaner import clean_text
from parsers.skill_extractor import extract_skills

def parse_jd(file_text):
    """Parse job description and extract role + skills"""
    text = clean_text(file_text)

    # Extract Job Role (look for keywords like "Job Title", "Role", "Position")
    role_match = re.search(r"(job role|job title|position)\s*[:\-]\s*(.*)", text, re.I)
    job_role = role_match.group(2).strip() if role_match else "Unknown"

    # Extract skills
    jd_skills = extract_skills(text)

    return {
        "role": job_role,
        "skills": jd_skills,
        "raw_text": text
    }
