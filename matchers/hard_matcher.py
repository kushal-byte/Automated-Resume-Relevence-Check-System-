# matchers/hard_matcher.py
def calculate_hard_match_score(resume_skills, jd_skills):
    """Calculate percentage match based on keyword overlap"""
    if not jd_skills:  # avoid division by zero
        return 0.0
    
    matched_skills = set(resume_skills) & set(jd_skills)
    total_jd_skills = len(set(jd_skills))
    
    coverage_percentage = len(matched_skills) / total_jd_skills * 100
    
    return {
        "score": round(coverage_percentage, 2),
        "matched_count": len(matched_skills),
        "total_jd_skills": total_jd_skills,
        "matched_skills": list(matched_skills),
        "missing_skills": list(set(jd_skills) - set(resume_skills))
    }

def calculate_fuzzy_match(resume_text, jd_skills):
    """Fuzzy matching for skill variations (JavaScript vs JS)"""
    # Install: pip install rapidfuzz
    from rapidfuzz import fuzz
    
    resume_lower = resume_text.lower()
    fuzzy_matches = []
    
    for skill in jd_skills:
        # Check if skill or common variations exist
        variations = get_skill_variations(skill)
        for variation in variations:
            if fuzz.partial_ratio(variation, resume_lower) > 80:
                fuzzy_matches.append(skill)
                break
                
    return list(set(fuzzy_matches))

def get_skill_variations(skill):
    """Common skill variations for fuzzy matching"""
    variations = {
        "javascript": ["js", "javascript", "node.js", "nodejs"],
        "python": ["python", "py"],
        "tensorflow": ["tensorflow", "tf"],
        "kubernetes": ["kubernetes", "k8s"],
        "postgresql": ["postgresql", "postgres", "psql"]
    }
    return variations.get(skill.lower(), [skill])
