# parsers/skill_extractor.py - ENHANCED VERSION
import re
from parsers.skills_list import skills

def extract_skills(text):
    """Extract known skills from text using dictionary matching"""
    if not text:
        return []
        
    # Convert to lowercase for matching
    text_lower = text.lower()
    found_skills = []

    # Enhanced skill extraction
    for skill in skills:
        skill_lower = skill.lower()
        
        # Multiple matching strategies
        patterns = [
            rf'\b{re.escape(skill_lower)}\b',  # Exact word boundary match
            rf'{re.escape(skill_lower)}(?:\.\s*js|js)',  # Handle variations like "node.js"
            rf'{re.escape(skill_lower)}(?:\s*\.\s*\w+)?'  # Handle extensions
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_skills.append(skill)
                break

    # Additional extraction for common variations
    skill_variations = {
        'javascript': ['js', 'javascript', 'ecmascript'],
        'python': ['python', 'py'],
        'node.js': ['nodejs', 'node.js', 'node js'],
        'postgresql': ['postgres', 'postgresql', 'psql'],
        'kubernetes': ['k8s', 'kubernetes'],
        'docker': ['docker', 'containerization'],
        'ci/cd': ['ci/cd', 'cicd', 'continuous integration', 'continuous deployment']
    }
    
    for main_skill, variations in skill_variations.items():
        for variation in variations:
            if variation in text_lower and main_skill not in found_skills:
                if main_skill in skills:  # Only add if it's in our skills list
                    found_skills.append(main_skill)

    # Remove duplicates and return
    return list(set(found_skills))

def debug_skills_extraction(text):
    """Debug version to see what's happening"""
    print(f"🔍 Text length: {len(text)}")
    print(f"🔍 First 300 chars: {text[:300]}")
    
    # Check for obvious skills manually
    obvious_skills = ['python', 'javascript', 'react', 'node.js', 'aws', 'docker']
    found_obvious = [skill for skill in obvious_skills if skill.lower() in text.lower()]
    print(f"🔍 Obvious skills found: {found_obvious}")
    
    skills_found = extract_skills(text)
    print(f"🔍 Total skills extracted: {len(skills_found)}")
    print(f"🔍 Skills: {skills_found}")
    
    return skills_found
