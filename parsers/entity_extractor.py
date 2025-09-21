# parsers/entity_extractor.py - Basic version
import re

class EntityExtractor:
    def __init__(self):
        print("✅ Entity extractor initialized (basic mode)")
    
    def extract_skills_with_nlp(self, text):
        """Basic entity extraction"""
        return {
            "persons": [],
            "organizations": [],
            "technologies": [],
            "skills": [],
            "locations": []
        }
    
    def extract_experience_years(self, text):
        """Extract years of experience using regex"""
        pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)'
        matches = re.findall(pattern, text.lower())
        years = [int(match) for match in matches if match.isdigit()]
        return max(years) if years else 0
    
    def extract_education_info(self, text):
        """Extract education info"""
        degrees = ["bachelor", "master", "phd", "b.tech", "m.tech"]
        found_degrees = [degree for degree in degrees if degree in text.lower()]
        
        return {
            "degrees": found_degrees,
            "fields": []
        }
