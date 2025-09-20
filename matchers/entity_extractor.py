# parsers/entity_extractor.py - SPACY ENTITY EXTRACTION
import spacy
from collections import Counter
import re

class EntityExtractor:
    def __init__(self):
        try:
            print("🧠 Loading spaCy model...")
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ spaCy model loaded successfully")
        except OSError:
            print("⚠️ spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_skills_with_nlp(self, text):
        """Extract skills using spaCy NLP"""
        if not self.nlp:
            return self._fallback_extraction(text)
        
        print("🔍 Extracting entities with spaCy...")
        
        doc = self.nlp(text)
        
        # Extract entities
        entities = {
            "persons": [],
            "organizations": [],
            "technologies": [],
            "skills": [],
            "locations": []
        }
        
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities["persons"].append(ent.text)
            elif ent.label_ == "ORG":
                entities["organizations"].append(ent.text)
            elif ent.label_ == "GPE":  # Geopolitical entity (locations)
                entities["locations"].append(ent.text)
        
        # Extract noun phrases as potential skills
        noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                       if len(chunk.text.split()) <= 3]  # Max 3 words
        
        # Filter technical terms
        tech_patterns = [
            r'\b\w+\.js\b', r'\b\w+script\b', r'\b\w+SQL\b', 
            r'\bAPI\b', r'\bSDK\b', r'\bIDE\b', r'\bOS\b'
        ]
        
        tech_terms = []
        for pattern in tech_patterns:
            tech_terms.extend(re.findall(pattern, text, re.IGNORECASE))
        
        entities["technologies"] = list(set(tech_terms))
        entities["skills"] = list(set(noun_phrases))
        
        return entities
    
    def extract_experience_years(self, text):
        """Extract years of experience using NLP"""
        if not self.nlp:
            return self._extract_years_regex(text)
        
        doc = self.nlp(text)
        
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*in',
            r'experience.*?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*year.*?experience'
        ]
        
        years = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text.lower())
            years.extend([int(match) for match in matches if match.isdigit()])
        
        return max(years) if years else 0
    
    def extract_education_info(self, text):
        """Extract education information"""
        degrees = [
            "bachelor", "master", "phd", "doctorate", "diploma", 
            "b.tech", "m.tech", "bca", "mca", "bsc", "msc"
        ]
        
        fields = [
            "computer science", "engineering", "information technology",
            "software engineering", "data science", "mathematics"
        ]
        
        found_degrees = []
        found_fields = []
        
        text_lower = text.lower()
        
        for degree in degrees:
            if degree in text_lower:
                found_degrees.append(degree)
        
        for field in fields:
            if field in text_lower:
                found_fields.append(field)
        
        return {
            "degrees": list(set(found_degrees)),
            "fields": list(set(found_fields))
        }
    
    def _fallback_extraction(self, text):
        """Fallback extraction without spaCy"""
        print("⚠️ Using fallback extraction (spaCy not available)")
        
        # Simple regex-based extraction
        entities = {
            "persons": [],
            "organizations": [],
            "technologies": [],
            "skills": [],
            "locations": []
        }
        
        # Extract email domains as organizations
        email_domains = re.findall(r'@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text)
        entities["organizations"] = [domain.split('.')[0] for domain in email_domains]
        
        return entities
    
    def _extract_years_regex(self, text):
        """Regex fallback for experience extraction"""
        pattern = r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:experience|exp)'
        matches = re.findall(pattern, text.lower())
        years = [int(match) for match in matches if match.isdigit()]
        return max(years) if years else 0

# Test function
def test_entity_extractor():
    """Test entity extraction functionality"""
    extractor = EntityExtractor()
    
    sample_text = """
    John Smith is a Python developer with 3+ years of experience at Google.
    He has worked with React.js, Node.js, and AWS in San Francisco.
    Bachelor's degree in Computer Science.
    """
    
    entities = extractor.extract_skills_with_nlp(sample_text)
    years = extractor.extract_experience_years(sample_text)
    education = extractor.extract_education_info(sample_text)
    
    print(f"✅ Entities extracted: {len(entities['skills'])} skills found")
    print(f"✅ Experience: {years} years")
    print(f"✅ Education: {education}")
    
    return len(entities['skills']) > 0

if __name__ == "__main__":
    test_entity_extractor()
