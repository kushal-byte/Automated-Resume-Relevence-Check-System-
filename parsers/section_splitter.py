# parsers/section_splitter.py - FIXED VERSION
import re

def split_sections(text):
    """Split resume into sections like education, skills, experience"""
    sections = {}
    current_section = "general"
    
    # Clean text first
    text = text.replace('\n', ' ').strip()
    
    # Split by common section headers (more comprehensive)
    section_patterns = [
        r'(professional\s+summary|summary|objective)',
        r'(technical\s+skills|skills|core\s+competencies|technologies)',
        r'(work\s+experience|experience|employment|professional\s+experience)',
        r'(education|academic\s+background|qualifications)',
        r'(projects|personal\s+projects|key\s+projects)',
        r'(certifications|certificates|credentials)',
        r'(achievements|accomplishments|awards)'
    ]
    
    # Find section boundaries
    section_starts = []
    for pattern in section_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            section_starts.append((match.start(), match.group().lower().strip()))
    
    # Sort by position
    section_starts.sort()
    
    # Extract sections
    if not section_starts:
        # Fallback: if no clear sections, try to extract skills manually
        sections["general"] = text
        sections["skills"] = extract_skills_section_fallback(text)
    else:
        for i, (start_pos, section_name) in enumerate(section_starts):
            # Determine end position
            if i + 1 < len(section_starts):
                end_pos = section_starts[i + 1][0]
                section_text = text[start_pos:end_pos]
            else:
                section_text = text[start_pos:]
            
            # Clean section name
            clean_name = re.sub(r'[^\w\s]', '', section_name).strip()
            sections[clean_name] = section_text.strip()
    
    return sections

def extract_skills_section_fallback(text):
    """Fallback to extract skills when section detection fails"""
    # Look for skills-related keywords
    skills_indicators = [
        r'programming languages?:?\s*([^.]*)',
        r'technical skills?:?\s*([^.]*)',
        r'technologies?:?\s*([^.]*)', 
        r'tools?:?\s*([^.]*)',
        r'frameworks?:?\s*([^.]*)',
        r'languages?:?\s*([^.]*)'
    ]
    
    skills_text = ""
    for pattern in skills_indicators:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            skills_text += " " + match
    
    return skills_text.strip() if skills_text else ""
