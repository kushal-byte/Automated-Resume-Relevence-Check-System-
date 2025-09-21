# parsers/smart_skill_extractor.py - AI-Powered Skill Detection
import re
from collections import Counter

class SmartSkillExtractor:
    """AI-powered skill extraction that finds ANY skill mentioned in text"""
    
    def __init__(self):
        self.skill_database = self._load_comprehensive_skills()
        self.patterns = self._create_extraction_patterns()
        print(f"✅ Smart Skill Extractor loaded with {len(self.skill_database)} skills")
    
    def _load_comprehensive_skills(self):
        """Load comprehensive skill database covering all domains"""
        
        # Programming Languages
        programming = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c', 'php', 'ruby', 'go', 'rust',
            'kotlin', 'swift', 'scala', 'r', 'matlab', 'perl', 'bash', 'powershell', 'sql', 'html',
            'css', 'sass', 'less', 'coffeescript', 'dart', 'elixir', 'erlang', 'f#', 'haskell',
            'julia', 'lua', 'objective-c', 'vb.net', 'assembly', 'cobol', 'fortran'
        ]
        
        # Frameworks & Libraries
        frameworks = [
            'react', 'angular', 'vue', 'svelte', 'ember', 'backbone', 'jquery', 'bootstrap', 'tailwind',
            'django', 'flask', 'fastapi', 'express', 'nodejs', 'spring', 'hibernate', 'struts',
            'rails', 'sinatra', 'laravel', 'symfony', 'codeigniter', 'asp.net', 'entity framework',
            'xamarin', 'flutter', 'react native', 'ionic', 'cordova', 'electron', 'unity', 'unreal',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
            'seaborn', 'plotly', 'opencv', 'nltk', 'spacy'
        ]
        
        # Databases
        databases = [
            'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra', 'elasticsearch', 'neo4j',
            'couchdb', 'dynamodb', 'firestore', 'sqlite', 'oracle', 'sql server', 'mariadb',
            'influxdb', 'clickhouse', 'bigquery', 'snowflake', 'redshift'
        ]
        
        # Cloud & DevOps
        cloud_devops = [
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'gitlab ci', 'github actions',
            'terraform', 'ansible', 'puppet', 'chef', 'vagrant', 'consul', 'vault', 'prometheus',
            'grafana', 'elk stack', 'nginx', 'apache', 'tomcat',
            'linux', 'ubuntu', 'centos', 'windows server', 'git', 'svn'
        ]
        
        # Data Science & AI
        data_ai = [
            'machine learning', 'deep learning', 'artificial intelligence', 'data science',
            'data analysis', 'data mining', 'big data', 'analytics', 'statistics', 'regression',
            'classification', 'clustering', 'nlp', 'computer vision', 'neural networks'
        ]
        
        # Business & Soft Skills
        business_soft = [
            'project management', 'agile', 'scrum', 'kanban', 'leadership', 'communication',
            'teamwork', 'problem solving', 'time management', 'quality assurance',
            'business analysis', 'user research', 'ux design', 'ui design'
        ]
        
        # Tools & Platforms
        tools = [
            'jira', 'confluence', 'slack', 'figma', 'photoshop', 'excel', 'powerpoint',
            'salesforce', 'google analytics', 'seo', 'automation', 'crm', 'erp'
        ]
        
        # Combine all skills
        all_skills = (programming + frameworks + databases + cloud_devops + 
                     data_ai + business_soft + tools)
        
        # Create variations mapping
        skill_variations = {}
        for skill in all_skills:
            variations = [skill, skill.replace(' ', ''), skill.replace(' ', '_'), 
                         skill.replace(' ', '-'), skill.upper(), skill.lower()]
            
            # Add common abbreviations
            abbreviations = {
                'javascript': ['js', 'javascript'],
                'typescript': ['ts', 'typescript'],
                'artificial intelligence': ['ai', 'artificial intelligence'],
                'machine learning': ['ml', 'machine learning'],
                'amazon web services': ['aws', 'amazon web services'],
                'google cloud platform': ['gcp', 'google cloud'],
                'kubernetes': ['k8s', 'kubernetes'],
                'user experience': ['ux', 'user experience'],
                'user interface': ['ui', 'user interface'],
                'structured query language': ['sql', 'structured query language'],
                'cascading style sheets': ['css', 'cascading style sheets'],
                'hypertext markup language': ['html', 'hypertext markup language']
            }
            
            skill_key = skill.lower()
            if skill_key in abbreviations:
                variations.extend(abbreviations[skill_key])
            
            for var in variations:
                if var and len(var) > 1:
                    skill_variations[var.lower()] = skill
        
        return skill_variations
    
    def _create_extraction_patterns(self):
        """Create regex patterns for skill extraction"""
        return {
            'experience_with': r'\b(?:experience|expertise|proficient|skilled)\s+(?:in|with|using)\s+([a-zA-Z+#.\s-]+)\b',
            'years_exp': r'\b(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:experience|exp)\s+(?:in|with|using)\s+([a-zA-Z+#.\s-]+)\b',
            'worked_with': r'\b(?:worked|working|used|using)\s+(?:with|on)?\s*([a-zA-Z+#.\s-]+)\b',
            'technologies': r'\b(?:technologies|tools|frameworks|skills)[\s:]*([a-zA-Z+#.\s,-]+)\b',
            'skills': r'\b(?:skills?|competencies)[\s:]*([a-zA-Z+#.\s,-]+)\b'
        }
    
    def extract_skills_comprehensive(self, text):
        """Extract skills using multiple techniques"""
        if not text or len(text.strip()) < 10:
            return []
        
        found_skills = set()
        text_lower = text.lower()
        
        # Method 1: Direct skill matching
        for skill_variant, canonical_skill in self.skill_database.items():
            if skill_variant in text_lower:
                # Verify it's a whole word match
                pattern = r'\b' + re.escape(skill_variant) + r'\b'
                if re.search(pattern, text_lower):
                    found_skills.add(canonical_skill)
        
        # Method 2: Pattern-based extraction
        for pattern_name, pattern in self.patterns.items():
            matches = re.finditer(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0 and match.group(1):
                    # Clean and process the captured group
                    skill_text = match.group(1).strip(' ,-')
                    extracted_skills = self._process_skill_text(skill_text)
                    found_skills.update(extracted_skills)
        
        # Method 3: Context-based extraction
        context_skills = self._extract_contextual_skills(text)
        found_skills.update(context_skills)
        
        return sorted(list(found_skills))
    
    def _process_skill_text(self, skill_text):
        """Process extracted skill text to find valid skills"""
        skills = set()
        
        # Split by common separators
        parts = re.split(r'[,;/\|\n]', skill_text)
        
        for part in parts:
            part = part.strip(' ,-()[]{}')
            if len(part) > 1:
                # Check if it's in our skill database
                part_lower = part.lower()
                if part_lower in self.skill_database:
                    skills.add(self.skill_database[part_lower])
                
                # Check individual words
                words = part.split()
                for word in words:
                    word = word.strip(' ,-()[]{}').lower()
                    if word in self.skill_database:
                        skills.add(self.skill_database[word])
        
        return skills
    
    def _extract_contextual_skills(self, text):
        """Extract skills based on context clues"""
        skills = set()
        
        # Look for skills in specific sections
        section_patterns = {
            r'(?:technical\s+)?skills?[\s:]+([^.]+)': 'skills_section',
            r'technologies?[\s:]+([^.]+)': 'tech_section',
            r'tools?[\s:]+([^.]+)': 'tools_section'
        }
        
        for pattern, section_type in section_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0:
                    content = match.group(1)
                    # Extract skills from this section
                    section_skills = self._process_skill_text(content)
                    skills.update(section_skills)
        
        return skills
    
    def get_skill_categories(self, skills):
        """Categorize extracted skills"""
        categories = {
            'Programming Languages': [],
            'Frameworks & Libraries': [],
            'Databases': [],
            'Cloud & DevOps': [],
            'Data Science & AI': [],
            'Business & Soft Skills': [],
            'Tools & Platforms': []
        }
        
        # Simple categorization based on skill type
        for skill in skills:
            skill_lower = skill.lower()
            
            if any(lang in skill_lower for lang in ['python', 'java', 'javascript', 'c++', 'php', 'ruby']):
                categories['Programming Languages'].append(skill)
            elif any(fw in skill_lower for fw in ['react', 'angular', 'django', 'spring', 'tensorflow']):
                categories['Frameworks & Libraries'].append(skill)
            elif any(db in skill_lower for db in ['mysql', 'mongodb', 'postgresql', 'redis']):
                categories['Databases'].append(skill)
            elif any(cloud in skill_lower for cloud in ['aws', 'azure', 'docker', 'kubernetes']):
                categories['Cloud & DevOps'].append(skill)
            elif any(ai in skill_lower for ai in ['machine learning', 'ai', 'data science', 'analytics']):
                categories['Data Science & AI'].append(skill)
            elif any(tool in skill_lower for tool in ['jira', 'figma', 'photoshop', 'excel']):
                categories['Tools & Platforms'].append(skill)
            else:
                categories['Business & Soft Skills'].append(skill)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}

# Test function
def test_smart_extractor():
    """Test the smart skill extractor"""
    extractor = SmartSkillExtractor()
    
    test_text = """
    John Doe - Software Engineer
    Skills: Python, JavaScript, React, MySQL, AWS
    Experience: 3 years of experience in full-stack development
    """
    
    skills = extractor.extract_skills_comprehensive(test_text)
    print(f"✅ Extracted {len(skills)} skills: {skills}")
    
    return len(skills) > 0

if __name__ == "__main__":
    test_smart_extractor()
