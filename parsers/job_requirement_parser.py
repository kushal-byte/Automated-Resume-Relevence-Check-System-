# parsers/job_requirement_parser.py - Advanced Job Requirement Analysis
import re
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from parsers.smart_skill_extractor import SmartSkillExtractor

@dataclass
class JobRequirement:
    """Structured job requirement data"""
    role_title: str
    company: str
    experience_required: str
    education_required: List[str]
    must_have_skills: List[str]
    good_to_have_skills: List[str]
    certifications: List[str]
    responsibilities: List[str]
    benefits: List[str]
    location: str
    employment_type: str
    salary_range: str
    industry: str
    seniority_level: str

class JobRequirementParser:
    """Parse job descriptions to extract structured requirements"""
    
    def __init__(self):
        self.skill_extractor = SmartSkillExtractor()
        self.patterns = self._initialize_patterns()
        print("✅ Job Requirement Parser initialized")
    
    def _initialize_patterns(self):
        """Initialize regex patterns for job parsing"""
        return {
            'role_title': [
                r'(?:job\s+title|position|role)[\s:]*([^\n.]{5,80})',
                r'^([A-Z][\w\s,]+(?:engineer|developer|manager|analyst|specialist|coordinator))\b',
                r'hiring\s+for[\s:]*([^\n.]{5,80})',
            ],
            'company': [
                r'(?:company|organization)[\s:]*([^\n]+)',
                r'(?:at|@)\s+([A-Z][a-zA-Z\s&,.-]+?)(?:\s|$)',
            ],
            'experience': [
                r'(?:experience|exp)[\s:]*(\d+[\+\-]*\s*(?:to|\-)\s*\d+\s*years?|\d+\+?\s*years?)',
                r'(\d+[\+\-]*)\s*(?:to|\-)\s*(\d+)\s*years?\s*(?:of\s+)?(?:experience|exp)',
                r'minimum\s+(\d+\+?)\s*years?',
                r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)',
            ],
            'education': [
                r'(?:education|degree|qualification)[\s:]*([^\n]+)',
                r'(?:bachelor|master|phd|doctorate|diploma|b\.tech|m\.tech|bca|mca|bsc|msc)[\s\.]*([^\n]*)',
                r'(?:degree\s+in|graduated\s+in)\s+([^\n]+)',
            ],
            'must_have': [
                r'(?:must\s+have|required|mandatory|essential)[\s:]*([^.]+)',
                r'(?:requirements|qualifications)[\s:]*([^.]+)',
                r'(?:should\s+have|need\s+to\s+have)[\s:]*([^.]+)',
            ],
            'good_to_have': [
                r'(?:good\s+to\s+have|nice\s+to\s+have|preferred|bonus|plus)[\s:]*([^.]+)',
                r'(?:additional|optional)[\s:]*([^.]+)',
            ],
            'responsibilities': [
                r'(?:responsibilities|duties|tasks)[\s:]*([^.]+)',
                r'(?:you\s+will|role\s+involves)[\s:]*([^.]+)',
            ],
            'certifications': [
                r'(?:certification|certified|certificate)[\s:]*([^.]+)',
                r'(?:aws|azure|google\s+cloud|oracle|cisco|microsoft)\s+certified[\s:]*([^.]*)',
            ],
            'salary': [
                r'(?:salary|compensation|package)[\s:]*([^.\n]+)',
                r'(?:\$|₹|€|£)\s*([0-9,.-]+(?:\s*(?:to|\-)\s*[0-9,.-]+)?)',
                r'([0-9,]+)\s*(?:to|\-)\s*([0-9,]+)\s*(?:per\s+)?(?:month|year|annum)',
            ],
            'location': [
                r'(?:location|based\s+in|office)[\s:]*([^.\n]+)',
                r'(?:remote|hybrid|onsite|work\s+from)[\s:]*([^.\n]*)',
            ]
        }
    
    def parse_job_description(self, jd_text: str) -> JobRequirement:
        """Parse job description into structured requirements"""
        
        if not jd_text:
            return self._create_empty_requirement()
        
        print("🔍 Parsing job requirements...")
        
        # Extract basic information
        role_title = self._extract_role_title(jd_text)
        company = self._extract_company(jd_text)
        experience = self._extract_experience(jd_text)
        education = self._extract_education(jd_text)
        location = self._extract_location(jd_text)
        salary = self._extract_salary(jd_text)
        
        # Extract skills and requirements
        must_have_skills, good_to_have_skills = self._extract_skills_by_priority(jd_text)
        certifications = self._extract_certifications(jd_text)
        responsibilities = self._extract_responsibilities(jd_text)
        
        # Determine job characteristics
        employment_type = self._determine_employment_type(jd_text)
        industry = self._determine_industry(jd_text, role_title)
        seniority_level = self._determine_seniority(role_title, experience)
        
        job_req = JobRequirement(
            role_title=role_title,
            company=company,
            experience_required=experience,
            education_required=education,
            must_have_skills=must_have_skills,
            good_to_have_skills=good_to_have_skills,
            certifications=certifications,
            responsibilities=responsibilities,
            benefits=[],  # Can be enhanced later
            location=location,
            employment_type=employment_type,
            salary_range=salary,
            industry=industry,
            seniority_level=seniority_level
        )
        
        print(f"✅ Parsed job: {role_title} at {company}")
        print(f"   📍 Location: {location}")
        print(f"   💼 Experience: {experience}")
        print(f"   🎯 Must-have skills: {len(must_have_skills)}")
        print(f"   ⭐ Good-to-have skills: {len(good_to_have_skills)}")
        
        return job_req
    
    def _extract_role_title(self, text: str) -> str:
        """Extract job role title"""
        for pattern in self.patterns['role_title']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for common job titles
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if any(title in line.lower() for title in 
                   ['engineer', 'developer', 'manager', 'analyst', 'specialist']):
                return line
        
        return "Unknown Role"
    
    def _extract_company(self, text: str) -> str:
        """Extract company name"""
        for pattern in self.patterns['company']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "Unknown Company"
    
    def _extract_experience(self, text: str) -> str:
        """Extract experience requirements"""
        for pattern in self.patterns['experience']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group().strip()
        
        # Look for fresher/entry level
        if re.search(r'\b(?:fresher|entry\s+level|0\s+years?)\b', text, re.IGNORECASE):
            return "0-1 years"
        
        return "Not specified"
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education requirements"""
        education = []
        
        for pattern in self.patterns['education']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                education.append(match.group().strip())
        
        # Common degree patterns
        degree_patterns = [
            r'\bb\.?tech\b', r'\bm\.?tech\b', r'\bbca\b', r'\bmca\b',
            r'\bbsc\b', r'\bmsc\b', r'\bba\b', r'\bmba\b',
            r'\bbachelor', r'\bmaster', r'\bphd\b', r'\bdoctorate\b'
        ]
        
        for pattern in degree_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                match = re.search(pattern + r'[^.\n]*', text, re.IGNORECASE)
                if match:
                    education.append(match.group().strip())
        
        return list(set(education)) if education else ["Any Graduate"]
    
    def _extract_skills_by_priority(self, text: str) -> Tuple[List[str], List[str]]:
        """Extract skills categorized by priority"""
        
        # Use smart extractor to get all skills
        all_skills = self.skill_extractor.extract_skills_comprehensive(text)
        
        must_have = []
        good_to_have = []
        
        # Categorize based on context
        text_lower = text.lower()
        
        # Split text into sections
        must_have_section = ""
        good_to_have_section = ""
        
        # Extract must-have skills
        for pattern in self.patterns['must_have']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                must_have_section += " " + match.group(1)
        
        # Extract good-to-have skills
        for pattern in self.patterns['good_to_have']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                good_to_have_section += " " + match.group(1)
        
        # Categorize skills
        for skill in all_skills:
            skill_lower = skill.lower()
            
            # Check if skill is in must-have section
            if skill_lower in must_have_section.lower():
                must_have.append(skill)
            # Check if skill is in good-to-have section
            elif skill_lower in good_to_have_section.lower():
                good_to_have.append(skill)
            # Default categorization based on job requirements context
            elif self._is_core_skill(skill, text):
                must_have.append(skill)
            else:
                good_to_have.append(skill)
        
        # Ensure no duplicates
        must_have = list(set(must_have))
        good_to_have = list(set(good_to_have) - set(must_have))
        
        return must_have, good_to_have
    
    def _is_core_skill(self, skill: str, text: str) -> bool:
        """Determine if a skill is core based on frequency and context"""
        skill_lower = skill.lower()
        text_lower = text.lower()
        
        # Count mentions
        mentions = text_lower.count(skill_lower)
        
        # Check for emphasis keywords around the skill
        emphasis_patterns = [
            rf'\b(?:required|must|essential|mandatory|need)\b[^.]*{re.escape(skill_lower)}',
            rf'{re.escape(skill_lower)}[^.]*\b(?:required|must|essential|mandatory)\b',
            rf'\b(?:experience|expertise|proficient)\b[^.]*{re.escape(skill_lower)}',
            rf'{re.escape(skill_lower)}[^.]*\b(?:years?|experience)\b'
        ]
        
        for pattern in emphasis_patterns:
            if re.search(pattern, text_lower):
                return True
        
        # If mentioned multiple times, likely core
        return mentions >= 2
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certification requirements"""
        certifications = []
        
        for pattern in self.patterns['certifications']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                cert = match.group().strip()
                if len(cert) > 5:  # Filter out too short matches
                    certifications.append(cert)
        
        return list(set(certifications))
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract job responsibilities"""
        responsibilities = []
        
        for pattern in self.patterns['responsibilities']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                resp = match.group(1).strip()
                # Split by bullet points or line breaks
                resp_list = re.split(r'[•\-\*]\s*|\n', resp)
                for r in resp_list:
                    r = r.strip()
                    if len(r) > 10:  # Filter meaningful responsibilities
                        responsibilities.append(r)
        
        return responsibilities[:10]  # Limit to top 10
    
    def _extract_location(self, text: str) -> str:
        """Extract job location"""
        for pattern in self.patterns['location']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Look for city names (basic patterns)
        city_pattern = r'\b(?:bangalore|mumbai|delhi|hyderabad|chennai|pune|kolkata|ahmedabad|remote|hybrid)\b'
        match = re.search(city_pattern, text, re.IGNORECASE)
        if match:
            return match.group()
        
        return "Not specified"
    
    def _extract_salary(self, text: str) -> str:
        """Extract salary information"""
        for pattern in self.patterns['salary']:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group().strip()
        
        return "Not specified"
    
    def _determine_employment_type(self, text: str) -> str:
        """Determine employment type"""
        text_lower = text.lower()
        
        if 'intern' in text_lower or 'internship' in text_lower:
            return "Internship"
        elif 'contract' in text_lower or 'freelance' in text_lower:
            return "Contract"
        elif 'part time' in text_lower or 'part-time' in text_lower:
            return "Part-time"
        else:
            return "Full-time"
    
    def _determine_industry(self, text: str, role_title: str) -> str:
        """Determine industry based on job content"""
        text_lower = (text + " " + role_title).lower()
        
        industry_keywords = {
            'Technology': ['software', 'tech', 'it', 'developer', 'engineer', 'programmer'],
            'Finance': ['finance', 'banking', 'fintech', 'investment', 'trading'],
            'Healthcare': ['healthcare', 'medical', 'hospital', 'pharma', 'clinical'],
            'Education': ['education', 'teaching', 'learning', 'university', 'academic'],
            'E-commerce': ['ecommerce', 'e-commerce', 'retail', 'shopping', 'marketplace'],
            'Marketing': ['marketing', 'advertising', 'promotion', 'brand', 'digital marketing'],
            'Consulting': ['consulting', 'advisory', 'strategy', 'management consulting'],
            'Manufacturing': ['manufacturing', 'production', 'industrial', 'automotive'],
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return industry
        
        return "General"
    
    def _determine_seniority(self, role_title: str, experience: str) -> str:
        """Determine seniority level"""
        title_lower = role_title.lower()
        
        if any(word in title_lower for word in ['senior', 'lead', 'principal', 'architect', 'manager']):
            return "Senior"
        elif any(word in title_lower for word in ['junior', 'associate', 'entry', 'trainee']):
            return "Junior"
        elif 'intern' in title_lower:
            return "Intern"
        else:
            # Determine by experience
            if '0' in experience or 'fresher' in experience.lower():
                return "Entry Level"
            elif any(num in experience for num in ['1', '2', '3']):
                return "Mid Level"
            else:
                return "Senior"
    
    def _create_empty_requirement(self) -> JobRequirement:
        """Create empty job requirement for error cases"""
        return JobRequirement(
            role_title="Unknown Role",
            company="Unknown Company",
            experience_required="Not specified",
            education_required=["Any Graduate"],
            must_have_skills=[],
            good_to_have_skills=[],
            certifications=[],
            responsibilities=[],
            benefits=[],
            location="Not specified",
            employment_type="Full-time",
            salary_range="Not specified",
            industry="General",
            seniority_level="Not specified"
        )
    
    def export_to_json(self, job_req: JobRequirement) -> str:
        """Export job requirement to JSON"""
        return json.dumps(job_req.__dict__, indent=2)

# Test function
def test_job_parser():
    """Test the job requirement parser"""
    parser = JobRequirementParser()
    
    sample_jd = """
    Senior Full Stack Developer - TechCorp Inc.
    
    Location: Bangalore, India (Hybrid)
    Experience: 3-5 years
    
    Job Description:
    We are looking for a Senior Full Stack Developer to join our growing team.
    
    Must Have Requirements:
    - 3+ years of experience in React.js and Node.js
    - Proficiency in JavaScript, TypeScript
    - Experience with MySQL and MongoDB
    - Knowledge of AWS cloud services
    - Bachelor's degree in Computer Science or related field
    
    Good to Have:
    - Experience with Docker and Kubernetes
    - Knowledge of microservices architecture
    - AWS certification preferred
    - Experience with CI/CD pipelines
    
    Responsibilities:
    - Develop and maintain web applications
    - Collaborate with cross-functional teams
    - Write clean, maintainable code
    - Participate in code reviews
    
    Package: 8-12 LPA
    """
    
    job_req = parser.parse_job_description(sample_jd)
    
    print("\n📋 Parsed Job Requirements:")
    print(f"Role: {job_req.role_title}")
    print(f"Company: {job_req.company}")
    print(f"Must-have skills: {job_req.must_have_skills}")
    print(f"Good-to-have skills: {job_req.good_to_have_skills}")
    
    return len(job_req.must_have_skills) > 0

if __name__ == "__main__":
    test_job_parser()
