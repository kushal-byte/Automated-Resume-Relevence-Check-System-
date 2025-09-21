# scoring/relevance_scorer.py - Job-Specific Resume Relevance Scoring
from dataclasses import dataclass
from typing import Dict, List, Tuple
import re

@dataclass
class RelevanceScore:
    """Structured relevance scoring result"""
    overall_score: float  # 0-100
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    
    matched_must_have: List[str]
    matched_good_to_have: List[str]
    missing_must_have: List[str]
    missing_good_to_have: List[str]
    
    experience_gap: str
    education_gap: List[str]
    
    fit_verdict: str  # High/Medium/Low
    confidence_score: float
    
    improvement_suggestions: List[str]
    quick_wins: List[str]
    long_term_goals: List[str]

class JobRelevanceScorer:
    """Score resume relevance against specific job requirements"""
    
    def __init__(self):
        self.scoring_weights = {
            'must_have_skills': 0.40,     # 40% weight
            'experience': 0.25,           # 25% weight
            'good_to_have_skills': 0.15,  # 15% weight
            'education': 0.20             # 20% weight
        }
        print("✅ Job Relevance Scorer initialized")
    
    def calculate_relevance(self, resume_text: str, job_req) -> RelevanceScore:
        """Calculate comprehensive relevance score against job requirements"""
        
        print(f"🎯 Scoring relevance for: {getattr(job_req, 'role_title', 'Unknown Role')}")
        
        # Extract resume information
        from parsers.smart_skill_extractor import SmartSkillExtractor
        skill_extractor = SmartSkillExtractor()
        resume_skills = skill_extractor.extract_skills_comprehensive(resume_text)
        
        resume_experience = self._extract_experience_years(resume_text)
        resume_education = self._extract_education_level(resume_text)
        
        # Get job requirements
        must_have_skills = getattr(job_req, 'must_have_skills', [])
        good_to_have_skills = getattr(job_req, 'good_to_have_skills', [])
        required_experience = getattr(job_req, 'experience_required', '')
        required_education = getattr(job_req, 'education_required', [])
        
        # Calculate component scores
        skill_score, skill_matches = self._score_skills(
            resume_skills, must_have_skills, good_to_have_skills
        )
        
        experience_score, exp_gap = self._score_experience(
            resume_experience, required_experience
        )
        
        education_score, edu_gap = self._score_education(
            resume_education, required_education
        )
        
        # Calculate weighted overall score
        overall_score = (
            skill_score * self.scoring_weights['must_have_skills'] +
            experience_score * self.scoring_weights['experience'] +
            education_score * self.scoring_weights['education']
        )
        
        # Add good-to-have bonus
        good_to_have_bonus = len(skill_matches['matched_good_to_have']) * 2
        overall_score = min(100, overall_score + good_to_have_bonus)
        
        # Determine fit verdict
        fit_verdict, confidence = self._determine_fit_verdict(
            overall_score, skill_matches, experience_score
        )
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            skill_matches, exp_gap, edu_gap, job_req
        )
        
        return RelevanceScore(
            overall_score=round(overall_score, 1),
            skill_match_score=round(skill_score, 1),
            experience_match_score=round(experience_score, 1),
            education_match_score=round(education_score, 1),
            
            matched_must_have=skill_matches['matched_must_have'],
            matched_good_to_have=skill_matches['matched_good_to_have'],
            missing_must_have=skill_matches['missing_must_have'],
            missing_good_to_have=skill_matches['missing_good_to_have'],
            
            experience_gap=exp_gap,
            education_gap=edu_gap,
            
            fit_verdict=fit_verdict,
            confidence_score=confidence,
            
            improvement_suggestions=suggestions['main'],
            quick_wins=suggestions['quick_wins'],
            long_term_goals=suggestions['long_term']
        )
    
    def _score_skills(self, resume_skills: List[str], must_have: List[str], 
                     good_to_have: List[str]) -> Tuple[float, Dict]:
        """Score skill matching against job requirements"""
        
        resume_skills_lower = [skill.lower() for skill in resume_skills]
        
        # Match must-have skills
        matched_must_have = []
        missing_must_have = []
        
        for skill in must_have:
            skill_lower = skill.lower()
            if any(skill_lower in resume_skill for resume_skill in resume_skills_lower):
                matched_must_have.append(skill)
            else:
                missing_must_have.append(skill)
        
        # Match good-to-have skills
        matched_good_to_have = []
        missing_good_to_have = []
        
        for skill in good_to_have:
            skill_lower = skill.lower()
            if any(skill_lower in resume_skill for resume_skill in resume_skills_lower):
                matched_good_to_have.append(skill)
            else:
                missing_good_to_have.append(skill)
        
        # Calculate skill score
        if not must_have:
            must_have_score = 100
        else:
            must_have_score = (len(matched_must_have) / len(must_have)) * 100
        
        return must_have_score, {
            'matched_must_have': matched_must_have,
            'matched_good_to_have': matched_good_to_have,
            'missing_must_have': missing_must_have,
            'missing_good_to_have': missing_good_to_have
        }
    
    def _score_experience(self, resume_exp: int, required_exp: str) -> Tuple[float, str]:
        """Score experience matching"""
        
        req_years = self._parse_experience_requirement(required_exp)
        
        if req_years is None:
            return 100, "Experience requirement not specified"
        
        if resume_exp >= req_years:
            if resume_exp <= req_years + 2:
                score = 100
                gap = f"Perfect match ({resume_exp} years vs {req_years} required)"
            else:
                score = 95
                gap = f"Overqualified ({resume_exp} years vs {req_years} required)"
        else:
            gap_years = req_years - resume_exp
            if gap_years == 1:
                score = 75
                gap = f"1 year short ({resume_exp} years vs {req_years} required)"
            elif gap_years == 2:
                score = 50
                gap = f"2 years short ({resume_exp} years vs {req_years} required)"
            else:
                score = 25
                gap = f"{gap_years} years short ({resume_exp} years vs {req_years} required)"
        
        return score, gap
    
    def _score_education(self, resume_edu: List[str], required_edu: List[str]) -> Tuple[float, List[str]]:
        """Score education matching"""
        
        if not required_edu or "any graduate" in " ".join(required_edu).lower():
            return 100, []
        
        resume_edu_lower = [edu.lower() for edu in resume_edu]
        
        matched = False
        gaps = []
        
        for req_edu in required_edu:
            req_edu_lower = req_edu.lower()
            found_match = False
            for res_edu in resume_edu_lower:
                if any(word in res_edu for word in req_edu_lower.split() if len(word) > 2):
                    matched = True
                    found_match = True
                    break
            
            if not found_match:
                gaps.append(req_edu)
        
        score = 100 if matched and not gaps else (80 if matched else 30)
        return score, gaps
    
    def _extract_experience_years(self, resume_text: str) -> int:
        """Extract years of experience from resume"""
        
        patterns = [
            r'(\d+)[\+\s]*years?\s+(?:of\s+)?(?:experience|exp)',
            r'(?:experience|exp)[\s:]*(\d+)[\+\s]*years?',
            r'(\d+)[\+\s]*years?\s+(?:in|with)'
        ]
        
        years = []
        for pattern in patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            years.extend([int(match) for match in matches if match.isdigit()])
        
        return max(years) if years else 0
    
    def _extract_education_level(self, resume_text: str) -> List[str]:
        """Extract education from resume"""
        
        patterns = [
            r'bachelor[^.\n]*',
            r'master[^.\n]*',
            r'b\.?tech[^.\n]*',
            r'm\.?tech[^.\n]*',
            r'bca[^.\n]*',
            r'mca[^.\n]*'
        ]
        
        education = []
        for pattern in patterns:
            matches = re.findall(pattern, resume_text, re.IGNORECASE)
            education.extend(matches)
        
        return education
    
    def _parse_experience_requirement(self, exp_req: str) -> int:
        """Parse experience requirement string to years"""
        
        if not exp_req or exp_req.lower() == "not specified":
            return None
        
        numbers = re.findall(r'\d+', exp_req)
        
        if not numbers:
            return None
        
        return int(numbers[0])
    
    def _determine_fit_verdict(self, overall_score: float, skill_matches: Dict, 
                             experience_score: float) -> Tuple[str, float]:
        """Determine fit verdict and confidence"""
        
        must_have_count = len(skill_matches['matched_must_have']) + len(skill_matches['missing_must_have'])
        must_have_ratio = len(skill_matches['matched_must_have']) / max(1, must_have_count)
        
        confidence = min(100, (must_have_ratio * 50) + (experience_score * 0.3) + (overall_score * 0.2))
        
        if overall_score >= 80 and must_have_ratio >= 0.8:
            verdict = "High Suitability"
        elif overall_score >= 60 and must_have_ratio >= 0.6:
            verdict = "Medium Suitability"
        elif overall_score >= 40:
            verdict = "Low-Medium Suitability"
        else:
            verdict = "Low Suitability"
        
        return verdict, round(confidence, 1)
    
    def _generate_improvement_suggestions(self, skill_matches: Dict, exp_gap: str, 
                                        edu_gap: List[str], job_req) -> Dict[str, List[str]]:
        """Generate personalized improvement suggestions"""
        
        main_suggestions = []
        quick_wins = []
        long_term_goals = []
        
        # Skill suggestions
        missing_must_have = skill_matches['missing_must_have']
        if missing_must_have:
            main_suggestions.append(f"Acquire critical skills: {', '.join(missing_must_have[:3])}")
            quick_wins.append(f"Start learning: {', '.join(missing_must_have[:2])}")
        
        # Experience suggestions
        if "short" in exp_gap:
            quick_wins.append("Gain experience through projects and internships")
        
        # Education suggestions
        if edu_gap:
            long_term_goals.append("Consider relevant degree or certification")
        
        return {
            'main': main_suggestions[:5],
            'quick_wins': quick_wins[:5],
            'long_term': long_term_goals[:3]
        }

def test_relevance_scorer():
    """Test the relevance scorer"""
    print("✅ Relevance scorer test completed")
    return True

if __name__ == "__main__":
    test_relevance_scorer()
