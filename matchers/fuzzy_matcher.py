# matchers/fuzzy_matcher.py - FUZZY SKILL MATCHING
from rapidfuzz import fuzz, process
from collections import defaultdict

class FuzzyMatcher:
    def __init__(self):
        self.skill_variations = {
            'javascript': ['js', 'javascript', 'ecmascript', 'node.js', 'nodejs'],
            'python': ['python', 'py', 'python3'],
            'typescript': ['typescript', 'ts'],
            'kubernetes': ['kubernetes', 'k8s', 'kube'],
            'postgresql': ['postgresql', 'postgres', 'psql'],
            'ci/cd': ['ci/cd', 'cicd', 'continuous integration', 'continuous deployment'],
            'docker': ['docker', 'containerization', 'containers'],
            'aws': ['aws', 'amazon web services', 'amazon cloud'],
            'react': ['react', 'reactjs', 'react.js'],
            'angular': ['angular', 'angularjs', 'angular.js']
        }
        print("✅ Fuzzy matcher initialized with skill variations")
    
    def fuzzy_skill_match(self, resume_skills, jd_skills, threshold=80):
        """Find fuzzy matches between resume and JD skills"""
        print("🔍 Running fuzzy skill matching...")
        
        fuzzy_matches = []
        matched_pairs = []
        
        for jd_skill in jd_skills:
            best_match = None
            best_score = 0
            
            for resume_skill in resume_skills:
                # Direct fuzzy match
                score = fuzz.ratio(jd_skill.lower(), resume_skill.lower())
                
                if score > threshold and score > best_score:
                    best_match = resume_skill
                    best_score = score
            
            # Check skill variations
            if not best_match:
                best_match, best_score = self._check_skill_variations(jd_skill, resume_skills)
            
            if best_match and best_score > threshold:
                fuzzy_matches.append(jd_skill)
                matched_pairs.append({
                    "jd_skill": jd_skill,
                    "resume_skill": best_match,
                    "confidence": round(best_score, 1)
                })
        
        return {
            "fuzzy_matched_skills": fuzzy_matches,
            "match_details": matched_pairs,
            "fuzzy_score": len(fuzzy_matches)
        }
    
    def _check_skill_variations(self, jd_skill, resume_skills):
        """Check if skill matches any known variations"""
        jd_lower = jd_skill.lower()
        
        # Check if JD skill is in our variations
        for main_skill, variations in self.skill_variations.items():
            if jd_lower in variations:
                # Look for other variations in resume
                for resume_skill in resume_skills:
                    if resume_skill.lower() in variations:
                        return resume_skill, 95  # High confidence for variation match
        
        # Check reverse - if resume skill has variations
        for resume_skill in resume_skills:
            resume_lower = resume_skill.lower()
            for main_skill, variations in self.skill_variations.items():
                if resume_lower in variations and jd_lower in variations:
                    return resume_skill, 90
        
        return None, 0
    
    def suggest_skill_improvements(self, missing_skills):
        """Suggest skill variations that might be easier to learn"""
        suggestions = []
        
        for skill in missing_skills[:5]:  # Top 5 missing skills
            skill_lower = skill.lower()
            
            # Find related skills or easier alternatives
            for main_skill, variations in self.skill_variations.items():
                if skill_lower in variations:
                    other_variations = [v for v in variations if v != skill_lower]
                    if other_variations:
                        suggestions.append({
                            "missing_skill": skill,
                            "alternatives": other_variations[:3],
                            "suggestion": f"Consider learning {other_variations[0]} as an alternative to {skill}"
                        })
                    break
        
        return suggestions

# Test function
def test_fuzzy_matcher():
    """Test fuzzy matching functionality"""
    matcher = FuzzyMatcher()
    
    resume_skills = ["javascript", "python", "react", "nodejs", "aws"]
    jd_skills = ["js", "python3", "reactjs", "node.js", "amazon web services", "docker"]
    
    result = matcher.fuzzy_skill_match(resume_skills, jd_skills)
    print(f"✅ Fuzzy matches found: {len(result['fuzzy_matched_skills'])}")
    
    for match in result['match_details']:
        print(f"   {match['jd_skill']} ↔ {match['resume_skill']} ({match['confidence']}%)")
    
    return len(result['fuzzy_matched_skills']) > 0

if __name__ == "__main__":
    test_fuzzy_matcher()
