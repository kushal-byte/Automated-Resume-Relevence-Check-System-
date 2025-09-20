# matchers/final_scorer.py
from matchers.hard_matcher import calculate_hard_match_score, calculate_fuzzy_match
from matchers.semantic_matcher import SemanticMatcher

class ResumeScorer:
    def __init__(self):
        self.semantic_matcher = SemanticMatcher()
        
    def calculate_final_score(self, resume_data, jd_data):
        """Calculate weighted final score combining all factors"""
        
        # Step 1: Hard Match (Keywords)
        hard_match = calculate_hard_match_score(
            resume_data["skills"], 
            jd_data["skills"]
        )
        
        # Step 2: Semantic Match (AI Embeddings)
        semantic_match = self.semantic_matcher.calculate_semantic_score(
            resume_data["raw_text"],
            jd_data["raw_text"]
        )
        
        # Step 3: Fuzzy Match
        fuzzy_skills = calculate_fuzzy_match(
            resume_data["raw_text"],
            jd_data["skills"]
        )
        fuzzy_bonus = len(fuzzy_skills) * 2  # 2 points per fuzzy match
        
        # Weighted scoring formula
        final_score = (
            0.4 * hard_match["score"] +           # 40% keyword match
            0.5 * semantic_match["score"] +       # 50% semantic similarity  
            0.1 * min(fuzzy_bonus, 20)           # 10% fuzzy bonus (max 20)
        )
        
        # Generate verdict
        verdict = self.get_verdict(final_score)
        
        return {
            "final_score": round(final_score, 2),
            "verdict": verdict,
            "breakdown": {
                "hard_match": hard_match,
                "semantic_match": semantic_match,
                "fuzzy_matches": fuzzy_skills
            },
            "suggestions": self.generate_suggestions(hard_match["missing_skills"])
        }
    
    def get_verdict(self, score):
        """Convert score to verdict categories"""
        if score >= 80:
            return "High Suitability"
        elif score >= 60:
            return "Medium Suitability"  
        else:
            return "Low Suitability"
    
    def generate_suggestions(self, missing_skills):
        """Generate improvement suggestions"""
        if not missing_skills:
            return "Great match! No major skills missing."
        
        suggestions = []
        if len(missing_skills) <= 3:
            suggestions.append(f"Consider adding skills: {', '.join(missing_skills[:3])}")
        else:
            suggestions.append(f"Focus on key skills: {', '.join(missing_skills[:3])}")
            suggestions.append("Consider relevant projects or certifications")
            
        return suggestions
