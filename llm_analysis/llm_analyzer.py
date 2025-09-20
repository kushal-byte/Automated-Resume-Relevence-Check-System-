# llm_analysis/llm_analyzer.py
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from llm_analysis.prompt_templates import (
    RESUME_ANALYSIS_PROMPT, 
    IMPROVEMENT_ROADMAP_PROMPT, 
    SKILLS_ENHANCEMENT_PROMPT
)

load_dotenv()

class LLMResumeAnalyzer:
    def __init__(self, model=None):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file")
        
        # Use the provided model, or fall back to environment variable/default
        llm_model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0.2,  # Low for consistency
            api_key=api_key
        )
        
        print(f"✅ LLM Analyzer initialized successfully with model: {llm_model}")
    
    def analyze_resume_vs_jd(self, resume_text, jd_text, keyword_match_data):
        """Comprehensive LLM-powered resume analysis"""
        print("🤖 Running LLM analysis...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert HR recruiter and technical hiring manager."),
                ("human", RESUME_ANALYSIS_PROMPT)
            ])
            
            chain = prompt | self.llm
            
            response = chain.invoke({
                "resume_text": resume_text[:3000],  # Truncate to avoid token limits
                "jd_text": jd_text[:2000],
                "matched_count": keyword_match_data.get("matched_count", 0),
                "total_skills": keyword_match_data.get("total_jd_skills", 0),
                "matched_skills": ", ".join(keyword_match_data.get("matched_skills", [])),
                "missing_skills": ", ".join(keyword_match_data.get("missing_skills", [])),
                "coverage_percentage": keyword_match_data.get("score", 0)
            })
            
            # Parse JSON response
            analysis = json.loads(response.content)
            print("✅ LLM analysis completed successfully")
            return analysis
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parsing error: {e}")
            return self._create_fallback_analysis(keyword_match_data)
        except Exception as e:
            print(f"❌ LLM analysis error: {e}")
            return self._create_fallback_analysis(keyword_match_data)
    
    def generate_improvement_roadmap(self, analysis_results):
        """Generate detailed improvement roadmap"""
        print("🗺️ Generating improvement roadmap...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a career coach specializing in tech careers."),
                ("human", IMPROVEMENT_ROADMAP_PROMPT)
            ])
            
            chain = prompt | self.llm
            
            response = chain.invoke({
                "analysis_results": json.dumps(analysis_results, indent=2)
            })
            
            roadmap = json.loads(response.content)
            print("✅ Improvement roadmap generated successfully")
            return roadmap
            
        except Exception as e:
            print(f"❌ Roadmap generation error: {e}")
            return self._create_fallback_roadmap()
    
    def enhance_skills_extraction(self, text):
        """Use LLM to extract and categorize skills more intelligently"""
        print("🧠 Enhancing skills extraction with LLM...")
        
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a technical skills extraction specialist."),
                ("human", SKILLS_ENHANCEMENT_PROMPT)
            ])
            
            chain = prompt | self.llm
            
            response = chain.invoke({
                "text": text[:2000]  # Truncate to avoid token limits
            })
            
            skills_data = json.loads(response.content)
            print("✅ Skills enhancement completed")
            return skills_data
            
        except Exception as e:
            print(f"❌ Skills enhancement error: {e}")
            return {"all_technical_skills": [], "error": str(e)}
    
    def _create_fallback_analysis(self, keyword_data):
        """Fallback analysis when LLM fails"""
        return {
            "overall_fit_score": max(1, int(keyword_data.get("score", 0) / 10)),
            "experience_alignment": "Unable to assess - manual review needed",
            "key_strengths": ["Technical skills present in resume"],
            "critical_gaps": keyword_data.get("missing_skills", [])[:3],
            "role_suitability": "Medium - based on keyword match only",
            "improvement_suggestions": ["Add missing technical skills", "Improve resume formatting"],
            "recommended_skills_to_learn": keyword_data.get("missing_skills", [])[:3],
            "project_recommendations": ["Build projects showcasing missing skills"],
            "certification_suggestions": ["Relevant industry certifications"],
            "interview_readiness": "Moderate preparation needed",
            "salary_expectations": "Market standard for skill level",
            "final_verdict": "Automated analysis only - requires manual review"
        }
    
    def _create_fallback_roadmap(self):
        """Fallback roadmap when LLM fails"""
        return {
            "immediate_actions": ["Update resume with missing skills", "Clean up resume formatting"],
            "week_1_plan": ["Research missing skills", "Start online tutorials"],
            "month_1_plan": ["Complete beginner courses", "Build first project"],
            "month_3_plan": ["Build portfolio", "Apply for relevant positions"],
            "priority_skills": ["As identified in job description"],
            "learning_resources": {
                "free_courses": ["freeCodeCamp", "Coursera free courses"],
                "paid_courses": ["Udemy", "Pluralsight"],
                "books": ["Technical books for identified skills"],
                "practice_platforms": ["LeetCode", "HackerRank"]
            },
            "portfolio_improvements": ["Build 2-3 projects showcasing skills"],
            "networking_suggestions": ["Join LinkedIn groups", "Attend tech meetups"],
            "quick_wins": ["Update LinkedIn profile", "Get recommendations"],
            "estimated_timeline": "3-6 months for significant improvement"
        }

# Test LLM connectivity
def test_llm_connection():
    """Test if LLM is working"""
    try:
        analyzer = LLMResumeAnalyzer()
        print("🧪 Testing LLM connection...")
        
        # Simple test
        result = analyzer.llm.invoke("Say 'Hello, LLM is working!' in JSON format: {\"status\": \"working\", \"message\": \"Hello, LLM is working!\"}")
        test_response = json.loads(result.content)
        
        print(f"✅ LLM Test Result: {test_response}")
        return True
        
    except Exception as e:
        print(f"❌ LLM Test Failed: {e}")
        return False

if __name__ == "__main__":
    test_llm_connection()
