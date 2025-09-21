# llm_analysis/langgraph_pipeline.py - Structured Analysis Pipeline
from langgraph.graph import StateGraph, END
from typing import Dict, List, TypedDict
import json
from llm_analysis.llm_analyzer import LLMResumeAnalyzer

class AnalysisState(TypedDict):
    """State object for the analysis pipeline"""
    resume_text: str
    jd_text: str
    basic_scores: Dict
    enhanced_skills: Dict
    llm_analysis: Dict
    improvement_roadmap: Dict
    final_result: Dict
    current_step: str
    errors: List[str]

class ResumeAnalysisPipeline:
    """LangGraph-powered structured analysis pipeline"""
    
    def __init__(self, model="x-ai/grok-4-fast:free"):
        self.llm_analyzer = LLMResumeAnalyzer(model=model)
        self.graph = self._create_pipeline()
        print("✅ LangGraph pipeline initialized")
    
    def _create_pipeline(self):
        """Create the structured analysis pipeline"""
        
        # Define the workflow graph
        workflow = StateGraph(AnalysisState)
        
        # Add nodes (analysis steps)
        workflow.add_node("skills_extraction", self._extract_skills_node)
        workflow.add_node("llm_analysis", self._llm_analysis_node)
        workflow.add_node("roadmap_generation", self._roadmap_generation_node)
        workflow.add_node("final_compilation", self._final_compilation_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define the flow
        workflow.set_entry_point("skills_extraction")
        
        # Add edges (flow control)
        workflow.add_edge("skills_extraction", "llm_analysis")
        workflow.add_edge("llm_analysis", "roadmap_generation")
        workflow.add_edge("roadmap_generation", "final_compilation")
        workflow.add_edge("final_compilation", END)
        workflow.add_edge("error_handler", END)
        
        # Add conditional edges for error handling
        workflow.add_conditional_edges(
            "skills_extraction",
            self._should_continue,
            {
                "continue": "llm_analysis",
                "error": "error_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "llm_analysis", 
            self._should_continue,
            {
                "continue": "roadmap_generation",
                "error": "error_handler"
            }
        )
        
        return workflow.compile()
    
    def _should_continue(self, state: AnalysisState) -> str:
        """Decide whether to continue or handle errors"""
        if state.get("errors"):
            return "error"
        return "continue"
    
    def _extract_skills_node(self, state: AnalysisState) -> AnalysisState:
        """Node 1: Enhanced skills extraction"""
        try:
            state["current_step"] = "skills_extraction"
            print("🔍 LangGraph: Extracting skills...")
            
            # Enhanced skills extraction
            enhanced_skills = self.llm_analyzer.enhance_skills_extraction(state["resume_text"])
            state["enhanced_skills"] = enhanced_skills
            
            print("✅ LangGraph: Skills extraction completed")
            return state
            
        except Exception as e:
            state["errors"].append(f"Skills extraction failed: {str(e)}")
            return state
    
    def _llm_analysis_node(self, state: AnalysisState) -> AnalysisState:
        """Node 2: LLM analysis"""
        try:
            state["current_step"] = "llm_analysis"
            print("🧠 LangGraph: Running LLM analysis...")
            
            # LLM analysis
            llm_analysis = self.llm_analyzer.analyze_resume_vs_jd(
                state["resume_text"],
                state["jd_text"], 
                state["basic_scores"]
            )
            state["llm_analysis"] = llm_analysis
            
            print("✅ LangGraph: LLM analysis completed")
            return state
            
        except Exception as e:
            state["errors"].append(f"LLM analysis failed: {str(e)}")
            return state
    
    def _roadmap_generation_node(self, state: AnalysisState) -> AnalysisState:
        """Node 3: Improvement roadmap generation"""
        try:
            state["current_step"] = "roadmap_generation"
            print("🗺️ LangGraph: Generating improvement roadmap...")
            
            # Generate roadmap
            roadmap = self.llm_analyzer.generate_improvement_roadmap(state["llm_analysis"])
            state["improvement_roadmap"] = roadmap
            
            print("✅ LangGraph: Roadmap generation completed")
            return state
            
        except Exception as e:
            state["errors"].append(f"Roadmap generation failed: {str(e)}")
            return state
    
    def _final_compilation_node(self, state: AnalysisState) -> AnalysisState:
        """Node 4: Final result compilation"""
        try:
            state["current_step"] = "final_compilation"
            print("📊 LangGraph: Compiling final results...")
            
            # Compile final result
            final_result = {
                "basic_scores": state["basic_scores"],
                "enhanced_skills": state["enhanced_skills"],
                "llm_analysis": state["llm_analysis"],
                "improvement_roadmap": state["improvement_roadmap"],
                "pipeline_status": "completed",
                "processing_steps": ["skills_extraction", "llm_analysis", "roadmap_generation", "compilation"]
            }
            
            state["final_result"] = final_result
            print("✅ LangGraph: Pipeline completed successfully")
            return state
            
        except Exception as e:
            state["errors"].append(f"Final compilation failed: {str(e)}")
            return state
    
    def _error_handler_node(self, state: AnalysisState) -> AnalysisState:
        """Error handling node"""
        print(f"❌ LangGraph: Handling errors - {len(state['errors'])} error(s)")
        
        state["final_result"] = {
            "pipeline_status": "failed",
            "errors": state["errors"],
            "last_successful_step": state.get("current_step", "unknown"),
            "partial_results": {
                "basic_scores": state.get("basic_scores", {}),
                "enhanced_skills": state.get("enhanced_skills", {}),
                "llm_analysis": state.get("llm_analysis", {}),
                "improvement_roadmap": state.get("improvement_roadmap", {})
            }
        }
        return state
    
    def run_structured_analysis(self, resume_text: str, jd_text: str, basic_scores: Dict) -> Dict:
        """Run the complete structured analysis pipeline"""
        print("🚀 Starting LangGraph structured analysis pipeline...")
        
        # Initialize state
        initial_state = AnalysisState(
            resume_text=resume_text,
            jd_text=jd_text,
            basic_scores=basic_scores,
            enhanced_skills={},
            llm_analysis={},
            improvement_roadmap={},
            final_result={},
            current_step="initializing",
            errors=[]
        )
        
        # Run the pipeline
        try:
            final_state = self.graph.invoke(initial_state)
            
            print("✅ LangGraph pipeline execution completed")
            return final_state["final_result"]
            
        except Exception as e:
            print(f"❌ LangGraph pipeline failed: {e}")
            return {
                "pipeline_status": "critical_failure",
                "error": str(e),
                "basic_scores": basic_scores
            }

# Test function
def test_langgraph_pipeline():
    """Test the LangGraph pipeline"""
    pipeline = ResumeAnalysisPipeline()
    
    sample_resume = "Python developer with React experience"
    sample_jd = "Looking for Python developer with React skills"
    sample_basic_scores = {
        "score": 75,
        "matched_skills": ["python", "react"],
        "missing_skills": ["docker"],
        "matched_count": 2,
        "total_jd_skills": 3
    }
    
    result = pipeline.run_structured_analysis(sample_resume, sample_jd, sample_basic_scores)
    print(f"✅ LangGraph test completed: {result.get('pipeline_status', 'unknown')}")
    return result.get('pipeline_status') == 'completed'

if __name__ == "__main__":
    test_langgraph_pipeline()
