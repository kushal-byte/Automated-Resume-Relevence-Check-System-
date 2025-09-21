# main.py - COMPLETE WITH LANGGRAPH + LANGSMITH
import os
from dotenv import load_dotenv

import time
# Load environment variables
load_dotenv()

# --- Configuration for OpenRouter ---
LLM_MODEL = "x-ai/grok-4-fast:free"  # Updated model name

# Set environment variables for the OpenAI client to use OpenRouter
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY", "")

# Import all modules - ENHANCED WITH NEW COMPONENTS
from parsers.pdf_parser import extract_text_pymupdf
from parsers.docx_parser import extract_text_docx
from parsers.cleaner import clean_text
from parsers.section_splitter import split_sections
from parsers.skill_extractor import extract_skills
from parsers.jd_parser import parse_jd
from llm_analysis.llm_analyzer import LLMResumeAnalyzer, test_llm_connection

# ENHANCED COMPONENTS
try:
    from matchers.final_scorer import EnhancedResumeScorer
    ENHANCED_SCORING = True
    print("✅ Enhanced scoring components loaded")
except ImportError:
    print("⚠️ Enhanced components not found, using basic scoring")
    ENHANCED_SCORING = False

# LANGGRAPH & LANGSMITH COMPONENTS
try:
    from llm_analysis.langgraph_pipeline import ResumeAnalysisPipeline
    from llm_analysis.langsmith_logger import logger, trace_llm_analysis
    ADVANCED_PIPELINE = True
    print("✅ LangGraph + LangSmith components loaded")
except ImportError:
    print("⚠️ LangGraph/LangSmith not found - install with: pip install langgraph langsmith")
    ADVANCED_PIPELINE = False

def load_file(file_path):
    """Load text from various file formats"""
    if file_path.endswith(".pdf"):
        return extract_text_pymupdf(file_path)
    elif file_path.endswith(".docx"):
        return extract_text_docx(file_path)
    elif file_path.endswith(".txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError("Unsupported file format")

def calculate_basic_scores(resume_skills, jd_skills):
    """Calculate basic matching scores (fallback)"""
    if not jd_skills:
        return {"score": 0, "matched_skills": [], "missing_skills": [], "matched_count": 0, "total_jd_skills": 0}
    
    matched_skills = list(set(resume_skills) & set(jd_skills))
    missing_skills = list(set(jd_skills) - set(resume_skills))
    
    coverage_score = len(matched_skills) / len(jd_skills) * 100
    
    return {
        "score": round(coverage_score, 2),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "matched_count": len(matched_skills),
        "total_jd_skills": len(jd_skills)
    }

@trace_llm_analysis if ADVANCED_PIPELINE else lambda x: x  # LangSmith tracing decorator
def complete_ai_analysis(resume_file, jd_file):
    """Complete AI-powered resume analysis with LangGraph + LangSmith"""
    
    print("🚀 STARTING ENHANCED AI-POWERED RESUME ANALYSIS")
    if ADVANCED_PIPELINE:
        print("   🔗 LangGraph: Structured pipeline")
        print("   🔍 LangSmith: Observability & logging")
    print("=" * 65)
    
    # Start LangSmith trace
    trace_id = None
    if ADVANCED_PIPELINE:
        trace_id = logger.start_trace("complete_resume_analysis", {
            "resume_file": resume_file,
            "jd_file": jd_file
        })
    
    # Test LLM connection first
    if not test_llm_connection():
        print("⚠️ LLM connection failed, continuing with mock analysis...")
    
    try:
        # Initialize components
        print("\n🔧 INITIALIZING ENHANCED COMPONENTS...")
        llm_analyzer = LLMResumeAnalyzer(model=LLM_MODEL)
        
        # LangGraph pipeline
        if ADVANCED_PIPELINE:
            pipeline = ResumeAnalysisPipeline(model=LLM_MODEL)
            print("✅ LangGraph pipeline initialized")
        
        if ENHANCED_SCORING:
            enhanced_scorer = EnhancedResumeScorer()
            print("✅ Enhanced scorer with semantic matching, fuzzy matching, and NLP entities")
        else:
            enhanced_scorer = None
            print("⚠️ Using basic scoring (install enhanced components for full tech stack)")
        
        # Step 1: Load and parse files
        print("\n📄 LOADING FILES...")
        resume_raw = load_file(resume_file)
        jd_raw = load_file(jd_file)
        print(f"✅ Resume loaded: {len(resume_raw)} chars")
        print(f"✅ JD loaded: {len(jd_raw)} chars")
        
        # Step 2: Process resume
        print("\n🔍 PROCESSING RESUME...")
        resume_clean = clean_text(resume_raw)
        resume_sections = split_sections(resume_clean)
        resume_skills = extract_skills(" ".join(resume_sections.values()))
        print(f"✅ Resume sections: {list(resume_sections.keys())}")
        print(f"✅ Resume skills found: {len(resume_skills)}")
        
        # Step 3: Process JD
        print("\n🔍 PROCESSING JOB DESCRIPTION...")
        jd_data = parse_jd(jd_raw)
        jd_skills = jd_data["skills"]
        print(f"✅ JD role: {jd_data['role']}")
        print(f"✅ JD skills found: {len(jd_skills)}")
        
        # Step 4: ENHANCED COMPREHENSIVE SCORING
        if ENHANCED_SCORING:
            print("\n🧮 RUNNING COMPREHENSIVE ANALYSIS...")
            print("   🔍 Hard Match: TF-IDF + keyword matching")
            print("   🧠 Semantic Match: Embeddings + cosine similarity") 
            print("   🔄 Fuzzy Match: Skill variations + rapidfuzz")
            print("   📊 Entity Analysis: spaCy NLP + experience extraction")
            
            comprehensive_result = enhanced_scorer.calculate_comprehensive_score(
                {"raw_text": resume_clean, "skills": resume_skills},
                {"raw_text": jd_raw, "skills": jd_skills}
            )
            
            basic_scores = {
                "score": comprehensive_result["breakdown"]["hard_match"]["score"],
                "matched_skills": comprehensive_result["breakdown"]["hard_match"]["matched_skills"],
                "missing_skills": comprehensive_result["breakdown"]["hard_match"]["missing_skills"],
                "matched_count": comprehensive_result["breakdown"]["hard_match"]["matched_count"],
                "total_jd_skills": comprehensive_result["breakdown"]["hard_match"]["total_jd_skills"]
            }
            
        else:
            # Fallback to basic scoring
            print("\n⚙️ CALCULATING BASIC SCORES...")
            basic_scores = calculate_basic_scores(resume_skills, jd_skills)
            comprehensive_result = None
            print(f"✅ Keyword match: {basic_scores['score']:.1f}%")
            print(f"✅ Matched skills: {basic_scores['matched_count']}/{basic_scores['total_jd_skills']}")
        
        # Step 5: LangGraph Structured Pipeline (if available)
        if ADVANCED_PIPELINE:
            print("\n🔗 RUNNING LANGGRAPH STRUCTURED PIPELINE...")
            pipeline_result = pipeline.run_structured_analysis(resume_clean, jd_raw, basic_scores)
            
            if pipeline_result.get("pipeline_status") == "completed":
                llm_analysis = pipeline_result["llm_analysis"]
                improvement_roadmap = pipeline_result["improvement_roadmap"]
                print("✅ LangGraph pipeline completed successfully")
            else:
                print("⚠️ LangGraph pipeline failed, using fallback analysis")
                llm_analysis = llm_analyzer.analyze_resume_vs_jd(resume_clean, jd_raw, basic_scores)
                improvement_roadmap = llm_analyzer.generate_improvement_roadmap(llm_analysis)
        else:
            # Standard LLM Analysis
            print("\n🧠 RUNNING LLM ANALYSIS...")
            llm_analysis = llm_analyzer.analyze_resume_vs_jd(resume_clean, jd_raw, basic_scores)
            
            print("\n🗺️ GENERATING IMPROVEMENT ROADMAP...")
            improvement_roadmap = llm_analyzer.generate_improvement_roadmap(llm_analysis)
        
        # Step 6: Display enhanced results
        if ENHANCED_SCORING:
            display_enhanced_results(comprehensive_result, llm_analysis, improvement_roadmap)
        else:
            display_structured_results(basic_scores, llm_analysis, improvement_roadmap, {})
        
        # Log success metrics (LangSmith)
        if ADVANCED_PIPELINE and trace_id:
            logger.log_metrics({
                "analysis_success": True,
                "resume_length": len(resume_raw),
                "jd_length": len(jd_raw),
                "skills_found": len(resume_skills),
                "pipeline_status": pipeline_result.get("pipeline_status", "fallback") if ADVANCED_PIPELINE else "standard",
                "enhanced_scoring": ENHANCED_SCORING
            })
            
            logger.end_trace(trace_id, {
                "pipeline_status": pipeline_result.get("pipeline_status", "fallback") if ADVANCED_PIPELINE else "standard",
                "final_score": llm_analysis.get("overall_fit_score", 0)
            }, "success")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        
        # Log error (LangSmith)
        if ADVANCED_PIPELINE and trace_id:
            logger.end_trace(trace_id, {}, "error", str(e))
            logger.log_metrics({
                "analysis_success": False,
                "error": str(e)
            })
        
        import traceback
        traceback.print_exc()

def display_enhanced_results(comprehensive_result, llm_analysis, roadmap):
    """Display enhanced results with full tech stack analysis"""
    
    print(f"\n{'='*75}")
    print("🎯 Automated Resume Relevance Check Report (Enhanced)")
    if ADVANCED_PIPELINE:
        print("   🔗 Powered by LangGraph + LangSmith")
    print("=" * 75)
    
    # Get breakdown
    breakdown = comprehensive_result["breakdown"]
    hard_match = breakdown["hard_match"]
    semantic_match = breakdown["semantic_match"]
    fuzzy_match = breakdown["fuzzy_match"]
    entity_analysis = breakdown["entity_analysis"]
    
    # RELEVANCE ANALYSIS - Enhanced 3 Steps
    print(f"\n📋 RELEVANCE ANALYSIS (Enhanced with Full Tech Stack)")
    print("-" * 60)
    
    # Step 1: Enhanced Hard Match
    print(f"\n🔍 STEP 1: ENHANCED HARD MATCH")
    print(f"   📊 TF-IDF Similarity: {hard_match.get('tfidf_similarity', 0):.1f}%")
    print(f"   🎯 Basic Coverage: {hard_match['basic_coverage']:.1f}%")
    print(f"   ⚖️  Combined Hard Score: {hard_match['score']:.1f}%")
    print(f"   ✅ Exact Matches: {hard_match['matched_count']}/{hard_match['total_jd_skills']} skills")
    print(f"   🔄 Fuzzy Matches: {fuzzy_match['fuzzy_score']} additional skills")
    
    # Display matched skills
    if hard_match['matched_skills']:
        print(f"   📝 Matched Skills: {', '.join(hard_match['matched_skills'][:8])}")
        if len(hard_match['matched_skills']) > 8:
            print(f"      ... and {len(hard_match['matched_skills']) - 8} more")
    
    # Display fuzzy matches
    if fuzzy_match.get('match_details'):
        print(f"   🔄 Fuzzy Matches Found:")
        for match in fuzzy_match['match_details'][:3]:
            print(f"      • {match['jd_skill']} ↔ {match['resume_skill']} ({match['confidence']}%)")
    
    # Step 2: Semantic Match with Embeddings
    print(f"\n🧠 STEP 2: SEMANTIC MATCH (Embeddings + Cosine Similarity)")
    print(f"   🤖 LLM Experience Score: {llm_analysis.get('overall_fit_score', 0)}/10")
    print(f"   📊 Embedding Similarity: {semantic_match.get('semantic_score', 0):.1f}%")
    print(f"   🔍 Context Understanding: {llm_analysis.get('experience_alignment', 'N/A')[:100]}...")
    
    # Entity Analysis Results
    print(f"\n📊 ENTITY ANALYSIS (spaCy NLP):")
    if entity_analysis.get('experience_years', 0) > 0:
        print(f"   💼 Experience Detected: {entity_analysis['experience_years']} years")
    if entity_analysis.get('education', {}).get('degrees'):
        print(f"   🎓 Education: {', '.join(entity_analysis['education']['degrees'])}")
    
    # Step 3: Enhanced Scoring & Verdict
    final_score = comprehensive_result["final_score"]
    print(f"\n⚖️ STEP 3: ENHANCED SCORING & VERDICT")
    print(f"   📐 Weighted Formula: Hard(40%) + Semantic(45%) + Fuzzy(10%) + Experience(3%) + Education(2%)")
    print(f"   🎯 Component Scores:")
    print(f"      • Hard Match: {hard_match['score']:.1f}%")
    print(f"      • Semantic: {semantic_match.get('semantic_score', 0):.1f}%") 
    print(f"      • Fuzzy Bonus: +{fuzzy_match['fuzzy_score'] * 3:.1f} points")
    if entity_analysis.get('experience_years', 0) > 0:
        print(f"      • Experience Bonus: +{min(entity_analysis['experience_years'] * 2, 10):.1f} points")
    print(f"   🏆 FINAL SCORE: {final_score}/100")
    
    # OUTPUT GENERATION
    print(f"\n📊 OUTPUT GENERATION")
    print("-" * 50)
    
    # Relevance Score
    print(f"\n🎯 RELEVANCE SCORE: {final_score}/100")
    
    # Enhanced Verdict
    verdict = comprehensive_result["verdict"]
    print(f"\n🏷️ VERDICT: {verdict}")
    
    # Missing Skills Analysis
    missing_skills = hard_match['missing_skills']
    print(f"\n❌ MISSING SKILLS/REQUIREMENTS:")
    for i, skill in enumerate(missing_skills[:8], 1):
        print(f"   {i}. {skill}")
    
    # Critical Gaps from LLM
    if llm_analysis.get('critical_gaps'):
        print(f"\n⚠️ CRITICAL GAPS (LLM Analysis):")
        for i, gap in enumerate(llm_analysis['critical_gaps'][:3], 1):
            print(f"   {i}. {gap}")
    
    # Enhanced Recommendations
    print(f"\n💡 ENHANCED SUGGESTIONS:")
    recommendations = comprehensive_result.get("recommendations", [])
    
    if roadmap and roadmap.get('immediate_actions'):
        print(f"\n   📋 IMMEDIATE ACTIONS:")
        for i, action in enumerate(roadmap['immediate_actions'][:3], 1):
            print(f"      {i}. {action}")
    
    if roadmap and roadmap.get('priority_skills'):
        print(f"\n   🎯 PRIORITY SKILLS TO LEARN:")
        for i, skill in enumerate(roadmap['priority_skills'][:5], 1):
            print(f"      {i}. {skill}")
    
    # Tech Stack Recommendations
    if recommendations:
        print(f"\n   🔧 TECH STACK RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"      {i}. {rec}")
    
    # Final LLM Verdict
    print(f"\n📋 FINAL RECOMMENDATION:")
    final_verdict = llm_analysis.get('final_verdict', 'Enhanced analysis completed successfully')
    if len(final_verdict) > 200:
        final_verdict = final_verdict[:200] + "..."
    print(f"   {final_verdict}")
    
    # LangSmith Session Summary (if available)
    if ADVANCED_PIPELINE:
        print(f"\n🔍 LANGSMITH OBSERVABILITY:")
        try:
            session_summary = logger.get_session_summary()
            print(f"   📊 Total Traces: {session_summary.get('total_traces', 0)}")
            print(f"   📈 Total Metrics: {session_summary.get('total_metrics', 0)}")
            print(f"   📁 Session ID: {session_summary.get('session_id', 'N/A')[:8]}...")
        except:
            print(f"   📊 Session data available in logs/ directory")
    
    print(f"\n{'='*75}")

def display_structured_results(basic_scores, llm_analysis, roadmap, enhanced_skills):
    """Fallback display for basic scoring (original function)"""
    
    print(f"\n{'='*70}")
    print("🎯 Automated Resume Relevance Check Report")
    if ADVANCED_PIPELINE:
        print("   🔗 LangGraph + LangSmith Integration Active")
    print("=" * 70)
    
    # RELEVANCE ANALYSIS - 3 Steps
    print(f"\n📋 RELEVANCE ANALYSIS")
    print("-" * 50)
    
    # Step 1: Hard Match
    print(f"\n🔍 STEP 1: HARD MATCH (Keyword & Skill Check)")
    print(f"   • Exact Matches: {basic_scores['matched_count']}/{basic_scores['total_jd_skills']} skills")
    print(f"   • Coverage Score: {basic_scores['score']:.1f}%")
    print(f"   • Matched Skills: {', '.join(basic_scores['matched_skills'][:8])}")
    if len(basic_scores['matched_skills']) > 8:
        print(f"     ... and {len(basic_scores['matched_skills']) - 8} more")
    
    # Step 2: Semantic Match  
    experience_fit = llm_analysis.get('overall_fit_score', 0)
    print(f"\n🧠 STEP 2: SEMANTIC MATCH (LLM Analysis)")
    print(f"   • Experience Alignment Score: {experience_fit}/10")
    print(f"   • Context Understanding: {llm_analysis.get('experience_alignment', 'N/A')[:100]}...")
    
    # Step 3: Scoring & Verdict
    hard_match_score = basic_scores['score']
    semantic_score = experience_fit * 10  # Convert to percentage
    final_score = (hard_match_score * 0.4) + (semantic_score * 0.6)  # Weighted formula
    
    print(f"\n⚖️ STEP 3: SCORING & VERDICT (Weighted Formula)")
    print(f"   • Formula: (Hard Match × 40%) + (Semantic Match × 60%)")
    print(f"   • Calculation: ({hard_match_score:.1f}% × 0.4) + ({semantic_score:.1f}% × 0.6)")
    print(f"   • Final Score: {final_score:.1f}/100")
    
    # OUTPUT GENERATION
    print(f"\n📊 OUTPUT GENERATION")
    print("-" * 50)
    
    # Relevance Score
    print(f"\n🎯 RELEVANCE SCORE: {final_score:.0f}/100")
    
    # Verdict
    if final_score >= 80:
        verdict = "🟢 HIGH SUITABILITY"
        verdict_desc = "Strong candidate - Recommend for interview"
    elif final_score >= 60:
        verdict = "🟡 MEDIUM SUITABILITY" 
        verdict_desc = "Good potential - Consider with training"
    else:
        verdict = "🔴 LOW SUITABILITY"
        verdict_desc = "Significant gaps - Major upskilling needed"
    
    print(f"\n🏷️ VERDICT: {verdict}")
    print(f"   • Assessment: {verdict_desc}")
    
    # Missing Skills/Projects/Certifications
    print(f"\n❌ MISSING SKILLS/REQUIREMENTS:")
    missing_items = basic_scores['missing_skills'][:8]  # Top 8 missing
    for i, item in enumerate(missing_items, 1):
        print(f"   {i}. {item}")
    
    if llm_analysis.get('critical_gaps'):
        print(f"\n⚠️ CRITICAL GAPS IDENTIFIED:")
        for i, gap in enumerate(llm_analysis['critical_gaps'][:3], 1):
            print(f"   {i}. {gap}")
    
    # Suggestions for Student Improvement
    print(f"\n💡 SUGGESTIONS FOR STUDENT IMPROVEMENT:")
    
    # Immediate actions
    if roadmap and roadmap.get('immediate_actions'):
        print(f"\n   📋 IMMEDIATE ACTIONS:")
        for i, action in enumerate(roadmap['immediate_actions'][:3], 1):
            print(f"      {i}. {action}")
    
    # Skills to learn
    if roadmap and roadmap.get('priority_skills'):
        print(f"\n   🎯 PRIORITY SKILLS TO LEARN:")
        for i, skill in enumerate(roadmap['priority_skills'][:5], 1):
            print(f"      {i}. {skill}")
    
    # Quick wins
    if roadmap and roadmap.get('quick_wins'):
        print(f"\n   🚀 QUICK WINS:")
        for i, win in enumerate(roadmap['quick_wins'][:3], 1):
            print(f"      {i}. {win}")
    
    # Final recommendation
    print(f"\n📋 FINAL RECOMMENDATION:")
    final_verdict = llm_analysis.get('final_verdict', 'Analysis completed successfully')
    if len(final_verdict) > 200:
        final_verdict = final_verdict[:200] + "..."
    print(f"   {final_verdict}")
    
    print(f"\n{'='*70}")

@trace_llm_analysis if ADVANCED_PIPELINE else lambda x: x
def complete_ai_analysis_api(resume_file, jd_file):
    """API version with LangGraph + LangSmith integration"""
    start_time = time.time()
    
    trace_id = None
    if ADVANCED_PIPELINE:
        trace_id = logger.start_trace("api_resume_analysis", {
            "resume_file": resume_file,
            "jd_file": jd_file,
            "api_call": True
        })
    
    try:
        llm_analyzer = LLMResumeAnalyzer(model=LLM_MODEL)
        
        # Initialize LangGraph pipeline if available
        if ADVANCED_PIPELINE:
            pipeline = ResumeAnalysisPipeline(model=LLM_MODEL)
        
        # Load and process files
        resume_raw = load_file(resume_file)
        jd_raw = load_file(jd_file)
        
        resume_clean = clean_text(resume_raw)
        resume_sections = split_sections(resume_clean)
        resume_skills = extract_skills(" ".join(resume_sections.values()))
        
        jd_data = parse_jd(jd_raw)
        jd_skills = jd_data["skills"]
        
        # Enhanced scoring if available
        if ENHANCED_SCORING:
            enhanced_scorer = EnhancedResumeScorer()
            comprehensive_result = enhanced_scorer.calculate_comprehensive_score(
                {"raw_text": resume_clean, "skills": resume_skills},
                {"raw_text": jd_raw, "skills": jd_skills}
            )
            
            final_score = comprehensive_result["final_score"]
            basic_scores = {
                "score": comprehensive_result["breakdown"]["hard_match"]["score"],
                "matched_skills": comprehensive_result["breakdown"]["hard_match"]["matched_skills"],
                "missing_skills": comprehensive_result["breakdown"]["hard_match"]["missing_skills"],
                "matched_count": comprehensive_result["breakdown"]["hard_match"]["matched_count"],
                "total_jd_skills": comprehensive_result["breakdown"]["hard_match"]["total_jd_skills"]
            }
        else:
            basic_scores = calculate_basic_scores(resume_skills, jd_skills)
            hard_match_score = basic_scores['score']
            semantic_score = 50
            final_score = (hard_match_score * 0.4) + (semantic_score * 0.6)
        
        # Run LangGraph pipeline if available
        if ADVANCED_PIPELINE:
            pipeline_result = pipeline.run_structured_analysis(resume_clean, jd_raw, basic_scores)
            
            if pipeline_result.get("pipeline_status") == "completed":
                llm_analysis = pipeline_result["llm_analysis"]
                improvement_roadmap = pipeline_result["improvement_roadmap"]
                pipeline_used = True
            else:
                llm_analysis = llm_analyzer.analyze_resume_vs_jd(resume_clean, jd_raw, basic_scores)
                improvement_roadmap = llm_analyzer.generate_improvement_roadmap(llm_analysis)
                pipeline_used = False
        else:
            llm_analysis = llm_analyzer.analyze_resume_vs_jd(resume_clean, jd_raw, basic_scores)
            improvement_roadmap = llm_analyzer.generate_improvement_roadmap(llm_analysis)
            pipeline_used = False
        
        # Determine verdict
        if final_score >= 80:
            verdict = "High Suitability"
            verdict_description = "Strong candidate - Recommend for interview"
        elif final_score >= 60:
            verdict = "Medium Suitability"
            verdict_description = "Good potential - Consider with training"
        else:
            verdict = "Low Suitability" 
            verdict_description = "Significant gaps - Major upskilling needed"
        
        # Finalize processing time
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)

        result = {
            "success": True,
            "enhanced_analysis": ENHANCED_SCORING,
            "langgraph_pipeline": pipeline_used,
            "langsmith_logging": ADVANCED_PIPELINE,
            "relevance_analysis": {
                "step_1_hard_match": {
                    "exact_matches": f"{basic_scores.get('matched_count', 0)}/{basic_scores.get('total_jd_skills', 0)}",
                    "coverage_score": basic_scores['score'],
                    "matched_skills": basic_scores['matched_skills'],
                    "tfidf_included": ENHANCED_SCORING,
                    "fuzzy_matches": [] if not ENHANCED_SCORING else comprehensive_result["breakdown"]["fuzzy_match"]["fuzzy_matched_skills"]
                },
                "step_2_semantic_match": {
                    "experience_alignment_score": llm_analysis.get('overall_fit_score', 0),
                    "context_understanding": llm_analysis.get('experience_alignment', ''),
                    "embedding_analysis": "Enhanced embeddings" if ENHANCED_SCORING else "LLM-powered analysis"
                },
                "step_3_scoring_verdict": {
                    "final_score": round(final_score, 1),
                    "enhanced_components": ENHANCED_SCORING
                }
            },
            "output_generation": {
                "relevance_score": f"{final_score:.0f}/100",
                "verdict": verdict,
                "verdict_description": verdict_description,
                "missing_skills": basic_scores['missing_skills'],
                "critical_gaps": llm_analysis.get('critical_gaps', []),
                "improvement_suggestions": {
                    "immediate_actions": improvement_roadmap.get('immediate_actions', [])[:3],
                    "priority_skills": improvement_roadmap.get('priority_skills', [])[:5], 
                    "quick_wins": improvement_roadmap.get('quick_wins', [])[:3]
                },
                "final_recommendation": llm_analysis.get('final_verdict', ''),
                "tech_stack_used": {
                    "semantic_embeddings": ENHANCED_SCORING,
                    "fuzzy_matching": ENHANCED_SCORING, 
                    "spacy_nlp": ENHANCED_SCORING,
                    "tfidf_scoring": ENHANCED_SCORING,
                    "faiss_vector_store": ENHANCED_SCORING,
                    "langgraph_pipeline": pipeline_used,
                    "langsmith_logging": ADVANCED_PIPELINE
                }
            },
            "processing_info": {
                "processing_time": processing_time
            }
        }
        
        # Log success
        if ADVANCED_PIPELINE and trace_id:
            logger.end_trace(trace_id, {
                "final_score": final_score,
                "pipeline_used": pipeline_used
            }, "success")
            
            logger.log_metrics({
                "api_success": True,
                "final_score": final_score,
                "pipeline_used": pipeline_used
            })
        
        return result
        
    except Exception as e:
        if ADVANCED_PIPELINE and trace_id:
            logger.end_trace(trace_id, {}, "error", str(e))
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # Check prerequisites
    print("🔧 Checking prerequisites...")
    
    # Check .env file
    if not os.path.exists('.env'):
        print("❌ .env file missing! Create it with your OPENROUTER_API_KEY")
        exit(1)
    
    # Check API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("❌ OPENROUTER_API_KEY not found in .env file!")
        print("💡 Add this to your .env file: OPENROUTER_API_KEY=your-key-here")
        exit(1)
    
    # Check files exist
    resume_file = "input/sample_resume.pdf"
    jd_file = "input/sample_jd.pdf"
    
    if not os.path.exists(resume_file):
        print(f"❌ Resume file not found: {resume_file}")
        exit(1)
        
    if not os.path.exists(jd_file):
        print(f"❌ JD file not found: {jd_file}")
        exit(1)
    
    print("✅ All prerequisites checked!")
    
    # Show final tech stack status
    print(f"\n🔧 TECH STACK STATUS:")
    print(f"   • Enhanced Scoring: {'✅ Active' if ENHANCED_SCORING else '⚠️ Basic'}")
    print(f"   • LangGraph Pipeline: {'✅ Active' if ADVANCED_PIPELINE else '⚠️ Not installed'}")
    print(f"   • LangSmith Logging: {'✅ Active' if ADVANCED_PIPELINE else '⚠️ Not installed'}")
    
    # Run the complete enhanced analysis
    complete_ai_analysis(resume_file, jd_file)
