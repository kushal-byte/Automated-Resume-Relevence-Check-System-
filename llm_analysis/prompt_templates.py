# llm_analysis/prompt_templates.py

RESUME_ANALYSIS_PROMPT = """You are an expert HR recruiter analyzing resumes against job descriptions.

RESUME:
{resume_text}

JOB DESCRIPTION:
{jd_text}

KEYWORD MATCH ANALYSIS:
- Matched Skills ({matched_count}/{total_skills}): {matched_skills}
- Missing Skills: {missing_skills}
- Coverage: {coverage_percentage}%

Please provide a comprehensive analysis in JSON format:
{{
    "overall_fit_score": <0-10 integer>,
    "experience_alignment": "<brief assessment of experience match>",
    "key_strengths": ["<strength1>", "<strength2>", "<strength3>"],
    "critical_gaps": ["<gap1>", "<gap2>", "<gap3>"],
    "role_suitability": "<High/Medium/Low with reasoning>",
    "improvement_suggestions": ["<actionable suggestion1>", "<actionable suggestion2>"],
    "recommended_skills_to_learn": ["<skill1>", "<skill2>", "<skill3>"],
    "project_recommendations": ["<project idea1>", "<project idea2>"],
    "certification_suggestions": ["<cert1>", "<cert2>"],
    "interview_readiness": "<assessment of interview preparation needed>",
    "salary_expectations": "<realistic salary range assessment>",
    "final_verdict": "<detailed reasoning for recommendation>"
}}

Focus on being practical, specific, and actionable in your recommendations."""

IMPROVEMENT_ROADMAP_PROMPT = """Based on this resume analysis, create a detailed improvement roadmap for the candidate.

ANALYSIS RESULTS:
{analysis_results}

Create a structured improvement plan in JSON format:
{{
    "immediate_actions": ["<action that can be done today>", "<another immediate action>"],
    "week_1_plan": ["<specific task for week 1>", "<another week 1 task>"],
    "month_1_plan": ["<month 1 goal>", "<another month 1 goal>"],
    "month_3_plan": ["<3 month goal>", "<another 3 month goal>"],
    "priority_skills": ["<highest priority skill>", "<second priority>", "<third priority>"],
    "learning_resources": {{
        "free_courses": ["<course recommendation>", "<another course>"],
        "paid_courses": ["<premium course>", "<another premium course>"],
        "books": ["<book recommendation>", "<another book>"],
        "practice_platforms": ["<platform>", "<another platform>"]
    }},
    "portfolio_improvements": ["<specific project to build>", "<another project>"],
    "networking_suggestions": ["<networking advice>", "<another networking tip>"],
    "quick_wins": ["<easy improvement>", "<another quick win>"],
    "estimated_timeline": "<realistic timeline to become job-ready>"
}}

Be specific with course names, book titles, and platform recommendations."""

SKILLS_ENHANCEMENT_PROMPT = """Analyze the following text and extract ALL technical skills, then categorize and enhance the skills list.

TEXT TO ANALYZE:
{text}

Extract and categorize skills comprehensively in JSON format:
{{
    "programming_languages": ["<language1>", "<language2>"],
    "web_frameworks": ["<framework1>", "<framework2>"],
    "databases": ["<db1>", "<db2>"],
    "cloud_platforms": ["<platform1>", "<platform2>"],
    "devops_tools": ["<tool1>", "<tool2>"],
    "testing_tools": ["<tool1>", "<tool2>"],
    "development_tools": ["<tool1>", "<tool2>"],
    "soft_skills": ["<skill1>", "<skill2>"],
    "methodologies": ["<methodology1>", "<methodology2>"],
    "all_technical_skills": ["<comprehensive list of all technical skills found>"],
    "skill_proficiency_estimate": {{
        "<skill>": "<Beginner/Intermediate/Advanced based on context>",
        "<another_skill>": "<proficiency_level>"
    }}
}}

Be thorough and include variations (e.g., JS and JavaScript, k8s and Kubernetes)."""
