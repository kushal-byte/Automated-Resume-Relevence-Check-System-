# AI-Powered Resume & Job Description Matcher

This project is an advanced AI-driven system for analyzing and matching resumes to job descriptions. It leverages modern NLP, semantic embeddings, fuzzy matching, and LLMs to provide comprehensive candidate-job fit analysis and actionable improvement suggestions.

## Features

- **PDF/DOCX Resume & JD Parsing:** Extracts and cleans text from resumes and job descriptions in PDF, DOCX, and TXT formats.
- **Skill Extraction:** Identifies technical and soft skills using dictionary matching, NLP, and LLM enhancement.
- **Section Splitting:** Splits resumes into logical sections (skills, experience, education, etc.).
- **Hard, Fuzzy, and Semantic Matching:** 
  - Hard match: Keyword overlap and TF-IDF similarity.
  - Fuzzy match: Handles skill variations (e.g., JS vs JavaScript).
  - Semantic match: Uses sentence embeddings and cosine similarity.
- **Entity Extraction:** Uses spaCy NLP to extract entities, experience years, and education info.
- **LLM Analysis:** Integrates with OpenRouter (OpenAI-compatible) LLMs for deep analysis, improvement roadmaps, and skill enhancement.
- **Comprehensive Scoring:** Weighted scoring formula combining all components for a final suitability verdict.
- **Actionable Recommendations:** Provides missing skills, critical gaps, and personalized improvement plans.

## Project Structure

```
.
├── main.py
├── config/
│   └── skills.yaml
├── input/
│   ├── sample_resume.pdf
│   └── sample_jd.pdf
├── llm_analysis/
│   ├── llm_analyzer.py
│   └── prompt_templates.py
├── matchers/
│   ├── entity_extractor.py
│   ├── final_scorer.py
│   ├── fuzzy_matcher.py
│   ├── hard_matcher.py
│   └── semantic_matcher.py
└── parsers/
    ├── cleaner.py
    ├── docx_parser.py
    ├── jd_parser.py
    ├── pdf_parser.py
    ├── section_splitter.py
    ├── skill_extractor.py
    └── skills_list.py
```

## Setup

1. **Clone the repository** and install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. **Configure API Keys:**
    - Create a `.env` file in the root directory:
      ```
      OPENROUTER_API_KEY=your-model-key
      OPENAI_MODEL=your-own-model 
      ```

3. **Prepare Input Files:**
    - Place your resume and job description files in the `input/` directory as `sample_resume.pdf` and `sample_jd.pdf`.

4. **Run the Analysis:**
    ```sh
    python main.py
    ```

## Requirements

- Python 3.8+
- [spaCy](https://spacy.io/) (`en_core_web_sm` model)
- [sentence-transformers](https://www.sbert.net/)
- [faiss-cpu](https://github.com/facebookresearch/faiss)
- [rapidfuzz](https://github.com/maxbachmann/RapidFuzz)
- [langchain](https://github.com/langchain-ai/langchain)
- [langchain-openai](https://github.com/langchain-ai/langchain)
- [python-dotenv](https://github.com/theskumar/python-dotenv)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [pdfplumber](https://github.com/jsvine/pdfplumber)
- [python-docx](https://github.com/python-openxml/python-docx)
- See `requirements.txt` for the full list.

## How It Works

1. **File Loading:** Loads and extracts text from resume and JD files.
2. **Preprocessing:** Cleans and splits resume into sections, extracts skills.
3. **Matching & Scoring:** Performs hard, fuzzy, and semantic matching between resume and JD.
4. **Entity Analysis:** Extracts entities, experience, and education using NLP.
5. **LLM Analysis:** Uses LLMs for deep fit analysis and improvement roadmap.
6. **Reporting:** Outputs a detailed, structured report with scores, verdict, missing skills, and recommendations.

## Customization

- **Skills Dictionary:** Edit `config/skills.yaml` or `parsers/skills_list.py` to add/remove skills.
- **Prompt Engineering:** Modify prompts in [`llm_analysis/prompt_templates.py`](llm_analysis/prompt_templates.py) for different LLM behaviors.
- **Scoring Logic:** Adjust weights and formulas in [`matchers/final_scorer.py`](matchers/final_scorer.py).

## License

MIT License

---

*Built with ❤️ by Innomatics Research Labs | KUSHAL MR & S MAYUR*
