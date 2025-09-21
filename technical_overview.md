# Technical Architecture

## Core Components
1. **Resume/JD Parser**: PyMuPDF, python-docx, spaCy
2. **Semantic Engine**: sentence-transformers, FAISS, cosine similarity
3. **Fuzzy Matcher**: RapidFuzz for skill variations
4. **LLM Integration**: OpenRouter + Grok for intelligent analysis
5. **Scoring Engine**: TF-IDF, weighted algorithms
6. **Web Interface**: FastAPI backend, Streamlit frontend

## Data Flow
1. File Upload → Text Extraction
2. NLP Processing → Entity Extraction  
3. Multi-Stage Analysis:
   - Hard Match (TF-IDF + Keywords)
   - Semantic Match (Embeddings + Cosine)
   - Fuzzy Match (Skill Variations)
   - LLM Analysis (Context Understanding)
4. Weighted Scoring → Final Verdict
5. Recommendations Generation → Export Report

## Scalability Features
- RESTful API design
- Async processing
- Vector database integration
- Modular architecture
- Cloud deployment ready
