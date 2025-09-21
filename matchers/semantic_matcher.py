# matchers/semantic_matcher.py - ENHANCED SEMANTIC MATCHER
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SemanticMatcher:
    def __init__(self):
        try:
            # Using a lightweight, high-performance model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Semantic matcher initialized with SentenceTransformer model")
        except Exception as e:
            print(f"⚠️ Could not load SentenceTransformer model: {e}")
            print("   Install with: pip install sentence-transformers")
            self.model = None

    def calculate_semantic_similarity(self, text1: str, text2: str) -> dict:
        """Calculate semantic similarity using sentence embeddings"""
        if not self.model:
            return {
                "semantic_score": 0.0,
                "error": "SentenceTransformer model not loaded"
            }
        
        try:
            # Generate embeddings for both texts
            embedding1 = self.model.encode(text1, convert_to_tensor=True)
            embedding2 = self.model.encode(text2, convert_to_tensor=True)
            
            # Calculate cosine similarity
            cosine_score = util.pytorch_cos_sim(embedding1, embedding2)
            
            return {
                "semantic_score": round(float(cosine_score[0][0]) * 100, 2)
            }
        except Exception as e:
            print(f"❌ Error during semantic similarity calculation: {e}")
            return {"semantic_score": 0.0, "error": str(e)}
