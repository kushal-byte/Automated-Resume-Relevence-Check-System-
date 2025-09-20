# matchers/semantic_matcher.py - ENHANCED WITH EMBEDDINGS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import faiss

class SemanticMatcher:
    def __init__(self):
        print("🧠 Loading semantic model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = 384  # Model dimension
        
        # Initialize FAISS vector store
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.stored_texts = []
        
        print("✅ Semantic matcher initialized with FAISS vector store")
    
    def encode_text(self, text):
        """Generate embeddings for text"""
        return self.model.encode([text], normalize_embeddings=True)[0]
    
    def calculate_semantic_similarity(self, resume_text, jd_text):
        """Calculate semantic similarity using embeddings + cosine similarity"""
        print("🔍 Computing semantic similarity...")
        
        # Generate embeddings
        resume_embedding = self.encode_text(resume_text)
        jd_embedding = self.encode_text(jd_text)
        
        # Calculate cosine similarity
        similarity = cosine_similarity([resume_embedding], [jd_embedding])[0][0]
        
        # Convert to percentage
        semantic_score = max(0, similarity * 100)
        
        return {
            "semantic_score": round(semantic_score, 2),
            "raw_similarity": round(similarity, 4),
            "resume_embedding": resume_embedding,
            "jd_embedding": jd_embedding
        }
    
    def add_to_vector_store(self, text, metadata=None):
        """Add text to FAISS vector store for similarity search"""
        embedding = self.encode_text(text)
        
        # Add to FAISS index
        self.index.add(np.array([embedding]).astype('float32'))
        self.stored_texts.append({"text": text, "metadata": metadata})
        
        return len(self.stored_texts) - 1  # Return index
    
    def semantic_search(self, query_text, k=5):
        """Search for similar texts in vector store"""
        if self.index.ntotal == 0:
            return []
        
        query_embedding = self.encode_text(query_text)
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            min(k, self.index.ntotal)
        )
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.stored_texts):
                similarity = 1 - (distance / 2)  # Convert L2 distance to similarity
                results.append({
                    "text": self.stored_texts[idx]["text"],
                    "metadata": self.stored_texts[idx].get("metadata"),
                    "similarity": round(similarity, 4),
                    "rank": i + 1
                })
        
        return results

# Test function
def test_semantic_matcher():
    """Test semantic matching functionality"""
    matcher = SemanticMatcher()
    
    # Test similarity
    resume = "Python developer with React experience and AWS deployment skills"
    jd = "Looking for full stack developer with Python, React, and cloud experience"
    
    result = matcher.calculate_semantic_similarity(resume, jd)
    print(f"✅ Semantic similarity test: {result['semantic_score']:.1f}%")
    
    return result['semantic_score'] > 50

if __name__ == "__main__":
    test_semantic_matcher()
