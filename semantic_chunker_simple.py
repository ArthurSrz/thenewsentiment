import re
import numpy as np
from typing import List, Dict, Optional
import ollama
from tqdm import tqdm

class SimpleSemanticChunker:
    """
    Simplified semantic chunker without sklearn dependency.
    Uses basic cosine similarity calculation.
    """
    
    def __init__(self, 
                 embedding_model: str = "granite-embedding",
                 similarity_threshold: float = 0.7,
                 max_chunk_size: int = 512,
                 min_chunk_size: int = 50,
                 sentence_overlap: int = 1):
        """Initialize the simple semantic chunker."""
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.sentence_overlap = sentence_overlap
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Clean and normalize text
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Simple sentence splitting on French punctuation
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts using Ollama."""
        embeddings = []
        
        for text in tqdm(texts, desc="Generating embeddings"):
            try:
                response = ollama.embed(
                    model=self.embedding_model,
                    input=text
                )
                embeddings.append(response["embeddings"][0])
            except Exception as e:
                print(f"Error getting embedding: {e}")
                # Fallback: use zero vector
                embeddings.append([0.0] * 384)
        
        return np.array(embeddings)
    
    def _calculate_similarity_breaks(self, embeddings: np.ndarray) -> List[int]:
        """Calculate where to break chunks based on similarity."""
        if len(embeddings) <= 1:
            return []
        
        break_points = []
        
        for i in range(1, len(embeddings)):
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
            if similarity < self.similarity_threshold:
                break_points.append(i)
        
        return break_points
    
    def chunk_text(self, text: str) -> List[Dict]:
        """Main method to chunk text semantically."""
        if not text or not text.strip():
            return []
        
        print(f"🔄 Chunking text of {len(text)} characters...")
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [{
                'chunk_id': 0,
                'text': text.strip(),
                'sentences': sentences,
                'char_count': len(text.strip())
            }]
        
        # For simple testing, just split by sentence count if too many sentences
        if len(sentences) > 10:
            # Simple chunk splitting without embeddings for testing
            chunks = []
            sentences_per_chunk = 3
            
            for i in range(0, len(sentences), sentences_per_chunk):
                chunk_sentences = sentences[i:i + sentences_per_chunk]
                chunk_text = '. '.join(chunk_sentences)
                
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append({
                        'chunk_id': len(chunks),
                        'text': chunk_text,
                        'sentences': chunk_sentences,
                        'char_count': len(chunk_text)
                    })
            
            print(f"✅ Created {len(chunks)} simple chunks")
            return chunks
        
        # Get embeddings for sentences
        embeddings = self._get_embeddings(sentences)
        
        # Calculate semantic break points
        break_points = self._calculate_similarity_breaks(embeddings)
        
        # Create chunks
        chunks = []
        start_idx = 0
        all_breaks = sorted(set(break_points + [len(sentences)]))
        
        for break_idx in all_breaks:
            if break_idx <= start_idx:
                continue
                
            chunk_sentences = sentences[start_idx:break_idx]
            chunk_text = '. '.join(chunk_sentences)
            
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'chunk_id': len(chunks),
                    'text': chunk_text,
                    'sentences': chunk_sentences,
                    'char_count': len(chunk_text)
                })
            
            start_idx = max(0, break_idx - self.sentence_overlap)
        
        print(f"✅ Created {len(chunks)} semantic chunks")
        return chunks

if __name__ == "__main__":
    # Test the simple chunker
    chunker = SimpleSemanticChunker(max_chunk_size=300)
    
    test_text = """
    Bonjour, je vais souvent dans ce magasin de proximité et franchement les prix sont un peu élevés. 
    Le personnel est sympa surtout la caissière du matin. 
    Par contre les fruits et légumes ne sont pas toujours frais. 
    Sinon pour le pain c'est correct, la baguette est bonne le matin.
    """
    
    print("Testing simple semantic chunker...")
    chunks = chunker.chunk_text(test_text)
    
    for chunk in chunks:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"Text: {chunk['text']}")
        print(f"Length: {chunk['char_count']} chars")