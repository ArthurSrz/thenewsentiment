import numpy as np
from typing import List, Dict, Tuple, Optional
import ollama
from tqdm import tqdm
import re

class MultiTopicDetector:
    """
    Topic detection that identifies ALL relevant topics per chunk.
    Each chunk can have multiple topics, and each topic gets sentiment analysis.
    """
    
    def __init__(self, 
                 embedding_model: str = "granite-embedding",
                 confidence_threshold: float = 0.5,
                 max_topics_per_chunk: int = 5):
        """
        Initialize the multi-topic detector.
        
        Args:
            embedding_model: Ollama embedding model name
            confidence_threshold: Minimum similarity score for topic detection
            max_topics_per_chunk: Maximum number of topics to detect per chunk
        """
        self.embedding_model = embedding_model
        self.confidence_threshold = confidence_threshold
        self.max_topics_per_chunk = max_topics_per_chunk
        
        # Define topic categories with French descriptions and keywords
        self.topic_definitions = {
            "prix": {
                "description": "Prix, tarifs, coût, cher, pas cher, abordable, économique, rapport qualité-prix, euro, argent",
                "keywords": ["prix", "cher", "coût", "tarif", "abordable", "économique", "gratuit", "payant", "euro", "€", "argent", "budget"],
                "phrases": ["trop cher", "pas cher", "bon prix", "prix élevé", "coûte cher", "rapport qualité prix"]
            },
            "caisse": {
                "description": "Caisse, paiement, ticket, facture, caissier, caissière, file d'attente, monnaie, carte",
                "keywords": ["caisse", "caissier", "caissière", "ticket", "facture", "paiement", "monnaie", "carte", "file", "attente"],
                "phrases": ["à la caisse", "faire la queue", "ticket de caisse", "payer"]
            },
            "accueil": {
                "description": "Accueil, service client, politesse, amabilité, sourire, bonjour, gentillesse, courtoisie",
                "keywords": ["accueil", "bonjour", "politesse", "amabilité", "sourire", "sympa", "gentil", "aimable", "courtois"],
                "phrases": ["bon accueil", "mal accueilli", "bien reçu", "service client"]
            },
            "proprete": {
                "description": "Propreté, nettoyage, hygiène, sale, propre, dégoûtant, impeccable, poussière, rangement",
                "keywords": ["propre", "sale", "propreté", "nettoyage", "hygiène", "dégoûtant", "impeccable", "poussière", "rangé"],
                "phrases": ["c'est propre", "pas propre", "bien rangé", "mal rangé"]
            },
            "choix": {
                "description": "Choix, sélection, variété, assortiment, rayon, achalandage, gamme, produits, référence",
                "keywords": ["choix", "sélection", "variété", "assortiment", "rayon", "achalandé", "gamme", "produits", "référence"],
                "phrases": ["bon choix", "peu de choix", "grand choix", "rayons vides", "bien achalandé"]
            },
            "colis": {
                "description": "Colis, livraison, point relais, récupérer, déposer, expédition, mondial relais, chronopost",
                "keywords": ["colis", "livraison", "relais", "récupérer", "déposer", "expédition", "mondial", "chronopost", "pickup"],
                "phrases": ["point relais", "récupérer un colis", "déposer un colis"]
            },
            "horaire": {
                "description": "Horaires, ouverture, fermeture, dimanche, heures, planning, fermé, ouvert, horaires",
                "keywords": ["horaire", "ouverture", "fermeture", "dimanche", "heures", "fermé", "ouvert", "planning", "midi"],
                "phrases": ["horaires d'ouverture", "fermé le dimanche", "ouvert tard", "ferme tôt"]
            },
            "fruits_legumes": {
                "description": "Fruits, légumes, produits frais, primeur, fraîcheur, pourri, mûr, pomme, salade, légume",
                "keywords": ["fruits", "légumes", "frais", "primeur", "fraîcheur", "pourri", "mûr", "pomme", "salade", "carotte"],
                "phrases": ["fruits et légumes", "produits frais", "légumes pourris", "fruits mûrs"]
            },
            "pain": {
                "description": "Pain, boulangerie, baguette, viennoiseries, croissant, pâtisserie, brioche, four",
                "keywords": ["pain", "boulangerie", "baguette", "viennoiseries", "croissant", "pâtisserie", "brioche", "four"],
                "phrases": ["bonne baguette", "pain frais", "viennoiseries", "coin boulangerie"]
            },
            "boucherie_poissonnerie": {
                "description": "Boucherie, poissonnerie, viande, poisson, charcuterie, fraîcheur, boucher, poissonnier",
                "keywords": ["boucherie", "poissonnerie", "viande", "poisson", "charcuterie", "boucher", "frais", "poissonnier"],
                "phrases": ["rayon boucherie", "viande fraîche", "poisson frais", "charcuterie"]
            },
            "gerant": {
                "description": "Gérant, patron, propriétaire, direction, responsable, chef, manager, dirigeant",
                "keywords": ["gérant", "patron", "propriétaire", "direction", "responsable", "chef", "manager", "dirigeant"],
                "phrases": ["le gérant", "patron du magasin", "propriétaire", "responsable"]
            },
            "personnel": {
                "description": "Personnel, équipe, employés, service, staff, vendeur, vendeuse, employé, équipe",
                "keywords": ["personnel", "équipe", "employés", "service", "staff", "vendeur", "vendeuse", "employé"],
                "phrases": ["le personnel", "les employés", "équipe sympa", "vendeur"]
            },
            "drive": {
                "description": "Drive, commande, retrait, voiture, livraison, click and collect, commande en ligne",
                "keywords": ["drive", "commande", "retrait", "voiture", "livraison", "click", "collect", "ligne"],
                "phrases": ["drive", "commande en ligne", "click and collect", "retrait drive"]
            },
            "magasin": {
                "description": "Magasin, commerce, épicerie, supérette, boutique, enseigne, local, établissement",
                "keywords": ["magasin", "commerce", "épicerie", "supérette", "boutique", "enseigne", "local", "établissement"],
                "phrases": ["le magasin", "ce commerce", "cette épicerie", "supérette"]
            },
            "essence": {
                "description": "Station essence, carburant, pompe, gazole, diesel, plein, fuel, station service",
                "keywords": ["essence", "carburant", "pompe", "gazole", "diesel", "plein", "station", "fuel"],
                "phrases": ["station essence", "faire le plein", "pompe à essence", "station service"]
            },
            "uber_eats": {
                "description": "Uber Eats, livraison, commande en ligne, application, delivery, livraison repas",
                "keywords": ["uber", "eats", "livraison", "application", "delivery", "commande", "ligne", "app"],
                "phrases": ["uber eats", "livraison de repas", "commande uber", "app de livraison"]
            },
            "froid": {
                "description": "Produits froids, réfrigérateur, congélateur, glaces, surgelés, température, frigo",
                "keywords": ["froid", "réfrigérateur", "congélateur", "glaces", "surgelés", "frigo", "température", "glacé"],
                "phrases": ["produits froids", "rayon surgelés", "congélateur", "glaces"]
            }
        }
        
        # Cache for topic embeddings
        self._topic_embeddings = None
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text using Ollama."""
        try:
            response = ollama.embed(
                model=self.embedding_model,
                input=text
            )
            return np.array(response["embeddings"][0])
        except Exception as e:
            print(f"Error getting embedding for: {text[:50]}... Error: {e}")
            return np.zeros(384)  # Fallback zero vector
    
    def _initialize_topic_embeddings(self):
        """Initialize embeddings for all topic definitions."""
        if self._topic_embeddings is not None:
            return
        
        print("🔄 Initializing topic embeddings for multi-topic detection...")
        self._topic_embeddings = {}
        
        for topic_name, topic_info in tqdm(self.topic_definitions.items(), desc="Creating topic embeddings"):
            # Combine description, keywords, and phrases for comprehensive topic representation
            topic_text = f"{topic_info['description']} {' '.join(topic_info['keywords'])} {' '.join(topic_info['phrases'])}"
            embedding = self._get_embedding(topic_text)
            self._topic_embeddings[topic_name] = embedding
        
        print("✅ Multi-topic embeddings initialized")
    
    def _extract_topic_context(self, chunk_text: str, topic: str) -> str:
        """
        Extract the part of the chunk text that's most relevant to the specific topic.
        This helps provide better context for sentiment analysis.
        
        Args:
            chunk_text: The full chunk text
            topic: The topic to extract context for
            
        Returns:
            Text snippet most relevant to the topic
        """
        topic_info = self.topic_definitions[topic]
        keywords = topic_info['keywords']
        phrases = topic_info['phrases']
        
        # Split chunk into sentences
        sentences = re.split(r'[.!?]+', chunk_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Find sentences that contain topic-related keywords or phrases
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for phrases first (more specific)
            for phrase in phrases:
                if phrase.lower() in sentence_lower:
                    relevant_sentences.append(sentence)
                    break
            else:
                # Check for keywords
                for keyword in keywords:
                    if keyword.lower() in sentence_lower:
                        relevant_sentences.append(sentence)
                        break
        
        # If we found relevant sentences, return them; otherwise return the full chunk
        if relevant_sentences:
            return '. '.join(relevant_sentences)
        else:
            return chunk_text
    
    def detect_all_topics_in_chunk(self, chunk_text: str) -> List[Dict[str, any]]:
        """
        Detect ALL relevant topics in a single text chunk.
        Returns all topics that meet the confidence threshold.
        
        Args:
            chunk_text: Text chunk to analyze
            
        Returns:
            List of detected topics with confidence scores and context
        """
        # Initialize topic embeddings if not done
        self._initialize_topic_embeddings()
        
        # Get embedding for the chunk
        chunk_embedding = self._get_embedding(chunk_text)
        
        # Calculate similarities with all topics
        topic_scores = []
        
        for topic_name, topic_embedding in self._topic_embeddings.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(chunk_embedding, topic_embedding)
            
            # Only include topics above confidence threshold
            if similarity >= self.confidence_threshold:
                # Extract topic-specific context from the chunk
                topic_context = self._extract_topic_context(chunk_text, topic_name)
                
                topic_scores.append({
                    'topic': topic_name,
                    'confidence': float(similarity),
                    'context': topic_context,  # Text relevant to this topic
                    'full_chunk': chunk_text   # Full chunk for reference
                })
        
        # Sort by confidence and limit to max topics
        topic_scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        return topic_scores[:self.max_topics_per_chunk]
    
    def detect_topics_in_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Detect ALL relevant topics in multiple chunks efficiently.
        Each chunk can have multiple topics detected.
        
        Args:
            chunks: List of chunk dictionaries from semantic chunker
            
        Returns:
            List of chunks enriched with multi-topic detection
        """
        print(f"🔍 Detecting all topics in {len(chunks)} chunks...")
        
        enriched_chunks = []
        
        for chunk in tqdm(chunks, desc="Multi-topic detection"):
            # Detect all topics for this chunk
            detected_topics = self.detect_all_topics_in_chunk(chunk['text'])
            
            # Add topic information to chunk
            enriched_chunk = chunk.copy()
            enriched_chunk['detected_topics'] = detected_topics
            enriched_chunk['topic_count'] = len(detected_topics)
            enriched_chunk['has_topics'] = len(detected_topics) > 0
            
            # Create lists for easier access
            enriched_chunk['topic_names'] = [t['topic'] for t in detected_topics]
            enriched_chunk['topic_confidences'] = [t['confidence'] for t in detected_topics]
            enriched_chunk['topic_contexts'] = [t['context'] for t in detected_topics]
            
            enriched_chunks.append(enriched_chunk)
        
        print("✅ Multi-topic detection completed")
        return enriched_chunks
    
    def get_topic_summary(self, chunks: List[dict]) -> Dict[str, Dict]:
        """
        Generate a summary of all detected topics across all chunks.
        
        Args:
            chunks: List of chunks with multi-topic detection results
            
        Returns:
            Summary dictionary with topic statistics
        """
        topic_summary = {}
        
        for chunk in chunks:
            for topic_info in chunk.get('detected_topics', []):
                topic_name = topic_info['topic']
                confidence = topic_info['confidence']
                
                if topic_name not in topic_summary:
                    topic_summary[topic_name] = {
                        'count': 0,
                        'total_confidence': 0.0,
                        'avg_confidence': 0.0,
                        'max_confidence': 0.0,
                        'chunks': [],
                        'contexts': []
                    }
                
                topic_summary[topic_name]['count'] += 1
                topic_summary[topic_name]['total_confidence'] += confidence
                topic_summary[topic_name]['max_confidence'] = max(
                    topic_summary[topic_name]['max_confidence'], confidence
                )
                topic_summary[topic_name]['chunks'].append(chunk['chunk_id'])
                topic_summary[topic_name]['contexts'].append(topic_info['context'])
        
        # Calculate average confidences
        for topic_name in topic_summary:
            topic_summary[topic_name]['avg_confidence'] = (
                topic_summary[topic_name]['total_confidence'] / 
                topic_summary[topic_name]['count']
            )
        
        return topic_summary
    
    def filter_chunks_by_topic(self, chunks: List[dict], topic_name: str) -> List[dict]:
        """
        Filter chunks that contain a specific topic.
        
        Args:
            chunks: List of chunks with multi-topic detection results
            topic_name: Name of the topic to filter by
            
        Returns:
            List of chunks containing the specified topic
        """
        return [
            chunk for chunk in chunks 
            if topic_name in chunk.get('topic_names', [])
        ]
    
    def get_chunks_without_topics(self, chunks: List[dict]) -> List[dict]:
        """
        Get chunks that don't have any detected topics.
        
        Args:
            chunks: List of chunks with multi-topic detection results
            
        Returns:
            List of chunks without detected topics
        """
        return [
            chunk for chunk in chunks 
            if not chunk.get('has_topics', False)
        ]

# Example usage and testing
if __name__ == "__main__":
    # Test the multi-topic detector
    detector = MultiTopicDetector(
        confidence_threshold=0.5,
        max_topics_per_chunk=5
    )
    
    # Test chunks with multiple topics
    test_chunks = [
        {
            'chunk_id': 0,
            'text': "Les prix sont vraiment élevés mais le personnel est très sympa et accueillant.",
            'char_count': 78
        },
        {
            'chunk_id': 1,
            'text': "Les fruits et légumes sont pourris et en plus les horaires ne sont pas respectés.",
            'char_count': 81
        },
        {
            'chunk_id': 2,
            'text': "Super magasin ! Le pain est excellent, la boulangerie fait du bon travail et le gérant est professionnel.",
            'char_count': 103
        },
        {
            'chunk_id': 3,
            'text': "La station essence est pratique et on peut récupérer nos colis au point relais facilement.",
            'char_count': 91
        }
    ]
    
    print("Testing multi-topic detector...")
    enriched_chunks = detector.detect_topics_in_chunks(test_chunks)
    
    for chunk in enriched_chunks:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"Text: {chunk['text']}")
        print(f"Detected topics ({chunk['topic_count']}):")
        
        for topic_info in chunk['detected_topics']:
            print(f"  - {topic_info['topic']}: {topic_info['confidence']:.3f}")
            print(f"    Context: {topic_info['context'][:80]}...")
    
    # Test topic summary
    print("\nTopic Summary:")
    summary = detector.get_topic_summary(enriched_chunks)
    for topic, stats in summary.items():
        print(f"{topic}: {stats['count']} occurrences, avg confidence: {stats['avg_confidence']:.3f}")