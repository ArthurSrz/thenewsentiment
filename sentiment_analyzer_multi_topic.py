from typing import List, Dict, Union, Optional
import torch
from transformers import pipeline
from tqdm import tqdm
import numpy as np

class MultiTopicSentimentAnalyzer:
    """
    French sentiment analysis that provides sentiment scores for EACH detected topic per chunk.
    Analyzes sentiment for every topic found in each chunk, using topic-specific context.
    """
    
    def __init__(self, 
                 model_name: str = "cmarkea/distilcamembert-base-sentiment",
                 device: Optional[str] = None,
                 batch_size: int = 8):
        """
        Initialize the multi-topic sentiment analyzer.
        
        Args:
            model_name: Hugging Face model for French sentiment analysis
            device: Device to use ('cpu', 'cuda', 'mps'). Auto-detected if None
            batch_size: Batch size for processing multiple texts
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = 0  # GPU
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 0  # Apple Silicon MPS
            else:
                self.device = -1  # CPU
        else:
            self.device = 0 if device != 'cpu' else -1
        
        print(f"🔄 Loading French sentiment model: {model_name}")
        print(f"📱 Using device: {'GPU' if self.device >= 0 else 'CPU'}")
        
        # Initialize the sentiment pipeline
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=self.device,
                batch_size=self.batch_size,
                return_all_scores=True
            )
            print("✅ Sentiment model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Falling back to multilingual model...")
            
            # Fallback to multilingual model
            self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=self.device,
                batch_size=self.batch_size,
                return_all_scores=True
            )
            print("✅ Fallback model loaded successfully")
    
    def _normalize_scores(self, raw_scores: List[Dict]) -> Dict[str, float]:
        """Normalize sentiment scores to consistent format."""
        # Convert to dictionary for easier access
        score_dict = {item['label']: item['score'] for item in raw_scores}
        
        # Handle different label formats from different models
        positive_score = 0.0
        negative_score = 0.0
        
        # Check for different positive labels
        for pos_label in ['POSITIVE', 'LABEL_1', '5 stars', '4 stars']:
            if pos_label in score_dict:
                positive_score = max(positive_score, score_dict[pos_label])
        
        # Check for different negative labels  
        for neg_label in ['NEGATIVE', 'LABEL_0', '1 star', '2 stars']:
            if neg_label in score_dict:
                negative_score = max(negative_score, score_dict[neg_label])
        
        # If we have star ratings, aggregate them
        if '1 star' in score_dict:
            negative_score = score_dict.get('1 star', 0) + score_dict.get('2 stars', 0)
            positive_score = score_dict.get('4 stars', 0) + score_dict.get('5 stars', 0)
            neutral_score = score_dict.get('3 stars', 0)
        else:
            neutral_score = 0.0
        
        # Determine final sentiment and confidence
        if positive_score > negative_score and positive_score > neutral_score:
            sentiment = 'POSITIVE'
            confidence = positive_score
        elif negative_score > positive_score and negative_score > neutral_score:
            sentiment = 'NEGATIVE'
            confidence = negative_score
        else:
            sentiment = 'NEUTRAL'
            confidence = max(positive_score, negative_score, neutral_score)
        
        return {
            'sentiment': sentiment,
            'confidence': float(confidence),
            'positive_score': float(positive_score),
            'negative_score': float(negative_score),
            'neutral_score': float(neutral_score)
        }
    
    def analyze_topic_in_context(self, topic_context: str, topic: str, full_chunk: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment for a specific topic using its extracted context.
        
        Args:
            topic_context: Text specific to the topic (extracted from chunk)
            topic: The topic name
            full_chunk: The full chunk text for fallback
            
        Returns:
            Sentiment analysis results for this topic in this context
        """
        # Use topic context if meaningful, otherwise use full chunk
        text_to_analyze = topic_context if topic_context.strip() and len(topic_context.strip()) > 10 else full_chunk
        
        if not text_to_analyze or not text_to_analyze.strip():
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0,
                'topic': topic,
                'analyzed_text': '',
                'used_context': False
            }
        
        try:
            # Analyze sentiment of the text
            raw_result = self.classifier(text_to_analyze.strip())
            
            # Handle single result (when return_all_scores=True)
            if isinstance(raw_result[0], list):
                raw_scores = raw_result[0]
            else:
                raw_scores = raw_result
            
            # Normalize scores
            sentiment_result = self._normalize_scores(raw_scores)
            
            # Add topic and context information
            sentiment_result.update({
                'topic': topic,
                'analyzed_text': text_to_analyze,
                'used_context': topic_context != full_chunk
            })
            
            return sentiment_result
            
        except Exception as e:
            print(f"❌ Error analyzing topic {topic}: {e}")
            return {
                'sentiment': 'NEUTRAL',
                'confidence': 0.0,
                'positive_score': 0.0,
                'negative_score': 0.0,
                'neutral_score': 1.0,
                'topic': topic,
                'analyzed_text': text_to_analyze,
                'used_context': False
            }
    
    def analyze_chunk_with_multiple_topics(self, chunk_with_topics: dict) -> dict:
        """
        Analyze sentiment for a chunk that has multiple detected topics.
        Each topic gets its own sentiment analysis.
        
        Args:
            chunk_with_topics: Chunk with detected topics and contexts
            
        Returns:
            Chunk enriched with sentiment analysis for each topic
        """
        detected_topics = chunk_with_topics.get('detected_topics', [])
        
        if not detected_topics:
            # No topics detected, analyze the full chunk
            full_chunk_sentiment = self.analyze_topic_in_context(
                chunk_with_topics['text'], 
                None, 
                chunk_with_topics['text']
            )
            
            enriched_chunk = chunk_with_topics.copy()
            enriched_chunk['topic_sentiments'] = []
            enriched_chunk['overall_sentiment'] = full_chunk_sentiment['sentiment']
            enriched_chunk['overall_confidence'] = full_chunk_sentiment['confidence']
            enriched_chunk['sentiment_count'] = 0
            
            return enriched_chunk
        
        # Analyze sentiment for each detected topic
        topic_sentiments = []
        
        for topic_info in detected_topics:
            topic = topic_info['topic']
            topic_context = topic_info['context']
            full_chunk = topic_info['full_chunk']
            topic_confidence = topic_info['confidence']
            
            # Analyze sentiment for this topic
            sentiment_result = self.analyze_topic_in_context(topic_context, topic, full_chunk)
            
            # Create topic-sentiment entry
            topic_sentiment = {
                'topic': topic,
                'topic_confidence': topic_confidence,
                'sentiment': sentiment_result['sentiment'],
                'sentiment_confidence': sentiment_result['confidence'],
                'positive_score': sentiment_result['positive_score'],
                'negative_score': sentiment_result['negative_score'],
                'neutral_score': sentiment_result['neutral_score'],
                'analyzed_text': sentiment_result['analyzed_text'],
                'used_context': sentiment_result['used_context']
            }
            
            topic_sentiments.append(topic_sentiment)
        
        # Calculate overall chunk sentiment (weighted by topic and sentiment confidence)
        overall_sentiment = self._calculate_overall_sentiment(topic_sentiments)
        
        # Create enriched chunk
        enriched_chunk = chunk_with_topics.copy()
        enriched_chunk['topic_sentiments'] = topic_sentiments
        enriched_chunk['overall_sentiment'] = overall_sentiment['sentiment']
        enriched_chunk['overall_confidence'] = overall_sentiment['confidence']
        enriched_chunk['sentiment_count'] = len(topic_sentiments)
        
        return enriched_chunk
    
    def _calculate_overall_sentiment(self, topic_sentiments: List[Dict]) -> Dict[str, Union[str, float]]:
        """
        Calculate overall sentiment for a chunk based on individual topic sentiments.
        Uses weighted voting based on confidence scores.
        
        Args:
            topic_sentiments: List of topic-sentiment results
            
        Returns:
            Overall sentiment and confidence
        """
        if not topic_sentiments:
            return {'sentiment': 'NEUTRAL', 'confidence': 0.0}
        
        # Weight sentiments by both topic confidence and sentiment confidence
        weighted_positive = 0.0
        weighted_negative = 0.0
        weighted_neutral = 0.0
        total_weight = 0.0
        
        for ts in topic_sentiments:
            # Calculate weight as product of topic detection confidence and sentiment confidence
            weight = ts['topic_confidence'] * ts['sentiment_confidence']
            
            if ts['sentiment'] == 'POSITIVE':
                weighted_positive += weight
            elif ts['sentiment'] == 'NEGATIVE':
                weighted_negative += weight
            else:
                weighted_neutral += weight
            
            total_weight += weight
        
        if total_weight == 0:
            return {'sentiment': 'NEUTRAL', 'confidence': 0.0}
        
        # Normalize weights
        positive_ratio = weighted_positive / total_weight
        negative_ratio = weighted_negative / total_weight
        neutral_ratio = weighted_neutral / total_weight
        
        # Determine overall sentiment
        if positive_ratio > negative_ratio and positive_ratio > neutral_ratio:
            return {'sentiment': 'POSITIVE', 'confidence': positive_ratio}
        elif negative_ratio > positive_ratio and negative_ratio > neutral_ratio:
            return {'sentiment': 'NEGATIVE', 'confidence': negative_ratio}
        else:
            return {'sentiment': 'NEUTRAL', 'confidence': neutral_ratio}
    
    def analyze_chunks_with_multiple_topics(self, chunks_with_topics: List[dict]) -> List[dict]:
        """
        Analyze sentiment for chunks that have multiple detected topics.
        Each topic in each chunk gets its own sentiment analysis.
        
        Args:
            chunks_with_topics: List of chunks with multi-topic detection
            
        Returns:
            List of chunks enriched with multi-topic sentiment analysis
        """
        print(f"🎭 Analyzing sentiment for multiple topics in {len(chunks_with_topics)} chunks...")
        
        enriched_chunks = []
        
        for chunk in tqdm(chunks_with_topics, desc="Multi-topic sentiment analysis"):
            enriched_chunk = self.analyze_chunk_with_multiple_topics(chunk)
            enriched_chunks.append(enriched_chunk)
        
        print("✅ Multi-topic sentiment analysis completed")
        return enriched_chunks
    
    def get_sentiment_by_topic_summary(self, chunks: List[dict]) -> Dict[str, Dict]:
        """
        Generate comprehensive sentiment summary grouped by topic across all chunks.
        
        Args:
            chunks: List of chunks with multi-topic sentiment analysis
            
        Returns:
            Summary dictionary with sentiment statistics per topic
        """
        topic_sentiment_summary = {}
        
        for chunk in chunks:
            for topic_sentiment in chunk.get('topic_sentiments', []):
                topic = topic_sentiment['topic']
                sentiment = topic_sentiment['sentiment']
                sentiment_confidence = topic_sentiment['sentiment_confidence']
                topic_confidence = topic_sentiment['topic_confidence']
                
                if topic not in topic_sentiment_summary:
                    topic_sentiment_summary[topic] = {
                        'total_occurrences': 0,
                        'sentiment_counts': {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0},
                        'avg_sentiment_confidence': 0.0,
                        'avg_topic_confidence': 0.0,
                        'total_sentiment_confidence': 0.0,
                        'total_topic_confidence': 0.0,
                        'sentiment_distribution': {},
                        'examples': []
                    }
                
                stats = topic_sentiment_summary[topic]
                stats['total_occurrences'] += 1
                stats['sentiment_counts'][sentiment] += 1
                stats['total_sentiment_confidence'] += sentiment_confidence
                stats['total_topic_confidence'] += topic_confidence
                
                # Store example
                stats['examples'].append({
                    'chunk_id': chunk.get('chunk_id'),
                    'text': topic_sentiment['analyzed_text'][:100] + '...',
                    'sentiment': sentiment,
                    'confidence': sentiment_confidence
                })
        
        # Calculate averages and percentages
        for topic, stats in topic_sentiment_summary.items():
            if stats['total_occurrences'] > 0:
                stats['avg_sentiment_confidence'] = stats['total_sentiment_confidence'] / stats['total_occurrences']
                stats['avg_topic_confidence'] = stats['total_topic_confidence'] / stats['total_occurrences']
                
                # Calculate sentiment distribution percentages
                total = stats['total_occurrences']
                stats['sentiment_distribution'] = {
                    'positive_pct': (stats['sentiment_counts']['POSITIVE'] / total) * 100,
                    'negative_pct': (stats['sentiment_counts']['NEGATIVE'] / total) * 100,
                    'neutral_pct': (stats['sentiment_counts']['NEUTRAL'] / total) * 100
                }
        
        return topic_sentiment_summary
    
    def get_overall_sentiment_by_topic(self, chunks: List[dict]) -> Dict[str, str]:
        """
        Get the overall sentiment for each topic across all chunks.
        Uses weighted voting based on confidence scores.
        
        Args:
            chunks: List of chunks with multi-topic sentiment analysis
            
        Returns:
            Dictionary mapping topic to overall sentiment
        """
        topic_votes = {}
        
        for chunk in chunks:
            for topic_sentiment in chunk.get('topic_sentiments', []):
                topic = topic_sentiment['topic']
                sentiment = topic_sentiment['sentiment']
                sentiment_confidence = topic_sentiment['sentiment_confidence']
                topic_confidence = topic_sentiment['topic_confidence']
                
                # Weight vote by both confidences
                weight = sentiment_confidence * topic_confidence
                
                if topic not in topic_votes:
                    topic_votes[topic] = {'POSITIVE': 0.0, 'NEGATIVE': 0.0, 'NEUTRAL': 0.0}
                
                topic_votes[topic][sentiment] += weight
        
        # Determine overall sentiment per topic
        topic_sentiments = {}
        for topic, votes in topic_votes.items():
            max_sentiment = max(votes, key=votes.get)
            topic_sentiments[topic] = max_sentiment
        
        return topic_sentiments

# Example usage and testing
if __name__ == "__main__":
    # Test the multi-topic sentiment analyzer
    analyzer = MultiTopicSentimentAnalyzer()
    
    # Test chunk with multiple topics (simulating output from multi-topic detector)
    test_chunks = [
        {
            'chunk_id': 0,
            'text': "Les prix sont vraiment élevés mais le personnel est très sympa et accueillant.",
            'detected_topics': [
                {
                    'topic': 'prix',
                    'confidence': 0.85,
                    'context': 'Les prix sont vraiment élevés',
                    'full_chunk': "Les prix sont vraiment élevés mais le personnel est très sympa et accueillant."
                },
                {
                    'topic': 'personnel',
                    'confidence': 0.78,
                    'context': 'le personnel est très sympa et accueillant',
                    'full_chunk': "Les prix sont vraiment élevés mais le personnel est très sympa et accueillant."
                }
            ],
            'topic_count': 2,
            'has_topics': True
        },
        {
            'chunk_id': 1,
            'text': "Les fruits sont pourris et les horaires ne sont pas respectés.",
            'detected_topics': [
                {
                    'topic': 'fruits_legumes',
                    'confidence': 0.92,
                    'context': 'Les fruits sont pourris',
                    'full_chunk': "Les fruits sont pourris et les horaires ne sont pas respectés."
                },
                {
                    'topic': 'horaire',
                    'confidence': 0.76,
                    'context': 'les horaires ne sont pas respectés',
                    'full_chunk': "Les fruits sont pourris et les horaires ne sont pas respectés."
                }
            ],
            'topic_count': 2,
            'has_topics': True
        }
    ]
    
    print("Testing multi-topic sentiment analyzer...")
    enriched_chunks = analyzer.analyze_chunks_with_multiple_topics(test_chunks)
    
    for chunk in enriched_chunks:
        print(f"\nChunk {chunk['chunk_id']}:")
        print(f"Text: {chunk['text']}")
        print(f"Overall sentiment: {chunk['overall_sentiment']} (confidence: {chunk['overall_confidence']:.3f})")
        print(f"Topic sentiments ({chunk['sentiment_count']}):")
        
        for ts in chunk['topic_sentiments']:
            print(f"  - {ts['topic']}: {ts['sentiment']} (confidence: {ts['sentiment_confidence']:.3f})")
            print(f"    Analyzed: {ts['analyzed_text'][:60]}...")
    
    # Test topic sentiment summary
    print("\nSentiment by Topic Summary:")
    summary = analyzer.get_sentiment_by_topic_summary(enriched_chunks)
    for topic, stats in summary.items():
        print(f"\n{topic}:")
        print(f"  Total occurrences: {stats['total_occurrences']}")
        print(f"  Sentiment distribution: {stats['sentiment_distribution']}")
        print(f"  Average confidence: {stats['avg_sentiment_confidence']:.3f}")