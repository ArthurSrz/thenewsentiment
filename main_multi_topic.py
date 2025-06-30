"""
Enhanced semantic sentiment analysis pipeline for French customer reviews.
Provides sentiment analysis for EVERY TOPIC in EVERY CHUNK.
Each chunk can have multiple topics, and each topic gets individual sentiment analysis.
"""

import pandas as pd
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import warnings

# Import our updated modules
from semantic_chunker_simple import SimpleSemanticChunker
from topic_detector_multi import MultiTopicDetector
from sentiment_analyzer_multi_topic import MultiTopicSentimentAnalyzer
from config import load_config, PipelineConfig

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class MultiTopicSentimentPipeline:
    """
    Complete pipeline for multi-topic sentiment analysis of French customer reviews.
    Each chunk can have multiple topics, and each topic gets individual sentiment analysis.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the multi-topic sentiment pipeline.
        
        Args:
            config: Pipeline configuration. Uses default if None.
        """
        self.config = config or load_config()
        
        print("🎯 Initializing Multi-Topic Sentiment Analysis Pipeline")
        print(f"📊 Configuration: {self.config.processing.input_data_path}")
        
        # Initialize components
        self.chunker = None
        self.topic_detector = None
        self.sentiment_analyzer = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all pipeline components with configuration."""
        print("\n🔧 Initializing pipeline components...")
        
        # Initialize semantic chunker
        print("1️⃣ Initializing semantic chunker...")
        self.chunker = SimpleSemanticChunker(
            embedding_model=self.config.chunker.embedding_model,
            similarity_threshold=self.config.chunker.similarity_threshold,
            max_chunk_size=self.config.chunker.max_chunk_size,
            min_chunk_size=self.config.chunker.min_chunk_size,
            sentence_overlap=self.config.chunker.sentence_overlap
        )
        
        # Initialize multi-topic detector
        print("2️⃣ Initializing multi-topic detector...")
        self.topic_detector = MultiTopicDetector(
            embedding_model=self.config.topic_detector.embedding_model,
            confidence_threshold=self.config.topic_detector.confidence_threshold,
            max_topics_per_chunk=self.config.topic_detector.max_topics_per_chunk
        )
        
        # Initialize multi-topic sentiment analyzer
        print("3️⃣ Initializing multi-topic sentiment analyzer...")
        self.sentiment_analyzer = MultiTopicSentimentAnalyzer(
            model_name=self.config.sentiment_analyzer.model_name,
            device=self.config.sentiment_analyzer.device,
            batch_size=self.config.sentiment_analyzer.batch_size
        )
        
        print("✅ All components initialized successfully!")
    
    def process_single_review(self, review_text: str, review_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Process a single review through the complete multi-topic pipeline.
        
        Args:
            review_text: The review text to process
            review_id: Optional review ID for tracking
            
        Returns:
            Complete multi-topic analysis results for the review
        """
        if not review_text or not review_text.strip():
            return {
                'review_id': review_id,
                'error': 'Empty review text',
                'chunks': [],
                'topic_sentiments': {},
                'overall_sentiment': 'NEUTRAL'
            }
        
        start_time = time.time()
        
        try:
            # Step 1: Semantic chunking
            print(f"🔄 Step 1: Chunking review {review_id}...")
            chunks = self.chunker.chunk_text(review_text)
            
            if not chunks:
                return {
                    'review_id': review_id,
                    'error': 'No chunks created',
                    'chunks': [],
                    'topic_sentiments': {},
                    'overall_sentiment': 'NEUTRAL'
                }
            
            # Step 2: Multi-topic detection
            print(f"🔍 Step 2: Detecting ALL topics for {len(chunks)} chunks...")
            chunks_with_topics = self.topic_detector.detect_topics_in_chunks(chunks)
            
            # Step 3: Multi-topic sentiment analysis
            print(f"🎭 Step 3: Analyzing sentiment for each topic in each chunk...")
            chunks_with_sentiment = self.sentiment_analyzer.analyze_chunks_with_multiple_topics(chunks_with_topics)
            
            # Step 4: Aggregate results
            topic_sentiment_summary = self.sentiment_analyzer.get_sentiment_by_topic_summary(chunks_with_sentiment)
            overall_topic_sentiments = self.sentiment_analyzer.get_overall_sentiment_by_topic(chunks_with_sentiment)
            
            processing_time = time.time() - start_time
            
            # Calculate statistics
            all_topics = []
            all_topic_sentiments = []
            
            for chunk in chunks_with_sentiment:
                for ts in chunk.get('topic_sentiments', []):
                    all_topics.append(ts['topic'])
                    all_topic_sentiments.append(ts['sentiment'])
            
            unique_topics = list(set(all_topics))
            
            # Calculate overall review sentiment (majority vote across all topic-sentiments)
            if all_topic_sentiments:
                sentiment_counts = {s: all_topic_sentiments.count(s) for s in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']}
                overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            else:
                overall_sentiment = 'NEUTRAL'
                sentiment_counts = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 1}
            
            return {
                'review_id': review_id,
                'original_text': review_text,
                'processing_time': processing_time,
                'chunks': chunks_with_sentiment,
                'chunk_count': len(chunks_with_sentiment),
                'unique_topics_detected': unique_topics,
                'total_topic_instances': len(all_topics),
                'topic_sentiments': overall_topic_sentiments,
                'topic_sentiment_summary': topic_sentiment_summary,
                'overall_sentiment': overall_sentiment,
                'overall_sentiment_distribution': sentiment_counts,
                'chunks_with_topics': sum(1 for c in chunks_with_sentiment if c.get('has_topics')),
                'chunks_without_topics': sum(1 for c in chunks_with_sentiment if not c.get('has_topics')),
                'stats': {
                    'char_count': len(review_text),
                    'chunk_count': len(chunks_with_sentiment),
                    'unique_topics': len(unique_topics),
                    'total_topic_instances': len(all_topics),
                    'processing_time': processing_time
                }
            }
            
        except Exception as e:
            print(f"❌ Error processing review {review_id}: {e}")
            return {
                'review_id': review_id,
                'error': str(e),
                'chunks': [],
                'topic_sentiments': {},
                'overall_sentiment': 'NEUTRAL'
            }
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a DataFrame of reviews through the complete multi-topic pipeline.
        
        Args:
            df: DataFrame with review text column
            
        Returns:
            Enhanced DataFrame with multi-topic analysis results
        """
        review_column = self.config.processing.review_column
        
        if review_column not in df.columns:
            raise ValueError(f"Review column '{review_column}' not found in DataFrame")
        
        # Filter out empty reviews
        df_clean = df[df[review_column].notna() & (df[review_column].str.strip() != '')].copy()
        df_clean = df_clean.reset_index(drop=True)
        
        # Limit processing if configured
        if self.config.processing.max_reviews_to_process:
            df_clean = df_clean.head(self.config.processing.max_reviews_to_process)
        
        print(f"\n📋 Processing {len(df_clean)} reviews...")
        
        # Process reviews
        results = []
        failed_count = 0
        
        progress_bar = tqdm(df_clean.iterrows(), total=len(df_clean), 
                           desc="Processing reviews", 
                           disable=not self.config.processing.enable_progress_bar)
        
        for idx, row in progress_bar:
            review_text = row[review_column]
            review_id = row.get('id', idx)
            
            result = self.process_single_review(review_text, review_id)
            
            if 'error' in result:
                failed_count += 1
            
            results.append(result)
            
            # Update progress bar with current stats
            if self.config.processing.enable_progress_bar:
                success_rate = ((idx + 1 - failed_count) / (idx + 1)) * 100
                progress_bar.set_postfix({
                    'Success': f'{success_rate:.1f}%',
                    'Failed': failed_count
                })
        
        print(f"✅ Processing completed! Success: {len(results) - failed_count}/{len(results)}")
        
        # Convert results to DataFrame
        return self._results_to_dataframe(results, df_clean)
    
    def _results_to_dataframe(self, results: List[Dict], original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert multi-topic analysis results to a structured DataFrame.
        
        Args:
            results: List of multi-topic analysis results
            original_df: Original DataFrame for reference
            
        Returns:
            DataFrame with multi-topic analysis results
        """
        # Create base DataFrame with original data
        output_df = original_df.copy()
        
        # Add analysis columns
        analysis_columns = {
            'chunk_count': [],
            'unique_topics_detected': [],
            'total_topic_instances': [],
            'chunks_with_topics': [],
            'chunks_without_topics': [],
            'overall_sentiment': [],
            'topic_sentiments': [],  # JSON string of topic -> sentiment mapping
            'processing_time': [],
            'char_count': [],
            'positive_topic_sentiments': [],
            'negative_topic_sentiments': [],
            'neutral_topic_sentiments': [],
            'has_error': [],
            'detailed_results': []  # JSON string with complete chunk-level results
        }
        
        for result in results:
            # Basic metrics
            analysis_columns['chunk_count'].append(result.get('chunk_count', 0))
            analysis_columns['unique_topics_detected'].append(', '.join(result.get('unique_topics_detected', [])))
            analysis_columns['total_topic_instances'].append(result.get('total_topic_instances', 0))
            analysis_columns['chunks_with_topics'].append(result.get('chunks_with_topics', 0))
            analysis_columns['chunks_without_topics'].append(result.get('chunks_without_topics', 0))
            analysis_columns['overall_sentiment'].append(result.get('overall_sentiment', 'NEUTRAL'))
            analysis_columns['processing_time'].append(result.get('processing_time', 0.0))
            analysis_columns['has_error'].append('error' in result)
            
            # Topic sentiments as JSON string
            topic_sentiments = result.get('topic_sentiments', {})
            analysis_columns['topic_sentiments'].append(json.dumps(topic_sentiments, ensure_ascii=False))
            
            # Stats
            stats = result.get('stats', {})
            analysis_columns['char_count'].append(stats.get('char_count', 0))
            
            # Count topic sentiments
            all_topic_sentiments = []
            for chunk in result.get('chunks', []):
                for ts in chunk.get('topic_sentiments', []):
                    all_topic_sentiments.append(ts['sentiment'])
            
            analysis_columns['positive_topic_sentiments'].append(all_topic_sentiments.count('POSITIVE'))
            analysis_columns['negative_topic_sentiments'].append(all_topic_sentiments.count('NEGATIVE'))
            analysis_columns['neutral_topic_sentiments'].append(all_topic_sentiments.count('NEUTRAL'))
            
            # Store detailed chunk results as JSON
            detailed_results = []
            for chunk in result.get('chunks', []):
                chunk_detail = {
                    'chunk_id': chunk.get('chunk_id'),
                    'text': chunk.get('text', ''),
                    'topic_count': chunk.get('topic_count', 0),
                    'overall_sentiment': chunk.get('overall_sentiment'),
                    'overall_confidence': chunk.get('overall_confidence'),
                    'topic_sentiments': chunk.get('topic_sentiments', [])
                }
                detailed_results.append(chunk_detail)
            
            analysis_columns['detailed_results'].append(json.dumps(detailed_results, ensure_ascii=False))
        
        # Add analysis columns to DataFrame
        for col_name, col_data in analysis_columns.items():
            output_df[col_name] = col_data
        
        return output_df
    
    def process_from_file(self, input_path: Optional[str] = None, 
                         output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Process reviews from an Excel file and save multi-topic results.
        
        Args:
            input_path: Path to input Excel file. Uses config if None.
            output_path: Path to output Excel file. Uses config if None.
            
        Returns:
            DataFrame with multi-topic analysis results
        """
        input_path = input_path or self.config.processing.input_data_path
        output_path = output_path or self.config.processing.output_data_path.replace('.xlsx', '_multi_topic.xlsx')
        
        print(f"📂 Loading data from: {input_path}")
        
        # Load data
        try:
            df = pd.read_excel(input_path)
            print(f"✅ Loaded {len(df)} rows from {input_path}")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
        
        # Process reviews
        result_df = self.process_dataframe(df)
        
        # Save results
        if output_path:
            print(f"💾 Saving results to: {output_path}")
            try:
                result_df.to_excel(output_path, index=False)
                print(f"✅ Results saved to {output_path}")
            except Exception as e:
                print(f"❌ Error saving results: {e}")
        
        return result_df
    
    def get_pipeline_stats(self, result_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate pipeline performance statistics for multi-topic analysis.
        
        Args:
            result_df: DataFrame with multi-topic analysis results
            
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            'total_reviews': len(result_df),
            'total_chunks': result_df['chunk_count'].sum(),
            'avg_chunks_per_review': result_df['chunk_count'].mean(),
            'total_processing_time': result_df['processing_time'].sum(),
            'avg_processing_time': result_df['processing_time'].mean(),
            'error_rate': (result_df['has_error'].sum() / len(result_df)) * 100,
            'chunks_with_topics': result_df['chunks_with_topics'].sum(),
            'chunks_without_topics': result_df['chunks_without_topics'].sum(),
            'topic_coverage': (result_df['chunks_with_topics'].sum() / result_df['chunk_count'].sum()) * 100,
            'total_topic_instances': result_df['total_topic_instances'].sum(),
            'avg_topic_instances_per_review': result_df['total_topic_instances'].mean(),
            'unique_topics_count': len(set([topic for topics in result_df['unique_topics_detected'] for topic in topics.split(', ') if topic])),
            'overall_sentiment_distribution': {
                'POSITIVE': (result_df['overall_sentiment'] == 'POSITIVE').sum(),
                'NEGATIVE': (result_df['overall_sentiment'] == 'NEGATIVE').sum(),
                'NEUTRAL': (result_df['overall_sentiment'] == 'NEUTRAL').sum()
            },
            'topic_sentiment_distribution': {
                'POSITIVE': result_df['positive_topic_sentiments'].sum(),
                'NEGATIVE': result_df['negative_topic_sentiments'].sum(),
                'NEUTRAL': result_df['neutral_topic_sentiments'].sum()
            }
        }
        
        return stats

def main():
    """Main execution function for multi-topic sentiment analysis."""
    print("🎯 Starting Multi-Topic Sentiment Analysis Pipeline")
    
    # Load configuration
    config = load_config(preset='fast')  # Use fast preset for testing
    
    # Initialize pipeline
    pipeline = MultiTopicSentimentPipeline(config)
    
    # Process reviews from file
    result_df = pipeline.process_from_file()
    
    # Generate and display statistics
    print("\n📊 Multi-Topic Pipeline Statistics:")
    stats = pipeline.get_pipeline_stats(result_df)
    
    print(f"Reviews processed: {stats['total_reviews']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    print(f"Chunks with topics: {stats['chunks_with_topics']} ({stats['topic_coverage']:.1f}%)")
    print(f"Total topic instances: {stats['total_topic_instances']}")
    print(f"Average topic instances per review: {stats['avg_topic_instances_per_review']:.1f}")
    print(f"Unique topics detected: {stats['unique_topics_count']}")
    print(f"Processing time: {stats['total_processing_time']:.1f}s")
    print(f"Error rate: {stats['error_rate']:.1f}%")
    
    print(f"\nOverall Review Sentiment Distribution:")
    for sentiment, count in stats['overall_sentiment_distribution'].items():
        percentage = (count / stats['total_reviews']) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    print(f"\nTopic-Level Sentiment Distribution:")
    for sentiment, count in stats['topic_sentiment_distribution'].items():
        percentage = (count / stats['total_topic_instances']) * 100 if stats['total_topic_instances'] > 0 else 0
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    print("\n🎉 Multi-topic analysis completed successfully!")
    print(f"📊 Each topic in each chunk now has individual sentiment analysis")
    
    return result_df

if __name__ == "__main__":
    main()