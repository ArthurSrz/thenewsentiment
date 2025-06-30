"""
Configuration management for the semantic sentiment analysis pipeline.
Centralizes all settings and parameters for easy tuning.
"""

import os
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict

@dataclass
class SemanticChunkerConfig:
    """Configuration for semantic chunking."""
    embedding_model: str = "granite-embedding"
    similarity_threshold: float = 0.7
    max_chunk_size: int = 512
    min_chunk_size: int = 50
    sentence_overlap: int = 1

@dataclass
class TopicDetectorConfig:
    """Configuration for topic detection."""
    embedding_model: str = "granite-embedding"
    confidence_threshold: float = 0.6
    max_topics_per_chunk: int = 3

@dataclass
class SentimentAnalyzerConfig:
    """Configuration for sentiment analysis."""
    model_name: str = "cmarkea/distilcamembert-base-sentiment"
    fallback_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    device: Optional[str] = None  # Auto-detect if None
    batch_size: int = 8

@dataclass
class ProcessingConfig:
    """Configuration for the main processing pipeline."""
    input_data_path: str = "fake_reviews_dataset.xlsx"
    output_data_path: str = "analyzed_reviews_results.xlsx"
    review_column: str = "text"
    enable_progress_bar: bool = True
    save_intermediate_results: bool = True
    max_reviews_to_process: Optional[int] = None  # None = process all

@dataclass
class PipelineConfig:
    """Main configuration class combining all component configs."""
    chunker: SemanticChunkerConfig
    topic_detector: TopicDetectorConfig
    sentiment_analyzer: SentimentAnalyzerConfig
    processing: ProcessingConfig

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary."""
        return cls(
            chunker=SemanticChunkerConfig(**config_dict.get('chunker', {})),
            topic_detector=TopicDetectorConfig(**config_dict.get('topic_detector', {})),
            sentiment_analyzer=SentimentAnalyzerConfig(**config_dict.get('sentiment_analyzer', {})),
            processing=ProcessingConfig(**config_dict.get('processing', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'chunker': asdict(self.chunker),
            'topic_detector': asdict(self.topic_detector),
            'sentiment_analyzer': asdict(self.sentiment_analyzer),
            'processing': asdict(self.processing)
        }
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(filepath):
            print(f"⚠️  Config file {filepath} not found. Using defaults.")
            return cls.get_default()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        print(f"✅ Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)
    
    @classmethod
    def get_default(cls) -> 'PipelineConfig':
        """Get default configuration."""
        return cls(
            chunker=SemanticChunkerConfig(),
            topic_detector=TopicDetectorConfig(),
            sentiment_analyzer=SentimentAnalyzerConfig(),
            processing=ProcessingConfig()
        )

# Performance presets
class PerformancePresets:
    """Predefined configuration presets for different use cases."""
    
    @staticmethod
    def fast_processing() -> PipelineConfig:
        """Fast processing with lower accuracy."""
        config = PipelineConfig.get_default()
        
        # Faster chunking
        config.chunker.similarity_threshold = 0.8  # Less sensitive
        config.chunker.max_chunk_size = 256  # Smaller chunks
        
        # Faster topic detection
        config.topic_detector.confidence_threshold = 0.7  # Higher threshold
        config.topic_detector.max_topics_per_chunk = 2  # Fewer topics
        
        # Faster sentiment analysis
        config.sentiment_analyzer.batch_size = 16  # Larger batches
        
        return config
    
    @staticmethod
    def accurate_processing() -> PipelineConfig:
        """Accurate processing with higher computational cost."""
        config = PipelineConfig.get_default()
        
        # More accurate chunking
        config.chunker.similarity_threshold = 0.6  # More sensitive
        config.chunker.max_chunk_size = 768  # Larger chunks for context
        config.chunker.sentence_overlap = 2  # More overlap
        
        # More accurate topic detection
        config.topic_detector.confidence_threshold = 0.5  # Lower threshold
        config.topic_detector.max_topics_per_chunk = 5  # More topics
        
        # More accurate sentiment analysis
        config.sentiment_analyzer.batch_size = 4  # Smaller batches for stability
        
        return config
    
    @staticmethod
    def cpu_optimized() -> PipelineConfig:
        """Optimized for CPU-only processing."""
        config = PipelineConfig.get_default()
        
        # CPU-friendly settings
        config.sentiment_analyzer.device = "cpu"
        config.sentiment_analyzer.batch_size = 4  # Smaller batches for CPU
        
        # Moderate accuracy settings
        config.chunker.max_chunk_size = 384
        config.topic_detector.max_topics_per_chunk = 2
        
        return config

# Environment-based configuration
def get_config_from_env() -> PipelineConfig:
    """Load configuration from environment variables."""
    config = PipelineConfig.get_default()
    
    # Override with environment variables if present
    if os.getenv('EMBEDDING_MODEL'):
        config.chunker.embedding_model = os.getenv('EMBEDDING_MODEL')
        config.topic_detector.embedding_model = os.getenv('EMBEDDING_MODEL')
    
    if os.getenv('SENTIMENT_MODEL'):
        config.sentiment_analyzer.model_name = os.getenv('SENTIMENT_MODEL')
    
    if os.getenv('SIMILARITY_THRESHOLD'):
        config.chunker.similarity_threshold = float(os.getenv('SIMILARITY_THRESHOLD'))
    
    if os.getenv('CONFIDENCE_THRESHOLD'):
        config.topic_detector.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))
    
    if os.getenv('BATCH_SIZE'):
        config.sentiment_analyzer.batch_size = int(os.getenv('BATCH_SIZE'))
    
    if os.getenv('INPUT_DATA_PATH'):
        config.processing.input_data_path = os.getenv('INPUT_DATA_PATH')
    
    if os.getenv('OUTPUT_DATA_PATH'):
        config.processing.output_data_path = os.getenv('OUTPUT_DATA_PATH')
    
    return config

# Configuration validation
def validate_config(config: PipelineConfig) -> bool:
    """Validate configuration parameters."""
    errors = []
    
    # Validate thresholds
    if not 0.0 <= config.chunker.similarity_threshold <= 1.0:
        errors.append("Similarity threshold must be between 0.0 and 1.0")
    
    if not 0.0 <= config.topic_detector.confidence_threshold <= 1.0:
        errors.append("Confidence threshold must be between 0.0 and 1.0")
    
    # Validate sizes
    if config.chunker.max_chunk_size <= config.chunker.min_chunk_size:
        errors.append("Max chunk size must be greater than min chunk size")
    
    if config.sentiment_analyzer.batch_size <= 0:
        errors.append("Batch size must be positive")
    
    # Validate file paths
    if not config.processing.input_data_path:
        errors.append("Input data path is required")
    
    if errors:
        print("❌ Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ Configuration validation passed")
    return True

# Main configuration loader
def load_config(config_path: Optional[str] = None, 
                preset: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration with priority: file > preset > environment > default.
    
    Args:
        config_path: Path to JSON config file
        preset: Preset name ('fast', 'accurate', 'cpu')
        
    Returns:
        Loaded configuration
    """
    # Start with default
    if preset:
        if preset == 'fast':
            config = PerformancePresets.fast_processing()
        elif preset == 'accurate':
            config = PerformancePresets.accurate_processing()
        elif preset == 'cpu':
            config = PerformancePresets.cpu_optimized()
        else:
            print(f"⚠️  Unknown preset '{preset}'. Using default.")
            config = PipelineConfig.get_default()
    else:
        config = PipelineConfig.get_default()
    
    # Override with environment variables
    env_config = get_config_from_env()
    if env_config != PipelineConfig.get_default():
        print("🌍 Applying environment variable overrides...")
        config = env_config
    
    # Override with file if provided
    if config_path:
        config = PipelineConfig.load_from_file(config_path)
    
    # Validate final configuration
    if not validate_config(config):
        print("⚠️  Using default configuration due to validation errors")
        config = PipelineConfig.get_default()
    
    return config

# Example usage
if __name__ == "__main__":
    # Create and save default config
    default_config = PipelineConfig.get_default()
    default_config.save_to_file("config.json")
    
    # Test loading
    loaded_config = PipelineConfig.load_from_file("config.json")
    
    # Test presets
    print("\nTesting presets:")
    fast_config = PerformancePresets.fast_processing()
    print(f"Fast config similarity threshold: {fast_config.chunker.similarity_threshold}")
    
    accurate_config = PerformancePresets.accurate_processing()
    print(f"Accurate config similarity threshold: {accurate_config.chunker.similarity_threshold}")
    
    print("\nConfiguration system ready! 🚀")