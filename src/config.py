# src/config.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a standardized, production-grade logger.
    Ensures that agent routing decisions and vector operations are traceable. (Khushal)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Mathematical constraints for semantic text chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Model identifiers for embeddings and generative inference
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-3.6-flash"