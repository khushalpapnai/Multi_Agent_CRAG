# src/vector_store.py
import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any
from src.config import get_logger, EMBEDDING_MODEL

logger = get_logger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "document_kb"):
        """
        Bootstraps the ChromaDB instance. Utilizes EphemeralClient to respect
        serverless container read-only file system constraints. (Khushal)
        """
        logger.info("Bootstrapping Ephemeral ChromaDB Instance.")
        self.client = chromadb.EphemeralClient()

        # Instantiate the localized HuggingFace embedding transform (Khushal)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )

        # Enforce cosine similarity as the primary geometric distance metric (Khushal)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Calculates embeddings and upserts the textual chunks into the HNSW graph. (Khushal)
        """
        if not chunks:
            logger.warning("Vector ingestion bypassed: Empty chunk array provided.")
            return

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        # Generate deterministic, unique vector identifiers
        ids = [f"{m['source']}_p{m['page']}_c{m['chunk_id']}" for m in metadatas]

        try:
            self.collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully upserted {len(texts)} dense vectors into ChromaDB.")
        except Exception as e:
            logger.error(f"Vector Database upsert failure: {str(e)}")
            raise RuntimeError(f"Vector ingestion halted: {str(e)}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes a nearest-neighbor cosine similarity search against the HNSW graph. (Khushal)
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )

            retrieved_data = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    retrieved_data.append({
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0
                    })
            return retrieved_data

        except Exception as e:
            logger.error(f"Retrieval graph traversal failed for query '{query}': {str(e)}")
            return []