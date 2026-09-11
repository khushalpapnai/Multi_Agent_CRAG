# src/agents/workflow.py (Khushal)
import json
from typing import List, Dict, Any, Generator
from src.vector_store import VectorStore
from src.agents.gemini_client import GeminiClient
from src.agents.state import AgentState
from src.agents.prompts import CRITIC_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT
from src.config import get_logger

logger = get_logger(__name__)

class AgenticWorkflow:
    def __init__(self, vector_store: VectorStore, gemini_client: GeminiClient):
        self.vector_store = vector_store
        self.llm = gemini_client

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Transforms the geometric vector dictionary payload into a structured string
        optimized for LLM multi-head attention processing. (Khushal)
        """
        formatted = ""
        for i, chunk in enumerate(chunks):
            metadata = chunk.get("metadata", {})
            formatted += f"\n--- Context Block {i+1} ---\n"
            formatted += f"Source Origin: {metadata.get('source', 'Unknown')} | Spatial Index (Page): {metadata.get('page', 'N/A')}\n"
            formatted += f"Extracted Data: {chunk.get('text', '')}\n"
        return formatted

    def execute(self, query: str) -> Generator[str, None, None]:
        """
        Orchestrates the Corrective RAG Multi-Agent workflow.
        Yields a string generator to support responsive UI paradigms. (Khushal)
        """
        logger.info(f"Initiating autonomous workflow for query matrix: {query}")

        # Sequence 1: Retriever Agent
        retrieved_chunks = self.vector_store.retrieve(query, top_k=5)
        if not retrieved_chunks:
            yield "The Retriever Agent failed to locate any semantic matches within the indexed vector space."
            return

        context_str = self._format_context(retrieved_chunks)

        # Sequence 2: Critic Agent (Evaluation)
        critic_prompt = CRITIC_SYSTEM_PROMPT.format(query=query, context=context_str)
        evaluation = self.llm.evaluate_context(critic_prompt)
        status = evaluation.get("status", "Ambiguous")
        logger.info(f"Critic Agent computed status: {status}")

        # Sequence 3: State Machine Routing (The Correction Phase)
        if status == "Incorrect":
            yield "Based on a rigorous evaluation of the uploaded documents, the Critic Agent determined that the necessary facts are absent. Halting generation to prevent AI hallucinations. "
            return

        # Sequence 4: Synthesizer Agent (Generation)
        synth_prompt = SYNTHESIZER_SYSTEM_PROMPT.format(query=query, context=context_str)

        yield f"*(System Telemetry: Critic Evaluation yielded **{status}**. Synthesizing response using validated data blocks...)*\n\n"

        for chunk in self.llm.stream_synthesize_response(synth_prompt):
            yield chunk