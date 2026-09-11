# src/agents/state.py (Khushal)
from typing import TypedDict, List, Dict, Any

class AgentState(TypedDict):
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    critic_evaluation: str
    filtered_chunks: List[Dict[str, Any]]
    final_answer: str
    status: str