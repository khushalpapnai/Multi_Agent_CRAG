# src/agents/gemini_client.py (Khushal)
import json
import streamlit as st
from google import genai
from google.genai import types
from typing import Dict, Any
from src.config import get_logger, LLM_MODEL

logger = get_logger(__name__)

class GeminiClient:
def __init__(self, api_key: str = None):
        """
        Initializes the unified Google GenAI SDK (2026 architecture).
        Fetches the API key from Streamlit secrets and configures automatic retries.
        """
        try:
            secure_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            logger.error("CRITICAL: GEMINI_API_KEY not found in Streamlit secrets!")
            secure_key = api_key 

        # Initialize the client with built-in HTTP retry options for 429 errors
        self.client = genai.Client(
            api_key=secure_key,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    initial_delay=2.0,  
                    attempts=5,         
                    jitter=1,           
                    http_status_codes=[429, 500, 503], 
                ),
                timeout=120 * 1000,    
            )
        )
        self.model_id = LLM_MODEL

    def evaluate_context(self, prompt: str) -> Dict[str, str]:
        """
        Triggers the Critic Agent. Enforces a strict JSON output schema to ensure
        programmatic parsability for state machine routing.
        """
        try:
            # Low temperature (0.1) enforces deterministic, analytical behavior (Khushal)
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response.text)
            return result
        except Exception as e:
            logger.error(f"Critic Agent execution failed: {str(e)}")
            # Fail-safe state fallback
            return {"status": "Ambiguous", "reasoning": "JSON parse failure during evaluation."}

    def synthesize_response(self, prompt: str) -> str:
        """
        Triggers the Synthesizer Agent. Returns unstructured narrative text.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3, # Medium temperature allows for fluid narrative prose
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Synthesizer Agent execution failed: {str(e)}")
            return "An internal system error occurred while synthesizing the final response."

    def stream_synthesize_response(self, prompt: str):
        """
        Yields a generator object for streaming the syntactic response sequentially
        to the Streamlit UI, mimicking real-time typewriter outputs.
        """
        try:
            response_stream = self.client.models.generate_content_stream(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.3)
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Generative streaming pipeline failed: {str(e)}")
            yield " Streaming connection abruptly terminated."
