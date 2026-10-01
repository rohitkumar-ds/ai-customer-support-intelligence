"""LLM provider implementations.

The default is MockLLM so the project is runnable without paid APIs. Gemini is an optional
provider selected through environment/configuration. No API key is ever stored in source code.
"""
from __future__ import annotations
import os, time
from typing import List, Dict
from src.llm.base import LLMClient
from src.llm.prompts import build_rag_prompt

class MockLLMClient(LLMClient):
    """Deterministic offline generator used for tests and demos."""

    def generate(self, query: str, context: str, history=None) -> str:
        """Return a transparent context-grounded response without pretending to be an LLM."""
        if not context.strip():
            return "I couldn't find sufficient information in the available support knowledge. This case should be reviewed by a support agent."
        first = context.split("\n")[0].strip()
        return f"Based on the available support policy: {first}"

class GeminiLLMClient(LLMClient):
    """Thin adapter around Google's current GenAI SDK."""

    def __init__(self, model_name:str, temperature:float=0.2, max_output_tokens:int=512):
        self.model_name=model_name
        self.temperature=temperature 
        self.max_output_tokens=max_output_tokens
        key=os.getenv("GEMINI_API_KEY")
        if not key: raise RuntimeError("GEMINI_API_KEY is required for the Gemini provider.")
        try:
            from google import genai
            self.client=genai.Client(api_key=key)
        except ImportError as exc:
            raise RuntimeError("Install google-genai to use Gemini.") from exc

    def generate(self, query:str, context:str, history=None)->str:
        """Generate a response using the grounded prompt and configured model."""
        prompt=build_rag_prompt(query,context,history)
        response=self.client.models.generate_content(
            model=self.model_name, contents=prompt,
            config={"temperature":self.temperature,"max_output_tokens":self.max_output_tokens},
        )
        return response.text or ""

def build_llm_client(settings) -> LLMClient:
    """Create the configured provider, defaulting to offline MockLLM."""
    if settings.llm.provider.lower()=="gemini":
        return GeminiLLMClient(settings.llm.model_name,settings.llm.temperature,settings.llm.max_output_tokens)
    return MockLLMClient()
