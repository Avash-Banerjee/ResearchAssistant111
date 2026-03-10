"""
ResearchIQ - Gemini LLM Client
================================
Unified LLM interface using Google Gemini API (google-genai SDK).
"""

import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from config.settings import GEMINI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)


class GeminiClient:
    """Singleton Gemini client with retry logic and error handling."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not set. Please add it to your .env file.\n"
                "Get your key at: https://aistudio.google.com/app/apikey"
            )

        try:
            # Try new SDK first (google-genai)
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=GEMINI_API_KEY)
            self._sdk = "new"
            self._types = types
            logger.info(f"GeminiClient initialized (google-genai SDK), model: {GEMINI_MODEL}")
        except ImportError:
            # Fallback to old SDK (google-generativeai)
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self._model = genai.GenerativeModel(GEMINI_MODEL)
            self._sdk = "old"
            logger.info(f"GeminiClient initialized (google-generativeai SDK), model: {GEMINI_MODEL}")

        self._initialized = True

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a response from Gemini."""
        try:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            if self._sdk == "new":
                from google.genai import types
                response = self._client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text
            else:
                response = self._model.generate_content(
                    full_prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    },
                )
                return response.text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def stream_generate(self, prompt: str, system_prompt: Optional[str] = None):
        """Stream response from Gemini (generator)."""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        if self._sdk == "new":
            from google.genai import types
            for chunk in self._client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=full_prompt,
            ):
                if chunk.text:
                    yield chunk.text
        else:
            response = self._model.generate_content(full_prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text


def get_llm() -> GeminiClient:
    return GeminiClient()
