"""Gemini (Google) provider."""

from src.ai.base import AIProvider
from src.core.exceptions import ConfigurationError


class GeminiProvider(AIProvider):
    name = "gemini"
    api_key_env = "GEMINI_API_KEY"

    def _generate(self, system: str, user: str, max_tokens: int) -> str:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ConfigurationError(
                "Gemini SDK is missing. Install google-genai from requirements.txt."
            ) from exc
        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system, max_output_tokens=max_tokens
            ),
        )
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        self.truncated = "MAX_TOKENS" in str(finish_reason)
        return response.text or ""
