"""Claude (Anthropic) provider."""

from src.ai.base import AIProvider
from src.core.exceptions import ConfigurationError


class AnthropicProvider(AIProvider):
    name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"

    def _generate(self, system: str, user: str, max_tokens: int) -> str:
        try:
            import anthropic
        except ImportError as exc:
            raise ConfigurationError(
                "Claude SDK is missing. Install anthropic from requirements.txt."
            ) from exc
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.truncated = getattr(response, "stop_reason", None) == "max_tokens"
        return "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
