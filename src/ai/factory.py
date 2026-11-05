"""Create the AIProvider selected in the project .env file."""

from src.ai.anthropic_provider import AnthropicProvider
from src.ai.base import AIProvider
from src.ai.gemini_provider import GeminiProvider
from src.core import config
from src.core.exceptions import ConfigurationError

PROVIDERS: dict[str, type[AIProvider]] = {
    GeminiProvider.name: GeminiProvider,
    AnthropicProvider.name: AnthropicProvider,
}


def create_provider(
    provider: str | None = None, model: str | None = None, api_key: str | None = None
) -> AIProvider:
    """Build a provider from arguments, falling back to the .env settings.

    Settings are read when this function is called, not at import time, so a
    different provider can be chosen per call (for example in tests).
    """
    name = (provider if provider is not None else config.AI_PROVIDER).strip().lower()
    try:
        cls = PROVIDERS[name]
    except KeyError:
        choices = " or ".join(repr(key) for key in PROVIDERS)
        raise ConfigurationError(f"Set AI_PROVIDER to {choices} in .env.") from None
    return cls(
        api_key if api_key is not None else config.API_KEYS.get(name),
        model if model is not None else config.AI_MODEL,
    )
