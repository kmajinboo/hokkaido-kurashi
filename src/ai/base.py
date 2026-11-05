"""AIProvider interface shared by every text-generation backend."""

from abc import ABC, abstractmethod

from src.core.exceptions import APIConnectionError, ConfigurationError


class AIProvider(ABC):
    """Interface for AI services that turn a prompt into report text.

    Subclasses implement ``_generate`` (one call to the service). Settings
    validation, error wrapping and the empty-response check live here, so the
    report code treats every provider the same way.
    """

    #: Provider name used in ``AI_PROVIDER`` (lowercase).
    name: str = ""
    #: Environment variable that holds the API key, for error messages.
    api_key_env: str = ""

    def __init__(self, api_key: str | None, model: str | None) -> None:
        if not api_key:
            raise ConfigurationError(f"{self.api_key_env} is not configured in the project .env file.")
        if not model:
            raise ConfigurationError("AI_MODEL is not configured in the project .env file.")
        self.api_key = api_key
        self.model = model

    #: Note added when the answer stopped at ``max_tokens``.
    TRUNCATION_NOTE = (
        "（注意: 出力の長さの上限に達したため、このレポートは途中で終わっている可能性があります）"
    )

    def generate(self, system: str, user: str, max_tokens: int = 1600) -> str:
        """Return the model's text answer for ``system`` and ``user`` prompts.

        Providers set ``self.truncated`` to True when the service reports that
        the answer stopped at ``max_tokens``; a note is then added at the end.
        """
        self.truncated = False
        try:
            text = self._generate(system, user, max_tokens)
        except ConfigurationError:
            raise
        except Exception as exc:
            raise APIConnectionError(
                f"{self.name} report generation failed. Check the API key, model name, and connection."
            ) from exc
        text = (text or "").strip()
        if not text:
            raise APIConnectionError(f"{self.name} returned an empty report.")
        if self.truncated:
            text = f"{text}\n\n{self.TRUNCATION_NOTE}"
        return text

    @abstractmethod
    def _generate(self, system: str, user: str, max_tokens: int) -> str:
        """Call the service once and return the raw answer text."""
        raise NotImplementedError
