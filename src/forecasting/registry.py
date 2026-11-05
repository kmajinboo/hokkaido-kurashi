"""Registry that maps model names to Forecaster classes."""

from src.core.exceptions import ConfigurationError
from src.forecasting.base import Forecaster
from src.forecasting.baseline import NaiveForecaster, SeasonalNaiveForecaster
from src.forecasting.statistical import (
    ARIMAForecaster,
    DriftForecaster,
    ETSForecaster,
    RecentDriftForecaster,
)

_REGISTRY: dict[str, type[Forecaster]] = {}


def register_forecaster(cls: type[Forecaster]) -> type[Forecaster]:
    """Register a Forecaster class under its ``name`` (usable as a decorator)."""
    if not (isinstance(cls, type) and issubclass(cls, Forecaster)) or not cls.name:
        raise ConfigurationError("Only Forecaster subclasses with a name can be registered")
    if cls.name in _REGISTRY and _REGISTRY[cls.name] is not cls:
        raise ConfigurationError(f"A forecasting model named {cls.name!r} is already registered")
    _REGISTRY[cls.name] = cls
    return cls


def available_models() -> list[str]:
    """Names of all registered forecasting models, in registration order."""
    return list(_REGISTRY)


def get_forecaster(name: str, **params) -> Forecaster:
    """Create a registered forecaster; ``params`` go to its constructor."""
    try:
        cls = _REGISTRY[name]
    except KeyError:
        names = ", ".join(_REGISTRY)
        raise ConfigurationError(f"Unknown forecasting model {name!r}. Available: {names}") from None
    return cls(**params)


register_forecaster(NaiveForecaster)
register_forecaster(SeasonalNaiveForecaster)
register_forecaster(DriftForecaster)
register_forecaster(RecentDriftForecaster)
register_forecaster(ETSForecaster)
register_forecaster(ARIMAForecaster)
