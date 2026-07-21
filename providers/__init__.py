"""Local and remote model-provider adapters."""

from providers.ollama import OllamaClient, OllamaError

__all__ = ["OllamaClient", "OllamaError"]
