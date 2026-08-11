"""Shared ESRC pipeline primitives (generate, extract, search, reason, calibrate)."""

from esrc.generate import GenerationResult, generate, get_client

__all__ = ["GenerationResult", "generate", "get_client"]
