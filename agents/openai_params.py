"""OpenAI output-limit parameter (audit item #36): `max_tokens` vs `max_completion_tokens`.

Newer official OpenAI models reject `max_tokens` with a 400 that names `max_completion_tokens`; compatible endpoints
(DeepSeek and others) only know `max_tokens`. No model table: send `max_tokens`, and if the 400 says to use
`max_completion_tokens`, retry once with it and remember the model for the rest of the process.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

import openai

# Process-wide memory of models that asked for max_completion_tokens (shared by main loop and side queries).
_COMPLETION_TOKENS_MODELS: set[str] = set()


def _asks_for_completion_tokens(error: Exception) -> bool:
    return isinstance(error, openai.BadRequestError) and "max_completion_tokens" in str(error)


async def create_with_output_limit(create: Callable[..., Awaitable[Any]], params: dict, limit: int | None) -> Any:
    """Call `create(**params)` with the output limit under the right parameter name; at most one switch-and-retry.

    The retry happens on the request itself, before any streamed chunk exists, so nothing has been printed yet.
    Any other 400 (and a failure of the retried request) is raised unchanged.
    """
    if limit is None:
        return await create(**params)
    model = params.get("model")
    name = "max_completion_tokens" if model in _COMPLETION_TOKENS_MODELS else "max_tokens"
    try:
        return await create(**{**params, name: limit})
    except Exception as error:
        if name != "max_tokens" or not _asks_for_completion_tokens(error):
            raise
    _COMPLETION_TOKENS_MODELS.add(model)
    return await create(**{**params, "max_completion_tokens": limit})
