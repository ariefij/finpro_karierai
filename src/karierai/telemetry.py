from __future__ import annotations

from typing import Any

from .config import get_settings


def get_langfuse_client() -> Any | None:
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse import Langfuse
    except Exception:
        return None
    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def get_callback_handler() -> Any | None:
    settings = get_settings()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        return None
    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def build_invoke_config(extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    callback = get_callback_handler()
    config: dict[str, Any] = {}
    if callback is not None:
        config['callbacks'] = [callback]
    if extra_metadata:
        config['metadata'] = extra_metadata
    return config
