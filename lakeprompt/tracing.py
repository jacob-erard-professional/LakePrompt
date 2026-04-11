from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


class PipelineLogger:
    """
    Lightweight stdout logger for tracing LakePrompt pipeline execution.
    """

    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def log(self, section: str, message: str, payload: Any | None = None) -> None:
        """
        Print a structured log block to stdout when enabled.
        """
        if not self.enabled:
            return

        print(f"[LakePrompt:{section}] {message}")
        if payload is not None:
            print(self._format_payload(payload))

    def _format_payload(self, payload: Any) -> str:
        """
        Render payloads into readable JSON-like text.
        """
        normalized = self._normalize(payload)
        try:
            return json.dumps(normalized, indent=2, ensure_ascii=True, sort_keys=True)
        except TypeError:
            return repr(normalized)

    def _normalize(self, value: Any) -> Any:
        """
        Convert arbitrary values into JSON-friendly structures.
        """
        if is_dataclass(value):
            return {key: self._normalize(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {str(key): self._normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._normalize(item) for item in value]
        return value


NULL_LOGGER = PipelineLogger(enabled=False)
