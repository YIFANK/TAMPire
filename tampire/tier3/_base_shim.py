"""Vendored copy of RoboLab's InferenceClient ABC, behaviour-identical to
`robolab/eval/base_client.py` (Apache-2.0, (c) 2026 NVIDIA). Used ONLY when the
real `robolab` package is not importable (i.e. off the Isaac/RTX hardware), so the
TAMPire client can be imported and self-tested. On real hardware the genuine base
class is imported instead — see client.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class InferenceClient(ABC):
    """Root client for policy inference (vendored; see module docstring)."""

    open_loop_horizon: int = 1

    def __init__(self) -> None:
        self._chunks: dict[int, np.ndarray] = {}
        self._counters: dict[int, int] = {}

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        extracted = self._extract_observation(obs, env_id=env_id)
        if self._needs_refresh(env_id):
            request = self._pack_request(extracted, instruction)
            response = self._query_server(request)
            chunk = self._postprocess_chunk(self._unpack_response(response))
            self._set_chunk(env_id, chunk)
        action = self._next_action(env_id)
        return {"action": action, "viz": self._build_visualization(extracted)}

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._chunks.clear()
            self._counters.clear()
        else:
            self._chunks.pop(env_id, None)
            self._counters.pop(env_id, None)

    def close(self) -> None:
        return None

    def visualize(self, obs: Any, *, env_id: int = 0):
        return self._build_visualization(self._extract_observation(obs, env_id=env_id))

    @abstractmethod
    def _extract_observation(self, raw_obs: Any, *, env_id: int = 0) -> dict: ...

    @abstractmethod
    def _pack_request(self, extracted_obs: dict, instruction: str) -> Any: ...

    @abstractmethod
    def _query_server(self, request: Any) -> Any: ...

    @abstractmethod
    def _unpack_response(self, response: Any) -> np.ndarray: ...

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        return chunk

    def _build_visualization(self, extracted_obs: dict):
        return None

    def _needs_refresh(self, env_id: int) -> bool:
        return env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon

    def _set_chunk(self, env_id: int, chunk: np.ndarray) -> None:
        self._chunks[env_id] = chunk
        self._counters[env_id] = 0

    def _next_action(self, env_id: int) -> np.ndarray:
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return action
