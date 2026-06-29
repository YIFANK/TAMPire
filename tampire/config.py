"""Central configuration. Loads .env and exposes Cerebras settings."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    api_key: str = os.environ.get("CEREBRAS_API_KEY", "")
    base_url: str = os.environ.get("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
    model: str = os.environ.get("TAMPIRE_MODEL", "gemma-4-31b")

    # Generation defaults. Low temperature for planning; the council bumps it for diversity.
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: float = 30.0

    # Loop control
    max_repair_rounds: int = 4
    council_size: int = 3  # agents in the debate council

    def require_key(self) -> None:
        if not self.api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY is not set. Add it to .env at the repo root."
            )


CONFIG = Config()
