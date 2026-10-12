"""Settings read from the environment, so nothing is hard-coded in the modules that use them."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Settings:
    provider: str
    base_model: str
    adapter_dir: Path
    threads: int
    openai_model: str
    max_new_tokens: int = 48  # the longest valid record is about 35 tokens


def get_settings() -> Settings:
    adapter = Path(os.getenv("TRIAGE_ADAPTER_DIR", "adapter"))
    return Settings(
        provider=os.getenv("TRIAGE_PROVIDER", "local").lower(),
        base_model=os.getenv("TRIAGE_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct"),
        adapter_dir=adapter if adapter.is_absolute() else ROOT / adapter,
        threads=int(os.getenv("TRIAGE_THREADS", "0")),
        openai_model=os.getenv("OPENAI_MODEL", "") or "gpt-4o-mini",
    )
