"""Settings, read from environment variables (or a .env file next to the project)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_timeout_s: float
    llm_max_retries: int
    llm_merge_system_prompt: bool

    db_path: Path

    episode_token_budget: int
    max_revisions: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(PROJECT_ROOT / ".env")
        env = os.environ.get
        return cls(
            llm_base_url=env("LLM_BASE_URL", "http://localhost:8000/v1"),
            llm_api_key=env("LLM_API_KEY", "not-needed"),
            llm_model=env("LLM_MODEL", "google/gemma-4-12b-it"),
            llm_timeout_s=float(env("LLM_TIMEOUT_S", "180")),
            llm_max_retries=int(env("LLM_MAX_RETRIES", "2")),
            llm_merge_system_prompt=_bool(env("LLM_MERGE_SYSTEM_PROMPT", "false")),
            db_path=Path(env("DB_PATH", str(PROJECT_ROOT / "data" / "story.db"))),
            episode_token_budget=int(env("EPISODE_TOKEN_BUDGET", "150000")),
            max_revisions=int(env("MAX_REVISIONS", "2")),
        )
