"""Configuration loader supporting .env files and Streamlit Cloud secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _clean(val: str) -> str:
    """Strip whitespace/newlines and stray quotes that sneak in when pasting keys."""
    return val.strip().strip("\"'").strip()


def _get(key: str, default: str = "") -> str:
    val = os.getenv(key)
    if val is not None and val.strip() != "":
        return _clean(val)
    try:
        import streamlit as st
        if key in st.secrets:
            return _clean(str(st.secrets[key]))
    except Exception:
        pass
    return default


@dataclass
class Settings:
    provider: str
    groq_key: str
    groq_model: str          # "" = auto-detect a currently available model
    groq_effort: str
    gemini_key: str
    gemini_model: str        # "" = auto-detect a currently available model
    gmail_address: str
    gmail_app_password: str
    user_name: str
    tz: str
    auto_seed: bool
    data_dir: Path
    app_password: str        # "" = no login screen


def load_settings() -> Settings:
    # override=True so that editing .env and pressing "Reload settings" takes effect
    load_dotenv(override=True)
    root = Path(__file__).resolve().parent.parent
    data_dir = Path(_get("DATA_DIR", str(root / "data")))
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        provider=_get("LLM_PROVIDER", "groq").lower(),
        groq_key=_get("GROQ_API_KEY", ""),
        groq_model=_get("GROQ_MODEL", ""),
        groq_effort=_get("GROQ_REASONING_EFFORT", ""),
        gemini_key=_get("GEMINI_API_KEY", ""),
        gemini_model=_get("GEMINI_MODEL", ""),
        gmail_address=_get("GMAIL_ADDRESS", ""),
        gmail_app_password=_get("GMAIL_APP_PASSWORD", ""),
        user_name=_get("USER_NAME", "Student"),
        tz=_get("TIMEZONE", "Asia/Kolkata"),
        auto_seed=_get("AUTO_SEED", "true").lower() in ("1", "true", "yes"),
        data_dir=data_dir,
        app_password=_get("APP_PASSWORD", ""),
    )
