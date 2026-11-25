"""Shared application state: one database/retriever per server, one agent per browser session."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import streamlit as st

from assistant.agent import Agent
from assistant.config import Settings, load_settings
from assistant.db import Database
from assistant.llm import build_llm
from assistant.rag import Retriever
from assistant.seed import seed_if_empty
from assistant.tools import Toolbox


@dataclass
class Ctx:
    settings: Settings
    db: Database
    retriever: Retriever
    agent: Agent
    now: datetime


@st.cache_resource(show_spinner=False)
def _core():
    settings = load_settings()
    db = Database(settings.data_dir / "assistant.db")
    retriever = Retriever(db)
    if settings.auto_seed:
        seed_if_empty(db, retriever, settings.tz)
    return settings, db, retriever


def new_agent(settings, db, retriever) -> Agent:
    return Agent(build_llm(settings, db=db), Toolbox(db, retriever, settings), db, settings)


def boot() -> Ctx:
    settings, db, retriever = _core()
    if "agent" not in st.session_state:
        st.session_state.agent = new_agent(settings, db, retriever)
        st.session_state.chat = []
        st.session_state.pending = None
        st.session_state.partial = {}
    now = datetime.now(ZoneInfo(settings.tz)).replace(tzinfo=None)
    return Ctx(settings, db, retriever, st.session_state.agent, now)


def reload_settings() -> None:
    """Re-read .env / secrets and rebuild the AI connection (keeps the data)."""
    _core.clear()
    for k in ("agent", "chat", "pending", "partial"):
        st.session_state.pop(k, None)


def reset_conversation() -> None:
    for k in ("agent", "chat", "pending", "partial", "queued"):
        st.session_state.pop(k, None)


def technical() -> bool:
    return bool(st.session_state.get("_tech", False))
