"""LLM access layer: Groq and Gemini through their OpenAI-compatible endpoints.

This module is deliberately defensive, because every failure it guards against
was observed in the wild:

* Model IDs get retired. Groq switched off ``llama-3.3-70b-versatile`` on
  2026-08-16 and Gemini 2.5 Flash is no longer served to new API keys. Instead
  of trusting one hard-coded name we ask each provider's ``/models`` endpoint
  what is available and walk a preference list. A model pinned in ``.env`` that
  has been retired simply falls through to the next candidate.
* Gemini 3.x models reject a tool-calling conversation unless the
  ``thought_signature`` they returned is sent back with the tool call.
* Error details used to be reduced to "HTTP 404". We keep the provider's own
  message so the Settings page can say exactly what is wrong.
"""
from __future__ import annotations

import hashlib
import io
import re
import time
import uuid
from dataclasses import dataclass, field

from openai import (APIConnectionError, APIError, APITimeoutError, AuthenticationError,
                    BadRequestError, InternalServerError, NotFoundError, OpenAI,
                    PermissionDeniedError, RateLimitError)

GROQ_URL = "https://api.groq.com/openai/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Ordered by preference. Only models that the provider actually lists are used.
GROQ_PREFERRED = ("openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b")
GEMINI_PREFERRED = ("gemini-flash-latest", "gemini-3.8-flash", "gemini-3.5-flash",
                    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite")
_GROQ_SKIP = ("whisper", "guard", "safeguard", "tts", "orpheus", "compound", "distil")
_GEMINI_OK = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?$")

MAX_OUT_TOKENS = 4096        # reasoning models spend part of this on hidden thinking
MODEL_CACHE_SECONDS = 3600
STT_MODELS = ("whisper-large-v3-turbo", "whisper-large-v3")
DUMMY_SIGNATURE = "skip_thought_signature_validator"   # Google's documented escape hatch

_MODEL_CACHE: dict[tuple[str, str], tuple[float, list[str]]] = {}


class LLMUnavailable(Exception):
    pass


class _ModelSkip(Exception):
    """This model cannot serve the request; try the next candidate."""


class _ProviderFatal(Exception):
    """Nothing else on this provider will work (bad key, no network)."""


@dataclass
class Provider:
    name: str
    client: OpenAI
    pinned: str = ""                      # model named in .env / secrets, may be retired
    effort: str = ""                      # reasoning effort override (Groq gpt-oss)
    model: str = ""                       # model that last worked
    candidates: list[str] = field(default_factory=list)
    dead: dict[str, str] = field(default_factory=dict)
    strip_extra: bool = False             # provider rejected our optional params
    dummy_signature: bool = False         # Gemini wants signatures we do not have


# ---------------------------------------------------------------- error handling
def _detail(e: Exception) -> str:
    """Pull the provider's own message out of an SDK exception."""
    body = getattr(e, "body", None)
    if isinstance(body, list) and body:
        body = body[0]
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"]).strip()
        if isinstance(err, str):
            return err
    return str(getattr(e, "message", None) or e).strip()


def _kind(e: Exception) -> str:
    text = _detail(e).lower() + " " + str(getattr(e, "code", "") or "").lower()
    if "api key" in text and any(w in text for w in ("not valid", "invalid", "expired", "missing")):
        return "auth"
    if isinstance(e, AuthenticationError):
        return "auth"
    if isinstance(e, PermissionDeniedError):
        return "model" if "model" in text else "auth"
    if isinstance(e, NotFoundError):
        return "model"
    if isinstance(e, RateLimitError):
        if "limit: 0" in text:
            return "model"           # no free-tier quota for this model at all
        if "per day" in text or "perday" in text or "daily" in text:
            return "quota"
        return "rate"
    if isinstance(e, BadRequestError):
        if any(w in text for w in ("decommission", "model_not_found", "does not exist",
                                   "no longer", "deprecated", "not supported for generatecontent")):
            return "model"
        if "tool_use_failed" in text or "failed_generation" in text:
            return "toolfail"
        if "thought_signature" in text or "thought signature" in text:
            return "signature"
        if any(w in text for w in ("reasoning", "unrecognized", "unknown parameter", "unsupported parameter")):
            return "param"
        return "bad"
    if isinstance(e, (APIConnectionError, APITimeoutError)):
        return "network"
    if isinstance(e, InternalServerError):
        return "server"
    return "other"


HINTS = {
    "auth": "The provider rejected the API key. Create a fresh key and paste it without quotes or spaces.",
    "model": "The configured model is retired or not available to this key. Remove the GROQ_MODEL / GEMINI_MODEL "
             "setting to let the app pick a working model automatically.",
    "quota": "The daily free quota is used up. It resets later, or another provider can take over.",
    "rate": "The provider is rate limiting requests. Wait a minute and try again.",
    "network": "The app could not reach the provider. Check the internet connection or firewall.",
    "server": "The provider is having a temporary problem. Try again shortly.",
}


# ---------------------------------------------------------------- helpers
def _normalize(msg) -> dict:
    calls = []
    for tc in (msg.tool_calls or []):
        call = {"id": tc.id or f"call_{uuid.uuid4().hex[:8]}",
                "name": tc.function.name,
                "arguments": tc.function.arguments or "{}"}
        extra = getattr(tc, "extra_content", None)
        if extra is None:
            extra = (getattr(tc, "model_extra", None) or {}).get("extra_content")
        if extra:                                  # Gemini 3.x thought signature
            call["extra_content"] = extra
        calls.append(call)
    return {"content": msg.content, "tool_calls": calls}


def _version_key(model_id: str) -> float:
    m = _GEMINI_OK.match(model_id)
    return float(m.group(1)) if m else 0.0


def pick_models(provider: str, listed: list[str], pinned: str = "") -> list[str]:
    """Order usable model IDs: pinned first, then preferred, then newest generic matches."""
    listed_set = set(listed)
    out: list[str] = [pinned] if pinned else []
    if provider == "groq":
        preferred = list(GROQ_PREFERRED)
        extras = sorted(m for m in listed if not any(s in m for s in _GROQ_SKIP) and m not in preferred)
    else:
        preferred = list(GEMINI_PREFERRED)
        extras = sorted((m for m in listed if _GEMINI_OK.match(m) and m not in preferred),
                        key=_version_key, reverse=True)
    out += [m for m in preferred if m in listed_set] + extras[:2]
    seen: set[str] = set()
    return [m for m in out if not (m in seen or seen.add(m))][:5]


class LLM:
    def __init__(self, providers: list[Provider], db=None, skipped: dict[str, str] | None = None):
        self.providers = providers
        self.db = db
        self.skipped = skipped or {}     # provider name -> why it is not configured

    # ------------------------------------------------------------ introspection
    @property
    def available(self) -> list[str]:
        return [p.name for p in self.providers]

    def describe(self) -> list[str]:
        return [f"{p.name} · {p.model or p.pinned or 'auto'}" for p in self.providers]

    # ------------------------------------------------------------ model discovery
    def _list_models(self, p: Provider) -> list[str]:
        return [str(getattr(m, "id", "")).removeprefix("models/") for m in p.client.models.list()]

    @staticmethod
    def _cache_key(p: Provider) -> tuple[str, str]:
        return p.name, hashlib.sha1(str(p.client.api_key).encode()).hexdigest()[:10]

    def _models(self, p: Provider) -> list[str]:
        if p.candidates and p.model:
            return [p.model] + [m for m in p.candidates if m != p.model]
        cached = _MODEL_CACHE.get(self._cache_key(p))
        if cached and time.time() - cached[0] < MODEL_CACHE_SECONDS:
            p.candidates = pick_models(p.name, cached[1], p.pinned)
        else:
            try:
                listed = self._list_models(p)
                _MODEL_CACHE[self._cache_key(p)] = (time.time(), listed)
                p.candidates = pick_models(p.name, listed, p.pinned)
            except Exception:
                # Listing failed (offline, bad key, ...). Fall back to what we know.
                preferred = GROQ_PREFERRED if p.name == "groq" else GEMINI_PREFERRED
                p.candidates = list(dict.fromkeys(([p.pinned] if p.pinned else []) + list(preferred)))[:4]
        return p.candidates or [p.pinned]

    # ------------------------------------------------------------ message shaping
    def _prep(self, p: Provider, messages: list[dict]) -> list[dict]:
        if p.name != "gemini":
            return messages
        names: dict[str, str] = {}
        out = []
        for raw in messages:
            m = dict(raw)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                calls = []
                for i, tc in enumerate(m["tool_calls"]):
                    tc = dict(tc)
                    names[tc["id"]] = tc["function"]["name"]
                    if p.dummy_signature and i == 0 and not tc.get("extra_content"):
                        tc["extra_content"] = {"google": {"thought_signature": DUMMY_SIGNATURE}}
                    calls.append(tc)
                m["tool_calls"] = calls
                if not m.get("content"):
                    m["content"] = None
            elif m.get("role") == "tool":
                m.setdefault("name", names.get(m.get("tool_call_id"), "tool"))
            elif m.get("role") == "assistant" and not m.get("content"):
                m["content"] = " "
            out.append(m)
        return out

    def _extra_body(self, p: Provider, model: str) -> dict:
        if p.strip_extra:
            return {}
        if p.name == "groq" and "gpt-oss" in model:
            return {"reasoning_effort": p.effort or "low"}
        if p.name == "gemini" and re.search(r"gemini-(?:\d|flash|pro)", model):
            return {"reasoning_effort": "low"}
        return {}

    # ------------------------------------------------------------ one provider
    def _try_model(self, p: Provider, model: str, messages: list[dict], tools):
        temperature = 0.2
        for attempt in range(3):
            t0 = time.perf_counter()
            kwargs = {"model": model, "messages": self._prep(p, messages), "temperature": temperature}
            kwargs["max_completion_tokens" if p.name == "groq" else "max_tokens"] = MAX_OUT_TOKENS
            extra = self._extra_body(p, model)
            if extra:
                kwargs["extra_body"] = extra
            if tools:
                kwargs["tools"], kwargs["tool_choice"] = tools, "auto"
            try:
                resp = p.client.chat.completions.create(**kwargs)
            except APIError as e:
                kind, detail = _kind(e), _detail(e)
                self._log_error(p, model, kind)
                if kind == "auth":
                    raise _ProviderFatal(f"API key rejected ({detail[:140]})")
                if kind == "model":
                    p.dead[model] = detail
                    raise _ModelSkip(f"{model} is not available ({detail[:140]})")
                if kind == "quota":
                    raise _ModelSkip(f"{model}: daily quota used up")
                if kind == "rate":
                    wait = 5.0
                    try:
                        wait = float(e.response.headers.get("retry-after", wait))
                    except Exception:
                        pass
                    if attempt == 2 or wait > 20:
                        raise _ModelSkip(f"{model}: rate limit reached")
                    time.sleep(min(wait, 20))
                elif kind in ("network", "server"):
                    if attempt == 2:
                        if kind == "network":
                            raise _ProviderFatal(f"connection problem: {detail[:140]}")
                        raise _ModelSkip(f"{model} server error: {detail[:140]}")
                    time.sleep(1.5 * (attempt + 1))
                elif kind == "toolfail" and attempt < 2:
                    temperature += 0.3                   # a different sample usually fixes a malformed call
                elif kind == "signature" and not p.dummy_signature:
                    p.dummy_signature = True
                elif kind == "param" and not p.strip_extra:
                    p.strip_extra = True
                else:
                    raise _ModelSkip(f"{model}: request rejected ({detail[:200]})")
                continue

            latency = (time.perf_counter() - t0) * 1000
            usage = getattr(resp, "usage", None)
            t_in = getattr(usage, "prompt_tokens", 0) or 0
            t_out = getattr(usage, "completion_tokens", 0) or 0
            if self.db:
                self.db.log_metric("llm", model, latency, provider=p.name, tokens_in=t_in, tokens_out=t_out)
            out = _normalize(resp.choices[0].message)
            out.update(provider=p.name, model=model, latency_ms=round(latency, 1), tokens=t_in + t_out)
            return out
        raise _ModelSkip(f"{model}: kept failing")

    def _chat_provider(self, p: Provider, messages: list[dict], tools):
        notes = []
        for model in self._models(p):
            if model in p.dead:
                notes.append(f"{model} was retired")
                continue
            try:
                out = self._try_model(p, model, messages, tools)
                p.model = model
                return out
            except _ModelSkip as e:
                notes.append(str(e))
        raise _ProviderFatal("; ".join(notes) or "no usable model found")

    def _log_error(self, p: Provider, model: str, kind: str) -> None:
        if self.db:
            try:
                self.db.log_metric("llm_error", model, 0, provider=p.name, status=kind)
            except Exception:
                pass

    # ------------------------------------------------------------ public API
    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        errors = []
        for p in self.providers:
            try:
                return self._chat_provider(p, messages, tools)
            except _ProviderFatal as e:
                errors.append(f"{p.name}: {e}")
        raise LLMUnavailable("; ".join(errors) or "no LLM provider configured")

    def transcribe(self, audio_bytes: bytes, filename: str = "voice.wav") -> str:
        """Transcribe voice input with Groq's Whisper endpoint."""
        groq_p = next((p for p in self.providers if p.name == "groq"), None)
        if not groq_p:
            raise LLMUnavailable("Voice input needs a Groq API key.")
        last: Exception | None = None
        for model in STT_MODELS:
            t0 = time.perf_counter()
            buf = io.BytesIO(audio_bytes)
            buf.name = filename
            try:
                res = groq_p.client.audio.transcriptions.create(model=model, file=buf, response_format="text")
            except APIError as e:
                last = e
                if _kind(e) in ("model", "bad"):
                    continue
                raise LLMUnavailable(_detail(e)[:200])
            if self.db:
                self.db.log_metric("stt", model, (time.perf_counter() - t0) * 1000, provider="groq")
            return str(res).strip()
        raise LLMUnavailable(_detail(last)[:200] if last else "no speech model available")

    # ------------------------------------------------------------ diagnostics
    def diagnose(self) -> list[dict]:
        """Run real checks against each configured provider and explain any failure."""
        ping = [{"type": "function", "function": {
            "name": "ping", "description": "Echo a short message back.",
            "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}}]
        report = []
        for name, why in self.skipped.items():
            report.append({"provider": name, "model": "", "checks": [], "hint": why, "configured": False})

        for p in self.providers:
            checks: list[dict] = []
            row = {"provider": p.name, "model": "", "checks": checks, "hint": "", "configured": True}
            report.append(row)

            def record(label, fn, row=row, checks=checks):
                t0 = time.perf_counter()
                try:
                    detail = fn()
                    checks.append({"check": label, "ok": True, "detail": detail or "ok",
                                   "ms": round((time.perf_counter() - t0) * 1000)})
                    return True
                except (_ProviderFatal, _ModelSkip, APIError, LLMUnavailable) as e:
                    msg = _detail(e) if isinstance(e, APIError) else str(e)
                    checks.append({"check": label, "ok": False, "detail": msg[:260],
                                   "ms": round((time.perf_counter() - t0) * 1000)})
                    if isinstance(e, APIError):
                        row["hint"] = HINTS.get(_kind(e), "")
                    elif "API key rejected" in msg:
                        row["hint"] = HINTS["auth"]
                    elif "connection" in msg:
                        row["hint"] = HINTS["network"]
                    elif "not available" in msg or "retired" in msg:
                        row["hint"] = HINTS["model"]
                    return False

            def list_models(p=p):
                ids = self._list_models(p)
                _MODEL_CACHE[self._cache_key(p)] = (time.time(), ids)
                p.candidates, p.model = [], ""
                chosen = pick_models(p.name, ids, p.pinned)
                return f"{len(ids)} models visible; will use: {', '.join(chosen[:3]) or 'none found'}"

            if not record("API key accepted", list_models):
                continue

            def basic(p=p, row=row):
                out = self._chat_provider(p, [{"role": "user", "content": "Reply with the single word: ready"}], None)
                row["model"] = out["model"]
                return f"{out['model']} answered in {out['latency_ms']:.0f} ms"

            if not record("Chat completion", basic):
                continue

            def tool_loop(p=p):
                msgs = [{"role": "user", "content": "Call the ping tool with the message 'hello', then tell me what it returned."}]
                first = self._chat_provider(p, msgs, ping)
                if not first["tool_calls"]:
                    raise _ModelSkip(f"{first['model']} did not call the tool")
                tc = first["tool_calls"][0]
                call = {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                if tc.get("extra_content"):
                    call["extra_content"] = tc["extra_content"]
                msgs += [{"role": "assistant", "content": first["content"] or "", "tool_calls": [call]},
                         {"role": "tool", "tool_call_id": tc["id"], "content": "pong: hello"}]
                second = self._chat_provider(p, msgs, ping)
                return f"tool call + follow-up worked ({second['latency_ms']:.0f} ms)"

            record("Tool calling (2-step)", tool_loop)
        return report


def build_llm(settings, db=None) -> LLM:
    order = [settings.provider] + [n for n in ("groq", "gemini") if n != settings.provider]
    providers: list[Provider] = []
    skipped: dict[str, str] = {}
    for name in order:
        if name == "groq":
            if settings.groq_key:
                providers.append(Provider("groq", OpenAI(api_key=settings.groq_key, base_url=GROQ_URL,
                                                         timeout=40, max_retries=0),
                                          pinned=settings.groq_model, effort=settings.groq_effort))
            else:
                skipped["groq"] = "No GROQ_API_KEY found. Add it to .env or to the app secrets."
        elif name == "gemini":
            if settings.gemini_key:
                providers.append(Provider("gemini", OpenAI(api_key=settings.gemini_key, base_url=GEMINI_URL,
                                                           timeout=40, max_retries=0),
                                          pinned=settings.gemini_model))
            else:
                skipped["gemini"] = "No GEMINI_API_KEY found. Add it to .env or to the app secrets."
    return LLM(providers, db=db, skipped=skipped)
