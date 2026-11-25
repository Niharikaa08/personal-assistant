"""LLM layer tests against a local mock of the Groq / Gemini OpenAI-compatible APIs.

Each scenario reproduces a failure that is documented in the provider changelogs:
retired models, Gemini's 400-for-bad-key, Gemini 3 thought signatures, Groq tool_use_failed.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from openai import OpenAI

from assistant.agent import Agent
from assistant.config import load_settings
from assistant.critic import CriticAgent
from assistant.db import Database
from assistant.llm import LLM, LLMUnavailable, Provider, pick_models
from assistant.rag import Retriever
from assistant.tools import Toolbox


class MockAPI:
    def __init__(self, models, handler):
        self.models, self.handler, self.log = models, handler, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/models"):
                    if outer.models is None:
                        return self._send(500, {"error": {"message": "boom"}})
                    return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in outer.models]})
                self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.log.append(body)
                self._send(*outer.handler(body, len(outer.log)))

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/v1"

    def client(self, key="k"):
        return OpenAI(api_key=key, base_url=self.url, timeout=5, max_retries=0)

    def close(self):
        self.server.shutdown()


def completion(content=None, tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return 200, {"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                 "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}


@pytest.fixture
def servers():
    made = []

    def make(models, handler):
        s = MockAPI(models, handler)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


MSG = [{"role": "user", "content": "hi"}]


def test_retired_pinned_groq_model_falls_through_to_a_live_one(servers):
    """The exact bug: .env pins llama-3.3-70b-versatile, which Groq switched off."""
    def handler(body, n):
        if body["model"] != "openai/gpt-oss-120b":
            return 404, {"error": {"message": f"The model `{body['model']}` does not exist or you do not have access to it.",
                                   "type": "invalid_request_error", "code": "model_not_found"}}
        return completion("hello")

    api = servers(["openai/gpt-oss-120b", "whisper-large-v3"], handler)
    p = Provider("groq", api.client(), pinned="llama-3.3-70b-versatile")
    llm = LLM([p])
    out = llm.chat(MSG)
    assert out["content"] == "hello" and out["model"] == "openai/gpt-oss-120b"
    assert "llama-3.3-70b-versatile" in p.dead
    n = len(api.log)
    llm.chat(MSG)                                   # the retired model is not retried
    assert [b["model"] for b in api.log[n:]] == ["openai/gpt-oss-120b"]
    assert api.log[-1]["reasoning_effort"] == "low"
    assert "max_completion_tokens" in api.log[-1]


def test_gemini_bad_key_is_explained_and_groq_takes_over(servers):
    def gemini_handler(body, n):                    # Gemini answers 400 (not 401) for a bad key
        return 400, [{"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                                "status": "INVALID_ARGUMENT"}}]

    gem = servers(["models/gemini-3.5-flash"], gemini_handler)
    grq = servers(["openai/gpt-oss-120b"], lambda b, n: completion("from groq"))
    gp = Provider("gemini", gem.client())
    qp = Provider("groq", grq.client())

    assert LLM([gp, qp]).chat(MSG)["provider"] == "groq"
    with pytest.raises(LLMUnavailable) as e:
        LLM([Provider("gemini", gem.client())]).chat(MSG)
    assert "API key rejected" in str(e.value) and "API key not valid" in str(e.value)


def test_gemini_free_tier_zero_quota_moves_to_next_model(servers):
    def handler(body, n):
        if body["model"] == "gemini-3.5-flash":
            return 429, [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                    "message": "Quota exceeded ... limit: 0, model: gemini-3.5-flash"}}]
        return completion("lite answered")

    api = servers(["models/gemini-3.5-flash", "models/gemini-3.5-flash-lite"], handler)
    out = LLM([Provider("gemini", api.client())]).chat(MSG)
    assert out["model"] == "gemini-3.5-flash-lite"


def test_groq_tool_use_failed_is_retried_with_new_sample(servers):
    def handler(body, n):
        if n == 1:
            return 400, {"error": {"message": "Failed to call a function.", "type": "invalid_request_error",
                                   "code": "tool_use_failed", "failed_generation": "<function=x>"}}
        return completion("ok")

    api = servers(["openai/gpt-oss-120b"], handler)
    out = LLM([Provider("groq", api.client())]).chat(MSG)
    assert out["content"] == "ok" and api.log[1]["temperature"] > api.log[0]["temperature"]


def test_models_endpoint_down_still_works_with_defaults(servers):
    api = servers(None, lambda b, n: completion("ok") if b["model"] == "openai/gpt-oss-120b"
                  else (404, {"error": {"message": "does not exist", "code": "model_not_found"}}))
    assert LLM([Provider("groq", api.client())]).chat(MSG)["content"] == "ok"


def test_gemini3_thought_signature_survives_a_full_agent_tool_loop(servers, tmp_path):
    """Gemini 3 rejects the 2nd step of a tool loop unless the signature is echoed back."""
    def handler(body, n):
        assistants = [m for m in body["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
        if not assistants:
            return completion(None, [{"id": "c1", "type": "function",
                                      "function": {"name": "list_tasks", "arguments": "{}"},
                                      "extra_content": {"google": {"thought_signature": "SIG123"}}}])
        sig = (assistants[-1]["tool_calls"][0].get("extra_content") or {}).get("google", {}).get("thought_signature")
        if sig != "SIG123":
            return 400, [{"error": {"code": 400, "status": "INVALID_ARGUMENT",
                                    "message": "Function call is missing a thought_signature in functionCall parts."}}]
        tool_msgs = [m for m in body["messages"] if m["role"] == "tool"]
        assert tool_msgs and tool_msgs[0]["name"] == "list_tasks"       # Gemini wants the tool name
        assert assistants[-1]["content"] is None                         # and no empty-string content
        return completion("You have no tasks.")

    api = servers(["models/gemini-3.5-flash"], handler)
    s = load_settings()
    db = Database(tmp_path / "g.db")
    llm = LLM([Provider("gemini", api.client())], db=db)
    agent = Agent(llm, Toolbox(db, Retriever(db), s), db, s)
    res = agent.send("what is on my list?")
    assert res["reply"] == "You have no tasks." and not res["error"]


def test_signature_missing_after_provider_switch_uses_documented_dummy(servers):
    def handler(body, n):
        a = [m for m in body["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
        sig = (a[-1]["tool_calls"][0].get("extra_content") or {}).get("google", {}).get("thought_signature") if a else "n/a"
        if sig is None:
            return 400, [{"error": {"code": 400, "message": "Function call is missing a thought_signature"}}]
        return completion("fine")

    api = servers(["models/gemini-3.5-flash"], handler)
    msgs = MSG + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "g1", "type": "function", "function": {"name": "list_tasks", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "g1", "content": "none"}]
    p = Provider("gemini", api.client())
    assert LLM([p]).chat(msgs)["content"] == "fine" and p.dummy_signature


def test_diagnose_reports_each_check_and_hint(servers):
    api = servers(["models/gemini-3.5-flash"], lambda b, n: (400, [{"error": {"message": "API key not valid."}}]))
    report = LLM([Provider("gemini", api.client())], skipped={"groq": "No GROQ_API_KEY found."}).diagnose()
    by = {r["provider"]: r for r in report}
    assert by["groq"]["configured"] is False
    g = by["gemini"]
    assert g["checks"][0]["ok"] is True and g["checks"][1]["ok"] is False and "key" in g["hint"].lower()


def test_diagnose_passes_on_a_healthy_provider(servers):
    def handler(body, n):
        if body.get("tools") and not any(m["role"] == "tool" for m in body["messages"]):
            return completion(None, [{"id": "t1", "type": "function", "function": {"name": "ping", "arguments": '{"message":"hello"}'}}])
        return completion("pong")

    api = servers(["openai/gpt-oss-120b"], handler)
    g = LLM([Provider("groq", api.client())]).diagnose()[0]
    assert [c["ok"] for c in g["checks"]] == [True, True, True] and g["model"] == "openai/gpt-oss-120b"


def test_pick_models_prefers_listed_and_skips_non_chat_models():
    assert pick_models("groq", ["whisper-large-v3", "openai/gpt-oss-120b", "meta-llama/llama-prompt-guard-2-86m"])[0] == "openai/gpt-oss-120b"
    got = pick_models("gemini", ["gemini-3.5-flash-image", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"])
    assert got[0] == "gemini-3.5-flash" and "gemini-3.5-flash-image" not in got


def test_critic_never_fakes_a_pass(tmp_path):
    class Broken:
        def chat(self, *a, **k):
            raise LLMUnavailable("down")

    class Rambling:
        def chat(self, *a, **k):
            return {"content": "Looks fine to me!", "tool_calls": []}

    db = Database(tmp_path / "c.db")
    for llm in (Broken(), Rambling()):
        v = CriticAgent(llm, db).audit_action("goal", "send_email", {"to": "a@b.co"}, "")
        assert v["verdict"] == "UNVERIFIED" and v["score"] is None
