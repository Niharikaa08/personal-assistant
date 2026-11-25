# Personal Assistant

An AI assistant that manages your tasks and calendar, answers questions from your documents, and drafts emails that only go out after you approve them.

**What makes it different**

- **Safety reviewer.** A second AI agent checks every sensitive action (email, delete) against your real data before you see it. If the review cannot run, the app says so instead of pretending it passed.
- **Deterministic scheduler.** Focus time is placed by an algorithm, not by the language model, so blocks never overlap events or ignore your breaks. Re-running it never double-books a task.
- **Hybrid document search.** Semantic embeddings and keyword search are combined with Reciprocal Rank Fusion.
- **Knowledge graph and telemetry** of what the assistant knows and how it performs.
- **Evaluation Lab** (*Insights → Reliability lab*): reproducible experiments that measure the three claims above instead of just asserting them. See [Evaluation](#evaluation).
- **Resilient AI layer.** Works with Groq and Gemini, picks a currently available model automatically, falls back between providers, and explains failures in plain language.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then paste a GROQ_API_KEY and/or GEMINI_API_KEY
streamlit run app.py
```

Open **Settings → Test connection** to confirm your key works. If something is wrong it tells you what and how to fix it.

## Troubleshooting the AI connection

| Symptom | Cause | Fix |
|---|---|---|
| "model not found" / HTTP 404 | A pinned model was retired | Leave `GROQ_MODEL` and `GEMINI_MODEL` empty |
| "API key rejected" | Wrong or expired key, or quotes/spaces pasted in | Create a new key, paste it plain |
| Works, then stops | Free-tier rate or daily limit | Wait, or add the second provider as fallback |
| Edited `.env`, nothing changed | Settings are read at startup | Settings → **Reload settings** |

## Evaluation

```bash
python -m assistant.evaluation --critic     # writes docs/EVALUATION.md
pytest                                      # unit + mock-API tests
```

The scheduler and retrieval experiments need no API key. The safety-reviewer experiment uses your AI provider. The retrieval corpus is synthetic and small (16 passages, 18 questions), so treat results as a controlled comparison, not a claim about real-world accuracy.

## Deploy to Streamlit Community Cloud (free)

1. Push this folder to a GitHub repository (`.env` and `data/*.db` are git-ignored).
2. At [share.streamlit.io](https://share.streamlit.io) choose **Create app**, select the repo, branch `main`, main file `app.py`. Under *Advanced settings* choose Python 3.12.
3. In *Advanced settings → Secrets* paste the contents of `.streamlit/secrets.toml.example` with your real keys and a password.
4. Deploy. First start downloads the embedding model (about a minute).

Things to know:

- **Data is not permanent.** Community Cloud storage is temporary, so tasks and memory can reset when the app restarts or is redeployed. The demo data reloads automatically.
- **Set `APP_PASSWORD`.** Without it, anyone with the link shares one database and uses your API quota.
- **Do not set Gmail secrets** on a public deployment. Emails then stay as safe previews.
- Apps that get no visitors for a while go to sleep and wake on the next visit.

## Project layout

```
app.py                 entry point, navigation, optional password gate
views/                 Assistant, Planner, Documents, Memory, Insights, Settings
ui/                    theme, shared state, reusable components, Reliability lab UI
assistant/             agent, critic, tools, scheduler, RAG, LLM layer, evaluation
tests/                 unit tests and mock-API tests for the AI layer
```
