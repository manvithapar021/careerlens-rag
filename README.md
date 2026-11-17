# CareerLens — RAG Job-Description Matcher

Paste a job description. CareerLens retrieves evidence from a corpus
built from Manvi Thapar's own resume and her GitHub projects' READMEs,
then produces a grounded fit analysis: matched skills with citations,
talking points for the interview, and an honest skills-gap list.

The point is the honesty rule: **any skill in the job description
with no supporting evidence in the corpus is listed as a gap — never
claimed.** CareerLens refuses to invent experience.

## What it does

- **Retrieval-augmented generation, end to end** — ingest, chunk,
  embed, store, retrieve, compose. No hosted service is required.
- **Grounded fit analysis** — every matched skill cites the resume or
  project chunk that evidences it, with a similarity score.
- **Skills-gap list** — JD skills with no evidence are named as gaps,
  e.g. React, Docker, TensorFlow. They are never presented as skills.
- **Talking points** — interview talking points drawn from the top
  retrieved chunks, each with its source.
- **Pluggable LLM** — with an API key set, a hosted LLM (OpenAI,
  Groq or Gemini) writes the narrative from the same retrieved
  evidence. Without a key, an extractive template composer does the
  job — and the UI says so.
- **Web UI + CLI + JSON API** — the same pipeline behind all three.

## Architecture

```
Job description
      │  embed (same embedder as the corpus)
      ▼
┌─────────────┐   cosine similarity    ┌──────────────────┐
│ Vector store │ ◄──────────────────── │ SQLite + numpy   │
│ (SQLite)     │   top-k chunks       │ chunks + vectors │
└──────┬──────┘                      └──────────────────┘
       │                                     ▲
       │  evidence                    embed  │
       ▼                                     │
┌──────────────┐   ┌────────────────┐  ┌─────┴───────┐
│ Composer     │   │ skill classifier│  │ Corpus      │
│ extractive   │   │ matched vs gaps │  │ resume.md + │
│ (no key) or  │   │ (closed skill   │  │ 7 project   │
│ hosted LLM   │   │  vocabulary)    │  │ READMEs     │
└──────────────┘   └────────────────┘  └─────────────┘
```

Pipeline: `data/corpus/*.md` → chunking (120-word windows, 30-word
overlap) → embeddings → SQLite vector store → top-k retrieval →
skill extraction/classification → grounded composition.

**Embeddings.** Two backends behind one interface:

- `sentence-transformers` (`all-MiniLM-L6-v2`) when the package is
  installed — see `requirements-optional.txt`.
- A from-scratch **TF-IDF embedder** (unigrams + bigrams, sublinear
  TF, smoothed IDF, L2-normalised) built on numpy only. This is the
  default in this repo: `sentence-transformers` pulls in PyTorch,
  which needs ~2 GB of disk and a large `/tmp` for the wheel, and the
  build VM ran out of temporary space installing it. The TF-IDF
  embedder is genuinely retrieval-grade on a corpus this size — the
  tests prove a Flask query retrieves the Flask projects first — and
  the demo runs with zero downloads. Install
  `requirements-optional.txt` and rebuild the index to switch to dense
  embeddings; `get_embedder()` picks them up automatically.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python build_index.py     # builds data/careerlens.db (12 chunks)
.venv/bin/python -m careerlens.cli --jd samples/sample_jd_backend.txt
```

Or run everything with `./run.sh`, then open the web app:

```bash
.venv/bin/python -m careerlens.app   # http://127.0.0.1:5000
```

Endpoints: `GET /` (form), `POST /analyze` (HTML result),
`GET|POST /api/analyze` (JSON, `jd` param), `GET /health`.

## Using a hosted LLM (optional)

```bash
export CAREERLENS_LLM_PROVIDER=openai   # or groq, or gemini
export OPENAI_API_KEY=...               # or GROQ_API_KEY / GEMINI_API_KEY
export CAREERLENS_MODEL=gpt-4o-mini     # optional; provider defaults apply
```

Keys are read from the environment only and are never committed.
The LLM writes the narrative section under a strict system prompt
(claim only cited skills; everything else is a gap). The
deterministic matched/gap lists are computed separately and are never
overridden by the LLM. If the LLM call fails, CareerLens falls back
to the extractive composer and says so. **With no key set, no hosted
LLM is called at all** — the "extractive" mode is clearly labelled in
the UI and CLI output.

## Demo

Sample job descriptions live in `samples/`. The backend one matches
13 of 13 skills; the AI/ML one matches 8 of 14 — Python, RAG, LLMs,
embeddings, vector databases, prompt engineering, Pandas, GenAI —
and honestly gaps TensorFlow, PyTorch, machine learning, Docker,
Kubernetes and React:

```
$ .venv/bin/python -m careerlens.cli --jd samples/sample_jd_ai_ml.txt
================================================================
CareerLens — fit analysis
Mode: extractive | Embedder: tfidf
================================================================

Partial fit: 8 of 14 skills named in this job description are evidenced
by Manvi's resume and projects; 6 are gaps.

MATCHED SKILLS (8/14):
  [x] Python — Manvi Thapar — Resume Summary [resume]  (similarity 0.29)
  [x] RAG — Manvi Thapar — Resume Summary [resume]  (similarity 0.29)
  [x] LLM — Manvi Thapar — Resume Summary [resume]  (similarity 0.29)
  ...

SKILLS GAPS (not claimed — no supporting evidence):
  [ ] TensorFlow
  [ ] PyTorch
  [ ] Machine Learning
  [ ] Docker
  [ ] Kubernetes
  [ ] React
```

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q
# 15 passed
```

Tests cover chunking (sizes and overlap), TF-IDF normalisation,
retrieval relevance (a Flask query retrieves the Flask projects
first, scores descending), store round-trips, citation presence,
gap detection (React / Docker / Kubernetes are gaps, never claimed),
the no-key extractive composer, verdict counts, and the CLI. They run
offline with no API key.

## What it does not do

- **It does not call a hosted LLM without a key.** The default
  extractive mode is templates over retrieved chunks, labelled as
  such. Nothing in the output pretends to be GPT output.
- **It does not claim skills without evidence.** The gap list is a
  feature, not a bug: a JD asking for React will surface React as a
  gap even when the rest of the fit is strong.
- **It is not a general-purpose chatbot.** The corpus is one person's
  resume and projects. Retrieval quality is bounded by that corpus;
  a JD about a field with no overlap will return low scores and
  mostly gaps — which is the correct answer.
- **It is not production-hardened.** SQLite + in-process search is
  fine for a personal corpus of dozens of chunks; it is not a
  replacement for a managed vector database at scale.

## Project layout

```
careerlens/
  chunking.py     overlapping word-window chunking
  embeddings.py   TF-IDF + sentence-transformers embedders
  store.py        SQLite-backed vector store (cosine search)
  skills.py       skill vocabulary, extraction, gap classification
  composer.py     extractive composer + pluggable hosted LLM
  pipeline.py     corpus loading, indexing, end-to-end analysis
  app.py          Flask web app (+ templates/)
  cli.py          terminal demo
data/corpus/      resume + project summaries (the evidence base)
samples/          sample job descriptions for the demo
tests/            15 pytest tests
```

## Built by

Manvi Thapar — 3rd-year BTech CSE, SRMIST Kattankulathur.
Python, Flask, numpy, SQLite. No API key required to run.
