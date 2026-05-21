# RAG + LiteLLM Gateway

A modular Retrieval-Augmented Generation (RAG) gateway built on local ChromaDB embeddings and LiteLLM-powered model routing, fallback, caching, and load balancing.

This project demonstrates a phased evolution from a basic RAG pipeline into a full LiteLLM gateway with multi-provider routing, fallback resilience, cost tracking, semantic caching, and model load balancing.

---

## Architecture Overview

```
 +-------------------+            +------------------------+            +------------------+
 |                   |            |                        |            |                  |
 |  Client / Script  | <--------> |  FastAPI Gateway       | <--------> |  RAG Pipeline    |
 |                   |  HTTP/API  |  (app/main.py)         |  Python    |  (app/rag/)      |
 +-------------------+            +------------------------+            +------------------+
                                         |   |   |   |
                                         |   |   |   +------------------+
                                         |   |   |                      |
                                         v   v   v                      v
                                   +------------------+          +------------------+
                                   |  LiteLLM Router  |          |  ChromaDB        |
                                   |  (app/gateway/)  |          |  Local embeddings|
                                   +------------------+          +------------------+
                                         |   |   |   |
                                         |   |   |   +--> Groq
                                         |   |   +--> Gemini
                                         |   +--> OpenAI
                                         +--> HuggingFace
```

### Key components

- `app/rag/pipeline.py`: ingest/query orchestration
- `app/rag/vectorstore/chroma.py`: local ChromaDB storage
- `app/rag/generation/generator.py`: LiteLLM-based answer generation
- `app/gateway/router.py`: multi-provider alias routing and fallbacks
- `app/gateway/fallback.py`: LiteLLM fallback behavior and manual fallback support
- `app/gateway/cost_tracker.py`: cost estimation and recording
- `app/gateway/cache.py`: exact and semantic caching
- `app/gateway/load_balancer.py`: runtime routing strategies and circuit breaker metrics

---

## Phase Summary

### Phase 1 — Basic RAG Pipeline

Demonstrates a local RAG flow:
- ingest text
- chunk and embed with a local HuggingFace model
- store embeddings in ChromaDB
- query and generate answers via LiteLLM/OpenAI

Run:

```bash
python scripts/test_rag.py
```

### Phase 2 — LiteLLM Multi-Provider Routing

Tests the same RAG query across all configured LiteLLM aliases.
This phase shows how the gateway can route requests to multiple providers using alias-based model selection.

Run:

```bash
python scripts/test_phase2.py
```

### Phase 3 — Automatic Fallbacks

Validates fallback behavior when a primary provider fails:
- primary model responds normally
- simulated Groq failure falls back to Gemini
- simulated Groq + Gemini failure falls back to OpenAI
- manual fallback control path
- provider health tracking

Run:

```bash
python scripts/test_phase3.py
```

### Phase 4 — Cost Tracking

Adds per-call cost tracking using LiteLLM callbacks:
- estimate cost before invocation
- record actual cost per request
- summarize spend by provider, alias, and model
- report budget usage

Run:

```bash
python scripts/test_phase4.py
```

### Phase 5 — Caching

Demonstrates the gateway's caching behavior:
- cache miss and store result
- exact cache hit returns instantly with zero cost
- semantic cache hit reuses similar query responses
- cache bypass with `use_cache=False`
- alias exclusion for non-cacheable routes

Run:

```bash
python scripts/test_phase5.py
```

### Phase 6 — Load Balancing

Exercises runtime load-balancing strategies and router behavior:
- reports current strategy and available aliases
- sends repeated requests to show distribution
- compares multiple LiteLLM routing strategies
- runs concurrent requests to observe load spreading
- prints load metrics and strategy recommendations
- shows switch history

Run:

```bash
python scripts/test_phase6.py
```

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone <repo-url> rag-litellm-gateway
cd rag-litellm-gateway
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
```

Activate it:

```powershell
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install python-dotenv litellm
```

> Note: `litellm` is required for gateway routing, fallback, caching, and load balancing.

### 4. Configure environment variables

Copy the example file:

```bash
copy .env.example .env
```

Then edit `.env` and set the required keys:

- `OPENAI_API_KEY` — required for Phase 1 and paid model support
- `GROQ_API_KEY` — optional free Groq provider
- `GEMINI_API_KEY` — optional free Gemini provider
- `HUGGINGFACE_API_KEY` — optional HuggingFace provider

Optional settings include cache type, ChromaDB path, and routing strategy.

### 5. Run the API server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is exposed under `/api/v1`, including:
- `/api/v1/health`
- `/api/v1/ingest`
- `/api/v1/rag`
- `/api/v1/admin`

---

## Configuration Highlights

Configuration is managed in `app/core/config.py` and loaded from `.env`.

Important settings:

- `EMBEDDING_MODEL`: local HF embedding model used for ChromaDB
- `CHROMA_PERSIST_DIR`: local ChromaDB persistence path
- `TOP_K`: number of retrieved chunks used per query
- `DEFAULT_MODEL_ALIAS`: default LiteLLM alias when not specified
- `CACHE_TYPE`: `local`, `redis`, or `redis-semantic`
- `CACHE_DISABLED_ALIASES`: aliases excluded from the cache
- `ROUTING_STRATEGY`: LiteLLM router strategy for same-alias models
- `ALLOWED_FAILS`, `COOLDOWN_TIME`, `NUM_RETRIES`, `REQUEST_TIMEOUT`

---

## Phase Execution Commands

```bash
python scripts/test_rag.py
python scripts/test_phase2.py
python scripts/test_phase3.py
python scripts/test_phase4.py
python scripts/test_phase5.py
python scripts/test_phase6.py
```

## Notes

- ChromaDB runs locally and is free.
- The gateway is designed to use free providers first, with paid fallback models available when needed.
- Phase 5 caching supports exact and semantic reuse to reduce cost and latency.
- Phase 6 builds on LiteLLM router capabilities to switch strategies at runtime.

---

## Recommended Workflow

1. Set up `.env` with `OPENAI_API_KEY`.
2. Run `python scripts/test_rag.py` to verify basic ingestion/query.
3. Add free provider keys and run `python scripts/test_phase2.py`.
4. Run fallback, cost, cache, and load-balance phases in order.

---

## License

This repository is provided as-is for evaluation and experimentation with RAG + LiteLLM gateway design.
