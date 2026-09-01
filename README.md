# 🛰️ SatQuery AI

**Single Agentic Multimodal Remote-Sensing Assistant**

SatQuery AI is an agentic satellite imagery assistant that automatically understands natural language questions, plans multi-step remote-sensing tasks, invokes authoritative scientific tools (NDVI, NDWI, NDBI, Change Detection) and remote GPU-hosted vision-language models (GeoChat-7B), and synthesizes grounded answers with visual evidence overlay maps.

---

## 🎯 Architecture Overview

```
                                 USER
                                  │
                                  ▼
                           STREAMLIT UI
                      (Uploads + Chat + Trace)
                                  │
                                  ▼
                      SATQUERY UNIFIED AGENT
                      (agent/controller.py)
                                  │
                                  ▼
                      LLM PLANNER / ROUTER
                  (llm/planner.py & core/planner.py)
                      (Strict Pydantic JSON Plan)
                                  │
                                  ▼
                         CAPABILITY REGISTRY
                         (core/registry.py)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
   GeoChat-7B API         Scientific Tools           Future Models
  (vlm/geochat.py)    (NDVI, NDWI, NDBI, Change)   (SAR, Seg, Objects)
        │                         │                         │
   [Colab GPU]                    │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                           EVIDENCE ENGINE
                        (evidence/engine.py)
                                  │
                                  ▼
                            LLM SYNTHESIS
                         (llm/synthesis.py)
                                  │
                                  ▼
                          ONE FINAL ANSWER
                      (Answer + Evidence Map +
                       Transparency Trace)
```

---

## ✨ Key Features

1. **Single Unified Agent Workflow**: The user uploads imagery and asks natural language questions without manually choosing tools or models.
2. **Authoritative Scientific Tools**:
   - **Vegetation**: True NDVI $(\frac{\text{NIR}-\text{Red}}{\text{NIR}+\text{Red}})$ Rouse et al. 1974
   - **Water**: True NDWI $(\frac{\text{Green}-\text{NIR}}{\text{Green}+\text{NIR}})$ McFeeters 1996
   - **Built-up**: True NDBI $(\frac{\text{SWIR}-\text{NIR}}{\text{SWIR}+\text{NIR}})$ Zha et al. 2003
   - **Change Detection**: Normalized color-space Euclidean difference mapping with adaptive thresholding and geospatial reprojection.
3. **Remote GPU GeoChat-7B Integration**: Dedicated authenticated HTTPS API running GeoChat-7B on Google Colab GPU (or dedicated GPU VMs) using 504x504 image tensors. The main app remains lightweight with zero in-app PyTorch/CUDA dependencies.
4. **Online LLM Planning & Synthesis**: OpenAI and Google Gemini provider abstractions for intent classification, task decomposition, and grounded result synthesis without metric hallucinations.
5. **Resilient Fallback Hierarchy**: If online LLM or remote GeoChat GPU are offline, the system automatically falls back to deterministic rule-based planning and local scientific tools without crashing.
6. **Execution Trace Transparency**: Interactive "How the agent solved this" UI expander displaying step-by-step reasoning, tool execution status, timing, and sensor metadata.

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone repository
git clone https://github.com/RishikGuthula/SatQuery-AI.git
cd SatQuery-AI

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac (.venv\Scripts\activate on Windows)

# Install lightweight dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

```ini
# Online LLM (OpenAI or Gemini)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=your-api-key

# GeoChat-7B Remote GPU Backend (Google Colab / Dedicated Server)
GEOCHAT_ENABLED=true
GEOCHAT_API_URL=https://your-colab-tunnel.trycloudflare.com
GEOCHAT_API_KEY=your-secret-key
GEOCHAT_TIMEOUT=120
```

> **Note**: If `.env` is left empty, SatQuery runs in **Deterministic Offline Mode** using local scientific tools and rule-based planning.

### 3. Run Application

```bash
streamlit run app.py
```

Visit `http://localhost:8501` in your browser.

---

## 🛰️ Google Colab GPU Setup for GeoChat-7B

For step-by-step copy-paste commands to launch the standalone GeoChat-7B inference server on a free or Pro Google Colab GPU runtime, see [docs/geochat-colab.md](docs/geochat-colab.md).

---

## 🧪 Running Tests

SatQuery includes a comprehensive test suite covering capability registration, LLM planning, GeoChat client mocking, synthesis grounding, session context, security limits, and remote sensing formulas:

```bash
source .venv/bin/activate
pytest -v
```

---

## 🔒 Security & Scientific Integrity

* **Zero Credential Leaks**: API keys are passed exclusively via environment variables and never logged or committed.
* **No Arbitrary Code Execution**: The agent only executes valid, pre-registered capabilities in `core/registry.py`.
* **Honest Remote Sensing Disclosures**: RGB proxy heuristics are strictly labeled and never claimed to be true multispectral satellite calculations.
* **Bounded File Limits**: Enforces `MAX_FILE_SIZE` and `MAX_DIMENSION` guards against memory exhaustion.

---

## 📄 License

MIT License. See `LICENSE` for details.
