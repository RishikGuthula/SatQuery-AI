# 🛰️ GeoChat-7B GPU Backend & Lifecycle Guide

This document describes the complete architecture, lifecycle management, and deployment procedures for running **GeoChat-7B** on Google Colab (development) and serverless GPU providers (production).

---

## 🔄 GeoChat GPU Lifecycle

### Development Lifecycle (Google Colab)

```
Google Colab runtime started (Session starts)
                  ↓
          GPU Initialized (Tesla T4)
                  ↓
      GeoChat-7B Loaded ONCE (FastAPI lifespan)
                  ↓
      Model Resident in GPU VRAM (5.5 GB allocated)
                  ↓
         FastAPI Server Listening (:8000)
                  ↓
    ┌─────────────┴─────────────┐
    │     API Request #1        │  → Forward pass on resident model
    │     API Request #2        │  → Forward pass on resident model
    │     API Request #N        │  → Forward pass on resident model
    └─────────────┬─────────────┘
                  ↓
Colab Runtime Terminated (Manual session disconnect)
```

> [!IMPORTANT]
> **Key Lifecycle Facts:**
> 1. **Model Persistence**: GeoChat-7B is loaded **once** when the server starts and remains resident in GPU memory across all HTTP requests. It is **never** reloaded per request.
> 2. **Session-Based Hosting**: In Google Colab, the GPU runtime is session-based. The GPU does **not** automatically start or stop per individual HTTP request. Keep the Colab notebook tab open while testing.
> 3. **Production Seamless Migration**: For production scale-to-zero behavior (where GPUs automatically spin up on request and shut down when idle), use the included `geochat_server/modal_app.py` or RunPod Serverless. The SatQuery AI client requires **zero code changes**—only `GEOCHAT_API_URL` changes!

---

## ⚡ Option A: 1-Click Development on Google Colab

### Quick Setup:
1. Open Google Colab and upload [geochat_colab_backend.ipynb](file:///Users/golisairam/OSS/SatQuery-AI/geochat_colab_backend.ipynb).
2. Set runtime to **GPU** (`Runtime` → `Change runtime type` → `T4 GPU`).
3. Click **Runtime** → **Run all**.
4. The notebook runs `geochat_server/colab_runner.py`, which:
   - Installs required GPU packages.
   - Loads GeoChat-7B into GPU memory in 4-bit mode.
   - Starts the FastAPI server in the background.
   - Opens an authenticated Cloudflare HTTPS tunnel.
   - Automatically runs a self-test inference.
   - Prints the live endpoint to copy into your SatQuery `.env`.

---

## 🚀 Option B: Production Serverless GPU Deployment (Modal)

To deploy GeoChat-7B with **automatic scale-to-zero** and cold-start caching:

```bash
# 1. Install modal CLI
pip install modal

# 2. Authenticate
modal setup

# 3. Create secret for authentication
modal secret create satquery-secrets GEOCHAT_SERVER_API_KEY=your-production-secret-key

# 4. Deploy the serverless app
modal deploy geochat_server/modal_app.py
```

Modal will output your permanent HTTPS URL (e.g. `https://your-org--satquery-geochat-gpu-fastapi-app.modal.run`).

In SatQuery AI `.env`:
```ini
GEOCHAT_ENABLED=true
GEOCHAT_API_URL=https://your-org--satquery-geochat-gpu-fastapi-app.modal.run
GEOCHAT_API_KEY=your-production-secret-key
GEOCHAT_TIMEOUT=120
```

---

## 🌐 API Contract Reference

### 1. Health Check
```http
GET /health
X-API-Key: your-api-key
```

**Response:**
```json
{
  "status": "ok",
  "model": "geochat-7b",
  "ready": true,
  "device": "cuda",
  "gpu": {
    "gpu_name": "Tesla T4",
    "allocated_gb": 5.48,
    "reserved_gb": 6.10
  }
}
```

### 2. Analyze Image
```http
POST /v1/analyze
X-API-Key: your-api-key
Content-Type: multipart/form-data

image: <binary_png_or_jpeg>
question: "Describe what you can see in this remote sensing image."
context: "{\"optional_key\": \"value\"}"
```

**Response:**
```json
{
  "answer": "The satellite image shows an agricultural area with rectangular crop fields bordering a winding river...",
  "model": "geochat-7b",
  "metadata": {
    "processing_time_seconds": 1.35,
    "original_dimensions": "512x512",
    "input_tensor_shape": [1, 3, 504, 504],
    "device": "cuda"
  }
}
```
