# 🛰️ GeoChat-7B GPU Inference Service

Standalone FastAPI GPU inference service for running **GeoChat-7B** (MBZUAI/geochat-7b) to provide remote visual reasoning for SatQuery AI.

---

## 🎯 Architecture

```
                  SatQuery AI (Client)
                          │
                          │ HTTPS (Authenticated)
                          ▼
                Secure Cloudflare Tunnel
                          │
                          ▼
            FastAPI Server (Port 8000)
                          │
                          ▼
                  GeoChat-7B Model
              (Loaded in GPU VRAM once)
```

---

## 🚀 Quickstart (Local GPU / Dedicated Server)

```bash
# 1. Clone repository & enter server directory
cd geochat_server

# 2. Install GPU dependencies
pip install -r requirements.txt

# 3. Set your secret API key
export GEOCHAT_SERVER_API_KEY="your-secret-api-key"

# 4. Start the server (load in 4-bit for 16GB VRAM, or fp16 for >=24GB)
LOAD_IN_4BIT=true uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## 🌐 Endpoints

### 1. Health Check
```http
GET /health
X-API-Key: your-secret-api-key
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
    "allocated_gb": 4.82,
    "reserved_gb": 5.20
  }
}
```

### 2. Image Analysis
```http
POST /v1/analyze
X-API-Key: your-secret-api-key
Content-Type: multipart/form-data

image: <binary_file>
question: "Describe this satellite image in detail."
context: "{\"tool_metrics\": {\"ndvi\": 0.65}}"
```

**Response:**
```json
{
  "answer": "The satellite image shows a dense forested area along a meandering river...",
  "model": "geochat-7b",
  "metadata": {
    "processing_time_seconds": 1.42,
    "original_dimensions": "1024x1024",
    "input_tensor_shape": [1, 3, 504, 504],
    "device": "cuda"
  }
}
```
