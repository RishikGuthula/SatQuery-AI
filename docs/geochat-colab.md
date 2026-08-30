# 🛰️ GeoChat-7B on Google Colab — Setup & Deployment Guide

This guide provides copy-paste ready instructions for deploying **GeoChat-7B** as a remote GPU inference backend for **SatQuery AI** using a free or Pro Google Colab GPU runtime.

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│       SatQuery AI (Client)      │
│     (Local / Streamlit / Vercel)│
└────────────────┬────────────────┘
                 │
                 │ HTTPS (Authenticated with X-API-Key)
                 ▼
┌─────────────────────────────────┐
│     Cloudflare Public Tunnel    │
│  (https://xxxx.trycloudflare.com)│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│        Google Colab GPU         │
│     • Tesla T4 / A100 GPU       │
│     • FastAPI Server (:8000)    │
│     • GeoChat-7B Model (VRAM)   │
└─────────────────────────────────┘
```

---

## ⚠️ Important Colab Limitations

> [!WARNING]
> * **Temporary Environment**: Google Colab sessions are ephemeral. When your session terminates or disconnects, the GPU memory and tunnel URL are lost.
> * **Reconnecting**: Upon restarting the Colab runtime, you must re-run the notebook cells and update `GEOCHAT_API_URL` in SatQuery AI.
> * **Production Migration**: For a permanent, 24/7 production backend, deploy `geochat_server/server.py` to **RunPod**, **Modal**, **Hugging Face Dedicated Endpoints**, or an AWS/GCP GPU VM. No changes to the SatQuery agent code are required—only `GEOCHAT_API_URL` changes.

---

## 📋 Colab Notebook Step-by-Step Instructions

Create a new Google Colab notebook and set the hardware accelerator to **GPU** (`Runtime` -> `Change runtime type` -> `T4 GPU` or `A100 GPU`).

### Cell 1: Check GPU Acceleration
```python
!nvidia-smi
```

---

### Cell 2: Install Dependencies & Cloudflare Tunnel
```bash
%%bash
# Install system packages & cloudflared for secure HTTPS tunnel
wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared-linux-amd64.deb

# Install Python requirements
pip install -q fastapi uvicorn[standard] python-multipart pydantic requests Pillow
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -q transformers==4.36.2 accelerate bitsandbytes sentencepiece einops
```

---

### Cell 3: Clone SatQuery Server & Set Environment
```python
import os

# Set your secret API key (Must match GEOCHAT_API_KEY in SatQuery .env)
os.environ["GEOCHAT_SERVER_API_KEY"] = "satquery-secret-token-2026"
os.environ["LOAD_IN_4BIT"] = "true"  # Enables 4-bit NF4 quantization to fit comfortably in T4 16GB VRAM
os.environ["GEOCHAT_MODEL_PATH"] = "MBZUAI/geochat-7b"
```

---

### Cell 4: Download GeoChat Server Script
```bash
%%bash
# Clone the repository or write server.py directly
git clone https://github.com/RishikGuthula/SatQuery-AI.git /content/SatQuery-AI 2>/dev/null || (cd /content/SatQuery-AI && git pull)
cp -r /content/SatQuery-AI/geochat_server /content/geochat_server
```

---

### Cell 5: Start FastAPI Server in Background
```python
import subprocess
import time

print("Starting GeoChat-7B FastAPI server...")
server_process = subprocess.Popen(
    ["uvicorn", "geochat_server.server:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Wait 30 seconds for model loading
time.sleep(25)
print("Server process started with PID:", server_process.pid)
```

---

### Cell 6: Launch Secure Cloudflare HTTPS Tunnel
```python
import subprocess
import re
import time

tunnel_process = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://localhost:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Parse tunnel public URL from output
tunnel_url = None
for line in iter(tunnel_process.stdout.readline, ''):
    match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
    if match:
        tunnel_url = match.group(1)
        break

print("================================================================")
print("🎉 SECURE GEOCHAT ENDPOINT ACTIVE!")
print("GEOCHAT_API_URL =", tunnel_url)
print("GEOCHAT_API_KEY =", os.environ["GEOCHAT_SERVER_API_KEY"])
print("================================================================")
```

---

### Cell 7: Verify Server Health
```python
import requests

resp = requests.get(
    f"{tunnel_url}/health",
    headers={"X-API-Key": os.environ["GEOCHAT_SERVER_API_KEY"]}
)
print("Health status:", resp.json())
```

---

### Cell 8: Test Real Remote Inference
```python
import requests
from PIL import Image
import io

# Create a test synthetic image
test_img = Image.new("RGB", (256, 256), color=(34, 139, 34))
buf = io.BytesIO()
test_img.save(buf, format="PNG")
buf.seek(0)

files = {"image": ("test.png", buf.getvalue(), "image/png")}
data = {"question": "What land cover is visible in this satellite imagery?"}
headers = {"X-API-Key": os.environ["GEOCHAT_SERVER_API_KEY"]}

resp = requests.post(f"{tunnel_url}/v1/analyze", files=files, data=data, headers=headers)
print("Inference Result:", resp.json())
```

---

## ⚙️ Configuring SatQuery AI to Use Your Colab Endpoint

In your SatQuery-AI `.env` file, set:

```ini
GEOCHAT_ENABLED=true
GEOCHAT_API_URL=https://your-tunnel-url.trycloudflare.com
GEOCHAT_API_KEY=satquery-secret-token-2026
GEOCHAT_TIMEOUT=120
```

Restart or run SatQuery:
```bash
streamlit run app.py
```
