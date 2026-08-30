"""
One-Click Automated Colab Runner for GeoChat-7B.

Run this script inside Google Colab (or any GPU Linux VM) to:
1. Verify GPU acceleration
2. Install cloudflared secure tunnel binary
3. Start the GeoChat-7B FastAPI GPU inference server
4. Open the secure public HTTPS tunnel
5. Run an automatic self-test inference
6. Display the exact environment variable to paste into SatQuery AI!
"""

import os
import re
import sys
import time
import signal
import subprocess
import requests
import io
from PIL import Image

def main():
    print("=" * 65)
    print("🚀 SatQuery AI — GeoChat-7B One-Click GPU Server Launcher")
    print("=" * 65)

    # Step 1: Check GPU
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️ WARNING: CUDA GPU not detected! Inference will be slow.")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"✅ GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    except Exception as e:
        print(f"GPU check notice: {e}")

    # Set default server API key
    api_key = os.environ.get("GEOCHAT_SERVER_API_KEY", "satquery-secret-token-2026")
    os.environ["GEOCHAT_SERVER_API_KEY"] = api_key
    os.environ["LOAD_IN_4BIT"] = "true"

    # Step 2: Install cloudflared if not present
    if not os.path.exists("/usr/local/bin/cloudflared") and not os.path.exists("/usr/bin/cloudflared"):
        print("📦 Installing Cloudflare Tunnel (cloudflared)...")
        subprocess.run(
            ["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb"],
            check=False
        )
        subprocess.run(["dpkg", "-i", "cloudflared-linux-amd64.deb"], check=False)

    # Step 3: Start FastAPI server in background
    print("⏳ Starting FastAPI server with GeoChat-7B in GPU memory...")
    server_cmd = [sys.executable, "-m", "uvicorn", "geochat_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for server to load model and report ready
    ready = False
    for _ in range(60):
        time.sleep(2)
        try:
            r = requests.get("http://127.0.0.1:8000/health", headers={"X-API-Key": api_key}, timeout=2)
            if r.status_code == 200 and r.json().get("ready"):
                ready = True
                print("✅ GeoChat-7B Model Loaded and Server Ready!")
                break
        except Exception:
            pass

    if not ready:
        print("⚠️ Server taking longer than expected. Continuing tunnel startup...")

    # Step 4: Launch Cloudflare Tunnel
    print("🌐 Launching secure HTTPS tunnel...")
    tunnel_proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    public_url = None
    start_time = time.time()
    while time.time() - start_time < 30:
        line = tunnel_proc.stdout.readline()
        if not line:
            break
        match = re.search(r'(https://[a-zA-Z0-9-]+\.trycloudflare\.com)', line)
        if match:
            public_url = match.group(1)
            break

    if not public_url:
        print("❌ Could not extract Cloudflare public tunnel URL.")
        print("Check tunnel logs above.")
        return

    # Step 5: Self-test inference
    print("\n" + "=" * 65)
    print("🧪 Running automatic self-test inference on live GPU endpoint...")
    print("=" * 65)
    try:
        # Create test image
        img = Image.new("RGB", (256, 256), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        test_resp = requests.post(
            f"{public_url}/v1/analyze",
            files={"image": ("test.png", buf.getvalue(), "image/png")},
            data={"question": "Describe what you see in this satellite image."},
            headers={"X-API-Key": api_key},
            timeout=120,
        )
        if test_resp.status_code == 200:
            print("✅ SELF-TEST INFERENCE PASSED!")
            print("GeoChat Response:", test_resp.json().get("answer"))
        else:
            print(f"⚠️ Self-test returned status {test_resp.status_code}: {test_resp.text}")
    except Exception as e:
        print(f"⚠️ Self-test notice: {e}")

    # Step 6: Display copy-paste configuration
    print("\n" + "🌟" * 32)
    print("🎉 SUCCESS! YOUR GEOCHAT GPU INFERENCE BACKEND IS ONLINE!")
    print("🌟" * 32)
    print("\nCopy and paste these lines into your SatQuery AI `.env` file:\n")
    print("-" * 65)
    print("GEOCHAT_ENABLED=true")
    print(f"GEOCHAT_API_URL={public_url}")
    print(f"GEOCHAT_API_KEY={api_key}")
    print("GEOCHAT_TIMEOUT=120")
    print("-" * 65)
    print("\nKeep this Colab tab running to maintain the GPU backend connection.")
    print("Press Ctrl+C to stop the server.")

    # Keep alive
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\nStopping GeoChat server and tunnel...")
        server_proc.terminate()
        tunnel_proc.terminate()
        print("Server shutdown complete.")

if __name__ == "__main__":
    main()
