"""
Live GeoChat Inference Verification Script.

Tests real communication between SatQuery-AI client and the live GeoChat API endpoint.
"""

import os
import sys
import io
import time
import requests
from PIL import Image

def verify_live_geochat():
    api_url = os.environ.get("GEOCHAT_API_URL", "").strip().rstrip("/")
    api_key = os.environ.get("GEOCHAT_API_KEY", "").strip()

    if not api_url:
        print("❌ GEOCHAT_API_URL is not set.")
        print("Please set GEOCHAT_API_URL before running live verification.")
        print("Example: export GEOCHAT_API_URL=https://your-tunnel.trycloudflare.com")
        sys.exit(1)

    print(f"Connecting to GeoChat endpoint: {api_url}")
    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    # 1. Test /health
    print("Testing /health endpoint...")
    try:
        health_resp = requests.get(f"{api_url}/health", headers=headers, timeout=10)
        print(f"HTTP Status: {health_resp.status_code}")
        print(f"Response: {health_resp.json()}")
        if health_resp.status_code != 200 or health_resp.json().get("status") != "ok":
            print("❌ Health check failed.")
            sys.exit(1)
        print("✅ Health check PASS.")
    except Exception as e:
        print(f"❌ Could not reach health endpoint: {e}")
        sys.exit(1)

    # 2. Test /v1/analyze with real image
    print("\nTesting /v1/analyze with satellite image...")
    img = Image.new("RGB", (300, 300), color=(46, 139, 87))  # Sea Green
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    question = "Describe what you can see in this remote sensing image."
    files = {"image": ("satellite_test.png", buf.getvalue(), "image/png")}
    data = {"question": question}

    t0 = time.time()
    try:
        analyze_resp = requests.post(f"{api_url}/v1/analyze", files=files, data=data, headers=headers, timeout=120)
        elapsed = time.time() - t0
        print(f"HTTP Status: {analyze_resp.status_code} ({elapsed:.2f}s)")
        print(f"Response Body: {analyze_resp.text}")

        if analyze_resp.status_code == 200:
            ans = analyze_resp.json().get("answer", "")
            if ans and len(ans) > 5:
                print("\n" + "=" * 60)
                print("🎉 REAL GEOCHAT INFERENCE VERIFIED: PASS")
                print(f"Returned Text: {ans}")
                print("=" * 60)
                sys.exit(0)
            else:
                print("❌ Returned answer was empty.")
                sys.exit(1)
        else:
            print(f"❌ Analyze endpoint failed with status {analyze_resp.status_code}")
            sys.exit(1)

    except Exception as e:
        print(f"❌ Analyze request failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_live_geochat()
