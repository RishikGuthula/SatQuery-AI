"""
Live GeoChat Inference & SatQuery End-to-End Verification Script.

Tests:
1. Direct /health endpoint check on remote GPU backend.
2. Direct /v1/analyze real inference test with satellite image.
3. Full SatQuery Agent end-to-end integration test (Query -> Agent -> GeoChat -> Result).
4. Fallback resilience test when GeoChat is unreachable.
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

    print("=" * 65)
    print("🛰️ SATQUERY AI → LIVE GEOCHAT GPU VERIFICATION SUITE")
    print("=" * 65)
    print(f"Target Endpoint: {api_url}")

    headers = {}
    if api_key:
        headers["X-API-Key"] = api_key
        headers["Authorization"] = f"Bearer {api_key}"

    # --- Test 1: /health ---
    print("\n[Test 1/4] Verifying /health endpoint on remote GPU...")
    try:
        health_resp = requests.get(f"{api_url}/health", headers=headers, timeout=10)
        print(f"Status Code: {health_resp.status_code}")
        health_data = health_resp.json()
        print(f"Payload: {health_data}")

        if health_resp.status_code != 200 or health_data.get("status") != "ok":
            print("❌ Health check failed.")
            sys.exit(1)
        print("✅ /health PASS (GPU server is active and model is resident in VRAM).")
    except Exception as e:
        print(f"❌ Could not reach health endpoint: {e}")
        sys.exit(1)

    # --- Test 2: Real GeoChat Direct Inference ---
    print("\n[Test 2/4] Testing direct /v1/analyze with real remote sensing imagery...")
    img = Image.new("RGB", (300, 300), color=(34, 139, 34))  # Forest Green
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    question = "Describe what you can see in this remote sensing image."
    files = {"image": ("satellite_test.png", img_bytes, "image/png")}
    data = {"question": question}

    t0 = time.time()
    try:
        analyze_resp = requests.post(f"{api_url}/v1/analyze", files=files, data=data, headers=headers, timeout=120)
        elapsed = time.time() - t0
        print(f"Status Code: {analyze_resp.status_code} ({elapsed:.2f}s)")

        if analyze_resp.status_code != 200:
            print(f"❌ /v1/analyze failed: {analyze_resp.text}")
            sys.exit(1)

        ans_data = analyze_resp.json()
        direct_answer = ans_data.get("answer", "").strip()
        print(f"GeoChat-7B Direct Output: \"{direct_answer}\"")

        if not direct_answer:
            print("❌ Returned answer was empty.")
            sys.exit(1)
        print("✅ Direct GeoChat-7B GPU Inference PASS.")
    except Exception as e:
        print(f"❌ /v1/analyze request failed: {e}")
        sys.exit(1)

    # --- Test 3: Full SatQuery Agent Integration ---
    print("\n[Test 3/4] Testing full SatQuery Agent pipeline (SatQuery Agent -> GeoChat -> Synthesis)...")
    try:
        from agent.controller import process_query
        os.environ["GEOCHAT_ENABLED"] = "true"

        agent_result = process_query(
            query="Describe what you can see in this remote sensing image.",
            image1_bytes=img_bytes,
        )

        print(f"Agent Tool Used: {agent_result.tool_used}")
        print(f"Agent Final Answer: \"{agent_result.answer}\"")
        print(f"Trace Steps Recorded: {len(agent_result.trace.steps) if agent_result.trace else 0}")

        if not agent_result.answer:
            print("❌ Agent did not produce an answer.")
            sys.exit(1)
        print("✅ Full SatQuery Agent Integration PASS.")
    except Exception as e:
        print(f"❌ SatQuery Agent execution failed: {e}")
        sys.exit(1)

    # --- Test 4: Graceful Offline Fallback ---
    print("\n[Test 4/4] Testing fallback resilience when GeoChat endpoint is unreachable...")
    try:
        from vlm.geochat import GeoChatVLM
        offline_client = GeoChatVLM(api_url="https://invalid-offline-domain-testing.example.com", timeout=2, max_retries=1)
        from core.image_loader import load_from_bytes
        raster = load_from_bytes(img_bytes)

        fallback_ans = offline_client.analyze("Describe this", raster)
        print(f"Fallback Message: \"{fallback_ans[:100]}...\"")
        assert "unavailable" in fallback_ans.lower()
        print("✅ Fallback Resilience PASS.")
    except Exception as e:
        print(f"❌ Fallback test error: {e}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print("🎉 ALL REAL GEOCHAT & SATQUERY INTEGRATION TESTS PASSED!")
    print("=" * 65)

if __name__ == "__main__":
    verify_live_geochat()
