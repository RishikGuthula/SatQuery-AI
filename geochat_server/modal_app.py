"""
GeoChat-7B Production Serverless Deployment on Modal.

Supports scale-to-zero, cold-start model caching, GPU auto-scaling,
and exposes the exact same /health and /v1/analyze API contract.

To deploy to production:
    pip install modal
    modal setup
    modal deploy geochat_server/modal_app.py
"""

from __future__ import annotations

import os

try:
    import modal
    HAS_MODAL = True
except ImportError:
    HAS_MODAL = False

if HAS_MODAL:
    app = modal.App("satquery-geochat-gpu")

    # Define container image with GPU dependencies
    geochat_image = (
        modal.Image.debian_slim(python_version="3.10")
        .apt_install("git", "wget")
        .pip_install(
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
            "python-multipart>=0.0.6",
            "pydantic>=2.0.0",
            "Pillow>=10.0.0",
            "torch>=2.1.0",
            "torchvision",
            "transformers==4.36.2",
            "accelerate>=0.21.0",
            "bitsandbytes>=0.41.0",
            "sentencepiece>=0.1.99",
            "einops>=0.6.1",
        )
        .run_commands(
            "git clone https://github.com/mbzuai-oryx/GeoChat.git /root/GeoChat",
        )
    )

    @app.cls(
        gpu="T4",  # Or "A10G" / "A100"
        image=geochat_image,
        scaledown_window=300,  # Keep warm for 5 minutes before scale-to-zero
        secrets=[modal.Secret.from_name("satquery-secrets", required_keys=["GEOCHAT_SERVER_API_KEY"])],
    )
    class GeoChatModel:
        @modal.enter()
        def load_model(self):
            """Loads GeoChat-7B once when container starts up and keeps in VRAM."""
            import sys
            sys.path.append("/root/GeoChat")
            from geochat_server.server import load_geochat_model, MODEL_STATE
            load_geochat_model()
            self.model_state = MODEL_STATE

        @modal.asgi_app()
        def fastapi_app(self):
            """Mounts the standard FastAPI server."""
            import sys
            sys.path.append("/root/GeoChat")
            from geochat_server.server import app as fastapi_server
            return fastapi_server
