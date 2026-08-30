"""
GeoChat-7B Standalone GPU Inference Service.

A production-grade FastAPI server for running GeoChat-7B inference inside a
GPU environment (Google Colab, Kaggle, RunPod, or dedicated GPU VM).
Exposes authenticated /health and /v1/analyze endpoints over HTTPS tunnel.
"""

from __future__ import annotations

import io
import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Security, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("geochat_server")

# Maximum upload size: 50 MB
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Global model state (loaded ONCE during startup)
MODEL_STATE = {
    "model": None,
    "tokenizer": None,
    "image_processor": None,
    "context_len": 2048,
    "device": "cpu",
    "ready": False,
    "load_error": None,
}

# --- Authentication Configuration ---
SERVER_API_KEY = os.environ.get("GEOCHAT_SERVER_API_KEY", "").strip()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    api_key_hdr: str | None = Security(api_key_header),
    bearer_token: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> bool:
    """Validate API key from either X-API-Key or Authorization: Bearer header."""
    if not SERVER_API_KEY:
        # If no key set on server, allow requests with warning in log
        return True

    token = None
    if api_key_hdr:
        token = api_key_hdr.strip()
    elif bearer_token:
        token = bearer_token.credentials.strip()

    if not token or token != SERVER_API_KEY:
        logger.warning("Unauthorized access attempt rejected.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True


def load_geochat_model():
    """
    Load GeoChat-7B into GPU memory once during server startup.
    Supports MBZUAI/geochat-7b or local checkpoint directory.
    """
    model_path = os.environ.get("GEOCHAT_MODEL_PATH", "MBZUAI/geochat-7b")
    logger.info(f"Loading GeoChat-7B from '{model_path}'...")

    try:
        import torch
        from transformers import AutoTokenizer, AutoConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        load_in_4bit = os.environ.get("LOAD_IN_4BIT", "false").lower() in ("true", "1")
        load_in_8bit = os.environ.get("LOAD_IN_8BIT", "false").lower() in ("true", "1")

        logger.info(f"Target device: {device.upper()} | Data type: {dtype} | 4-bit: {load_in_4bit} | 8-bit: {load_in_8bit}")

        # Attempt to import GeoChat specific model classes if repository is cloned,
        # otherwise use standard AutoModelForCausalLM / LLaVA architecture
        try:
            # When cloned from https://github.com/mbzuai-oryx/GeoChat
            from geochat.model.builder import load_pretrained_model
            from geochat.mm_utils import get_model_name_from_path

            model_name = get_model_name_from_path(model_path)
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                load_8bit=load_in_8bit,
                load_4bit=load_in_4bit,
                device=device,
            )
            logger.info("GeoChat loaded via official geochat.model.builder.")
        except ImportError:
            logger.info("geochat package not found in sys.path. Loading via HuggingFace Transformers...")
            from transformers import AutoModelForCausalLM, CLIPImageProcessor

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                size={"shortest_edge": 504},
                crop_size={"height": 504, "width": 504},
            )

            kwargs = {"torch_dtype": dtype}
            if load_in_4bit:
                kwargs["load_in_4bit"] = True
            elif load_in_8bit:
                kwargs["load_in_8bit"] = True
            elif device == "cuda":
                kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            context_len = 2048

        MODEL_STATE["model"] = model
        MODEL_STATE["tokenizer"] = tokenizer
        MODEL_STATE["image_processor"] = image_processor
        MODEL_STATE["context_len"] = context_len
        MODEL_STATE["device"] = device
        MODEL_STATE["ready"] = True
        MODEL_STATE["load_error"] = None

        logger.info("✅ GeoChat-7B loaded successfully and ready for inference!")

    except Exception as e:
        logger.error(f"❌ Failed to load GeoChat-7B: {e}", exc_info=True)
        MODEL_STATE["ready"] = False
        MODEL_STATE["load_error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: load model at startup, clean up on shutdown."""
    load_geochat_model()
    yield
    logger.info("Shutting down GeoChat server.")
    MODEL_STATE.clear()


app = FastAPI(
    title="GeoChat-7B GPU Inference Service",
    description="Dedicated remote GPU inference API for SatQuery AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check(authenticated: bool = Depends(verify_api_key)):
    """Health check endpoint. Returns GPU memory and model readiness."""
    gpu_mem = {}
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            device_name = torch.cuda.get_device_name(0)
            gpu_mem = {
                "gpu_name": device_name,
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
            }
    except Exception:
        pass

    return {
        "status": "ok" if MODEL_STATE["ready"] else "error",
        "model": "geochat-7b",
        "ready": MODEL_STATE["ready"],
        "device": MODEL_STATE["device"],
        "gpu": gpu_mem,
        "error": MODEL_STATE["load_error"],
    }


class AnalysisResponse(BaseModel):
    answer: str
    model: str
    metadata: dict[str, Any]


@app.post("/v1/analyze", response_model=AnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    question: str = Form(...),
    context: str = Form(default=""),
    authenticated: bool = Depends(verify_api_key),
):
    """
    Main inference endpoint. Preprocesses uploaded image to 504x504 tensor,
    formats prompt with GeoChat conversation template, runs generation, and returns answer.
    """
    if not MODEL_STATE["ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"GeoChat model is not loaded. Error: {MODEL_STATE['load_error']}",
        )

    t_start = time.time()

    # Read and validate image
    img_bytes = await image.read()
    if len(img_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"Image exceeds max size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    orig_width, orig_height = pil_img.size

    # Run inference
    try:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        model = MODEL_STATE["model"]
        tokenizer = MODEL_STATE["tokenizer"]
        image_processor = MODEL_STATE["image_processor"]
        device = MODEL_STATE["device"]

        # Preprocess image to tensor (504x504 for GeoChat-7B)
        image_tensor = image_processor.preprocess(pil_img, return_tensors="pt")["pixel_values"]
        image_tensor = image_tensor.to(device=device, dtype=model.dtype if hasattr(model, "dtype") else torch.float16)

        # Build prompt using GeoChat template format
        # GeoChat conversation uses: Human: <image>\n{question}\nAssistant:
        prompt_text = f"Human: <image>\n{question.strip()}\nAssistant:"

        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)

        # Stopping criteria for end of generation
        stop_str = "</s>"
        keywords = ["</s>", "Human:"]

        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords, tokenizer, input_ids):
                self.keywords = keywords
                self.keyword_ids = [tokenizer(kw).input_ids[0] for kw in keywords if tokenizer(kw).input_ids]

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for kw_id in self.keyword_ids:
                    if input_ids[0, -1] == kw_id:
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(keywords, tokenizer, input_ids)])

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=350,
                use_cache=True,
                stopping_criteria=stopping_criteria,
            )

        # Decode output tokens
        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        # Clean trailing stop tokens or conversational artifacts
        if outputs.endswith("</s>"):
            outputs = outputs[:-4].strip()
        if "Human:" in outputs:
            outputs = outputs.split("Human:")[0].strip()

        if not outputs:
            outputs = "Analysis completed, but no descriptive text was produced."

        elapsed = time.time() - t_start
        logger.info(f"Inference completed in {elapsed:.2f}s for query: '{question}'")

        return AnalysisResponse(
            answer=outputs,
            model="geochat-7b",
            metadata={
                "processing_time_seconds": round(elapsed, 3),
                "original_dimensions": f"{orig_width}x{orig_height}",
                "input_tensor_shape": list(image_tensor.shape),
                "device": device,
            },
        )

    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
