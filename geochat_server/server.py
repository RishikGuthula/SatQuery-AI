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
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

# Global model state (loaded ONCE during startup)
MODEL_STATE = {
    "model": None,
    "tokenizer": None,
    "image_processor": None,
    "context_len": 2048,
    "device": "cpu",
    "ready": False,
    "load_error": None,
    "use_official_builder": False,
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


def tokenizer_image_token(prompt: str, tokenizer, image_token_index: int = IMAGE_TOKEN_INDEX, return_tensors: str | None = None):
    """
    Encode prompt text into token IDs while replacing <image> with image_token_index (-200).
    Ensures image tensor embeddings align with multimodal projector weights.
    """
    import torch

    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors == "pt":
        return torch.tensor(input_ids, dtype=torch.long)
    return input_ids


def patch_clip_vision_tower():
    """
    Patch CLIPVisionTower in both imported modules and future dynamic imports
    to fix the meta-device RuntimeError during from_pretrained().
    """
    def safe_clip_vision_tower_init(self, vision_tower, args, delay_load=False):
        super(type(self), self).__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, "mm_vision_select_layer", -2)
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            from transformers import CLIPVisionConfig
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    # 1. Patch in geochat if imported
    try:
        import geochat.model.multimodal_encoder.clip_encoder as ce
        ce.CLIPVisionTower.__init__ = safe_clip_vision_tower_init
        logger.info("✅ Patched geochat.model.multimodal_encoder.clip_encoder.CLIPVisionTower.__init__")
    except Exception:
        pass

    # 2. Patch in sys.modules if any transformers_modules already loaded
    for mod_name, mod in list(sys.modules.items()):
        if "clip_encoder" in mod_name and hasattr(mod, "CLIPVisionTower"):
            mod.CLIPVisionTower.__init__ = safe_clip_vision_tower_init
            logger.info(f"✅ Patched {mod_name}.CLIPVisionTower.__init__")

    # 3. Hook dynamic module loader
    try:
        import transformers.dynamic_module_utils as dmu
        if not getattr(dmu, "_geochat_hooked", False):
            orig_get_class = dmu.get_class_from_dynamic_module

            def hooked_get_class(*args, **kwargs):
                cls = orig_get_class(*args, **kwargs)
                if hasattr(cls, "__name__") and cls.__name__ == "CLIPVisionTower":
                    cls.__init__ = safe_clip_vision_tower_init
                return cls

            dmu.get_class_from_dynamic_module = hooked_get_class
            dmu._geochat_hooked = True
    except Exception:
        pass


def load_geochat_model():
    """
    Load GeoChat-7B into GPU memory once during server startup.
    Supports MBZUAI/geochat-7b or local checkpoint directory with 4-bit quantization.
    """
    model_path = os.environ.get("GEOCHAT_MODEL_PATH", "MBZUAI/geochat-7b")
    logger.info(f"Loading GeoChat-7B from '{model_path}'...")

    # Discover and add GeoChat paths to sys.path
    possible_paths = [
        "/kaggle/working/GeoChat",
        "/kaggle/working/SatQuery-AI",
        "/content/GeoChat",
        "/content/SatQuery-AI",
        os.path.join(os.getcwd(), "GeoChat"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "GeoChat")),
    ]
    for p in possible_paths:
        if os.path.exists(p) and p not in sys.path:
            sys.path.insert(0, p)
            logger.info(f"Added '{p}' to sys.path")

    try:
        import torch
        from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        load_in_4bit = os.environ.get("LOAD_IN_4BIT", "true").lower() in ("true", "1")
        load_in_8bit = os.environ.get("LOAD_IN_8BIT", "false").lower() in ("true", "1")

        logger.info(f"Target device: {device.upper()} | Data type: {dtype} | 4-bit: {load_in_4bit} | 8-bit: {load_in_8bit}")

        # Always apply CLIP vision tower meta-device fix before loading
        patch_clip_vision_tower()

        # Strategy 1: Attempt to load via official GeoChat package if installed/cloned
        loaded = False
        try:
            from geochat.model.builder import load_pretrained_model
            from geochat.mm_utils import get_model_name_from_path

            patch_clip_vision_tower()
            model_name = get_model_name_from_path(model_path) or "geochat-7b"
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,
                model_name=model_name,
                load_8bit=load_in_8bit,
                load_4bit=load_in_4bit,
                device=device,
            )
            MODEL_STATE["use_official_builder"] = True
            loaded = True
            logger.info("✅ GeoChat loaded successfully via official geochat.model.builder.")
        except Exception as e:
            logger.warning(f"geochat builder failed or not found ({e}). Trying direct HuggingFace loader with meta-device fix...")

        # Strategy 2: Direct Hugging Face Transformers loader with 504x504 CLIP image processor
        if not loaded:
            from transformers import AutoModelForCausalLM, CLIPImageProcessor

            patch_clip_vision_tower()

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            image_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                size={"shortest_edge": 504},
                crop_size={"height": 504, "width": 504},
            )

            kwargs = {"torch_dtype": dtype}
            if load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif load_in_8bit:
                kwargs["load_in_8bit"] = True

            if device == "cuda":
                kwargs["device_map"] = "auto"

            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **kwargs
            )
            patch_clip_vision_tower()

            # Ensure vision tower weights are loaded outside of meta-device context
            vision_tower = getattr(model, "get_vision_tower", lambda: getattr(getattr(model, "model", None), "vision_tower", None))()
            if vision_tower is not None and not getattr(vision_tower, "is_loaded", False):
                logger.info("Initializing CLIP Vision Tower weights outside of meta-device context...")
                vision_tower.load_model()
                vision_tower.to(device=device, dtype=dtype)
                if hasattr(vision_tower, "image_processor") and vision_tower.image_processor is not None:
                    image_processor = vision_tower.image_processor

            context_len = getattr(model.config, "max_sequence_length", 2048)
            MODEL_STATE["use_official_builder"] = False

        MODEL_STATE["model"] = model
        MODEL_STATE["tokenizer"] = tokenizer
        MODEL_STATE["image_processor"] = image_processor
        MODEL_STATE["context_len"] = context_len
        MODEL_STATE["device"] = device
        MODEL_STATE["ready"] = True
        MODEL_STATE["load_error"] = None

        logger.info("✅ GeoChat-7B is fully initialized in GPU memory and ready for inference!")

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
        image_tensor = image_tensor.to(
            device=device,
            dtype=model.dtype if hasattr(model, "dtype") else torch.float16,
        )

        # Build prompt using GeoChat template format
        # Vicuna / GeoChat conversation format: Human: <image>\n{question}\nAssistant:
        prompt_text = f"Human: {DEFAULT_IMAGE_TOKEN}\n{question.strip()}\nAssistant:"

        # Encode input IDs replacing <image> with IMAGE_TOKEN_INDEX (-200)
        input_ids = tokenizer_image_token(prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).to(device)

        # Stopping criteria for end of generation
        stop_keywords = ["</s>", "Human:"]

        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords, tok, ids):
                self.keywords = keywords
                self.keyword_ids = [tok(kw).input_ids[0] for kw in keywords if tok(kw).input_ids]

            def __call__(self, ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                for kw_id in self.keyword_ids:
                    if ids[0, -1] == kw_id:
                        return True
                return False

        stopping_criteria = StoppingCriteriaList([KeywordsStoppingCriteria(stop_keywords, tokenizer, input_ids)])

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=400,
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
            outputs = "GeoChat analysis completed, but no descriptive text was produced."

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
