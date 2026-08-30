import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
_model = None
_processor = None

def get_vlm_model():
    """Load model and processor lazily into GPU/CPU memory."""
    global _model, _processor
    if _model is None or _processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )
    return _model, _processor

def run_single_image_vqa(image: Image.Image, query: str) -> dict:
    """
    Analyzes a single satellite image using Qwen2-VL to answer natural language questions.
    """
    trace_log = [
        "Executing Module: Single Image VQA Engine",
        f"Model: {MODEL_ID}",
        f"User Query: '{query}'"
    ]
    
    try:
        model, processor = get_vlm_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trace_log.append(f"Inference Device: {device.upper()}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text", 
                        "text": f"You are an expert satellite imagery analyst. Answer the following question accurately based on the provided satellite image: {query}"
                    },
                ],
            }
        ]
        
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        trace_log.append("Status: VQA Generation Completed Successfully.")
        
        return {
            "answer": output_text.strip(),
            "visual_evidence": image,
            "confidence": 0.88,
            "trace": "\n".join(trace_log)
        }

    except Exception as e:
        trace_log.append(f"Error encountered: {str(e)}")
        return {
            "answer": f"Unable to process image query. Error: {str(e)}",
            "visual_evidence": image,
            "confidence": 0.00,
            "trace": "\n".join(trace_log)
        }