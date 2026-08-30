import numpy as np
from PIL import Image
import io

def load_image_as_array(uploaded_file):
    """Convert uploaded image to numpy array"""
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image).astype(np.float32)

def calculate_ndvi(image_array):
    """
    NDVI = (NIR - Red) / (NIR + Red)
    Since we only have RGB, we approximate:
    Red = channel 0, Green = channel 1 (used as fake NIR)
    """
    red = image_array[:, :, 0]
    green = image_array[:, :, 1]  # approximating NIR with green
    
    ndvi = (green - red) / (green + red + 1e-6)
    return ndvi

def calculate_ndwi(image_array):
    """
    NDWI = (Green - NIR) / (Green + NIR)
    Approximated using RGB
    """
    green = image_array[:, :, 1]
    red = image_array[:, :, 0]  # approximating NIR with red
    
    ndwi = (green - red) / (green + red + 1e-6)
    return ndwi

def calculate_ndbi(image_array):
    """
    NDBI approximation for built-up areas
    """
    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]
    
    ndbi = (red - blue) / (red + blue + 1e-6)
    return ndbi

def create_mask(index, threshold=0.1):
    """Create a binary mask from index"""
    mask = (index > threshold).astype(np.uint8) * 255
    return mask

def analyze_image(uploaded_file, query):
    """
    Main function that analyzes the image based on query
    """
    image_array = load_image_as_array(uploaded_file)
    query_lower = query.lower()

    result = {
        "answer": "",
        "mask": None,
        "index_name": "",
        "confidence": 0.75
    }

    if any(word in query_lower for word in ["water", "lake", "river", "pond", "ndwi"]):
        index = calculate_ndwi(image_array)
        mask = create_mask(index, threshold=0.05)
        result["answer"] = "Water bodies have been highlighted in the image using NDWI approximation."
        result["mask"] = mask
        result["index_name"] = "NDWI (Water)"
        result["confidence"] = 0.78

    elif any(word in query_lower for word in ["vegetation", "forest", "plant", "green", "ndvi"]):
        index = calculate_ndvi(image_array)
        mask = create_mask(index, threshold=0.1)
        result["answer"] = "Vegetation areas have been highlighted using NDVI approximation."
        result["mask"] = mask
        result["index_name"] = "NDVI (Vegetation)"
        result["confidence"] = 0.80

    elif any(word in query_lower for word in ["built", "building", "urban", "city", "ndbi", "construction"]):
        index = calculate_ndbi(image_array)
        mask = create_mask(index, threshold=0.1)
        result["answer"] = "Built-up / urban areas have been highlighted using NDBI approximation."
        result["mask"] = mask
        result["index_name"] = "NDBI (Built-up)"
        result["confidence"] = 0.72

    else:
        result["answer"] = "I can currently detect water, vegetation, and built-up areas. Please ask about one of these."
        result["confidence"] = 0.40

    return result