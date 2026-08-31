# ChangeFormer Bi-Temporal Satellite Change Detection

SatQuery-AI integrates **ChangeFormer**, a Siamese Transformer neural network architecture for remote sensing bi-temporal change detection.

---

## 🎯 Purpose

ChangeFormer takes two satellite or aerial images of the same location acquired at different times ($T_1$ = Before, $T_2$ = After) and produces a pixel-accurate binary change mask along with statistical change evidence.

ChangeFormer is an evidence-extraction model, not a text-generation model. It does not invent semantic explanations or ungrounded labels.

---

## 📥 Required Input

- **Primary Image ($T_1$)**: Baseline / "Before" image (RGB or multispectral).
- **Secondary Image ($T_2$)**: Subsequent / "After" image (RGB or multispectral).
- **Spatial Alignment**: $T_1$ and $T_2$ must have matching spatial dimensions, or CRS/transform metadata allowing geospatial reprojection.

---

## 💬 Example User Queries

The unified agent routes to ChangeFormer when two images are provided with temporal change intent:

- *"What changed between these two images?"*
- *"Compare these satellite images."*
- *"Find changes between the before and after images."*
- *"Detect changes over time."*
- *"Show areas that changed."*
- *"Did the built-up area change?"*

---

## 📊 Output Schema

ChangeFormer returns an `AnalysisResult` containing:

- **Answer**: Deterministic natural language breakdown with changed pixel count, changed percentage, confidence score, device info, and input dimensions.
- **Evidence Map**: RGB overlay highlighting detected change regions in bright red over $T_1$.
- **Mask**: 2D binary uint8 numpy array (`0` = no change, `255` = changed pixel).
- **Metadata**:
  - `model`: `"ChangeFormer-BiTemporal"`
  - `changed_pixels`: Integer pixel count
  - `total_pixels`: Total spatial pixels analyzed
  - `change_percentage`: Float (e.g. `12.4%`)
  - `confidence`: Float model probability (0.0 to 1.0)
  - `device`: `"mps"` (Apple Silicon M2) or `"cpu"` / `"cuda"`

---

## ⚡ Device Strategy & Apple Silicon M2 Support

SatQuery-AI uses PyTorch device abstraction to run ChangeFormer locally:

1. **PyTorch MPS (`torch.device("mps")`)**: Automatically preferred on Mac M2 ARM processors for hardware-accelerated unified memory inference.
2. **PyTorch CPU (`torch.device("cpu")`)**: Automatic fallback if MPS is unavailable.
3. **CUDA (`torch.device("cuda")`)**: Supported seamlessly if deployed on GPU servers.

---

## 💾 Model Weights & Checkpoints

- Default checkpoint path: `models/checkpoints/changeformer_levir_cd.pth`.
- Model weight files (`.pt`, `.pth`) and `models/checkpoints/` are automatically excluded from Git via `.gitignore`.
- Configurable checkpoint path via environment variable: `CHANGEFORMER_CHECKPOINT`.
