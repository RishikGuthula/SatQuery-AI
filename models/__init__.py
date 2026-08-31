"""Models package for SatQuery AI."""
from models.changeformer import ChangeFormerAdapter, get_device as get_changeformer_device
from models.bifold import BIFOLDAdapter, BIFOLDInputError, BIGEARTHNET_19_CLASSES, get_device as get_bifold_device

__all__ = [
    "ChangeFormerAdapter",
    "get_changeformer_device",
    "BIFOLDAdapter",
    "BIFOLDInputError",
    "BIGEARTHNET_19_CLASSES",
    "get_bifold_device",
]
