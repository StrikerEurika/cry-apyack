import io

import numpy as np
from PIL import Image


def bytes_to_ndarray(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB np.ndarray."""
    img = Image.open(io.BytesIO(data))
    return np.array(img.convert("RGB"))


def ndarray_to_bytes(arr: np.ndarray, fmt: str = "png") -> bytes:
    """Encode an RGB np.ndarray into PNG/JPEG bytes."""
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format=fmt.upper())
    return buf.getvalue()
