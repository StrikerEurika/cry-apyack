import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def create_test_image_bytes(width: int = 512, height: int = 512) -> bytes:
    """Helper to generate dummy RGB image bytes for testing."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint():
    with client as c:
        response = c.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "device" in data
        assert "loaded_models" in data


def test_models_endpoint():
    with client as c:
        response = c.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0


def test_predict_endpoint():
    with client as c:
        img_bytes = create_test_image_bytes(512, 512)
        files = {"file": ("test.png", img_bytes, "image/png")}
        response = c.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "test.png"
        assert data["width"] == 512
        assert data["height"] == 512
        assert "crack_count" in data
        assert "crack_coverage_percentage" in data
        assert "bounding_boxes" in data
        assert "detections" in data


def test_predict_image_endpoint():
    with client as c:
        img_bytes = create_test_image_bytes(256, 256)
        files = {"file": ("test.png", img_bytes, "image/png")}
        response = c.post("/predict/image?render_type=overlay", files=files)
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"
        assert len(response.content) > 0
