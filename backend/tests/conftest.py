import io
import sys
import zipfile
from pathlib import Path
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Add backend directory to sys.path so pytest can import app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.auth import get_current_user
from app.models import User

def mock_get_current_user():
    return User(id=1, username="test_user", email="test@test.local")

app.dependency_overrides[get_current_user] = mock_get_current_user

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def sample_valid_image_bytes():
    """Generates a simple 100x100 RGB image byte stream."""
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture
def sample_small_image_bytes():
    """Generates a 10x10 RGB image byte stream (below min resolution)."""
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

@pytest.fixture
def sample_corrupt_bytes():
    return b"not an image binary payload"

@pytest.fixture
def sample_zip_bytes(sample_valid_image_bytes):
    """Generates a zip archive containing 2 valid images."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("test1.jpg", sample_valid_image_bytes)
        z.writestr("nested/test2.jpg", sample_valid_image_bytes)
        z.writestr("readme.txt", "text file should be rejected")
    return buf.getvalue()
