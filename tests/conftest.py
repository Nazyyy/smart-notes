# ### FILE: tests/conftest.py
"""
Pytest Configuration and Test Fixtures.
Provides isolated database sessions, synthetic test images, and test API clients.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Set test environment overrides before app imports
os.environ["ENVIRONMENT"] = "testing"
os.environ["DEBUG"] = "false"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:////tmp/smart_notes_test_sqlite.db"
os.environ["ML_USE_TRANSFORMER"] = "false"

from app.config import Settings, get_settings
from app.db.session import Base, get_db_session
from app.api.main import app
from app.ml.weights_initializer import ensure_weights_exist


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory cleaned up after test session."""
    tmp = Path(tempfile.mkdtemp(prefix="smart_notes_test_"))
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="session")
def test_settings(temp_dir: Path) -> Settings:
    """Override application settings with isolated storage and weights paths."""
    settings = get_settings()
    settings.STORAGE_ROOT = temp_dir / "storage"
    settings.STORAGE_ROOT.mkdir(parents=True, exist_ok=True)
    settings.ML_MODEL_WEIGHTS_PATH = temp_dir / "weights" / "test_crnn_weights.pt"
    settings.ML_USE_TRANSFORMER = False
    ensure_weights_exist(settings.ML_MODEL_WEIGHTS_PATH)
    return settings


@pytest_asyncio.fixture(scope="session")
async def test_engine(temp_dir: Path):
    """Create isolated SQLite async engine for test execution and bind module engine."""
    import app.db.session as session_module
    from sqlalchemy import event

    db_file = temp_dir / "test_suite.db"
    test_db_url = f"sqlite+aiosqlite:///{db_file}"

    engine = create_async_engine(test_db_url, echo=False, connect_args={"timeout": 60})

    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_test_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    session_module.engine = engine
    session_module.async_session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with engine.begin() as conn:
        import app.models.entities  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()
    if db_file.exists():
        db_file.unlink()


@pytest_asyncio.fixture(scope="function")
async def test_db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide transactional database session for direct test fixtures."""
    import app.db.session as session_module
    async with session_module.async_session_factory() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def async_client(test_engine, test_settings: Settings) -> AsyncGenerator[AsyncClient, None]:
    """Provide HTTPX async client bound to FastAPI application with isolated sessions."""
    import app.db.session as session_module

    async def override_get_db_session() -> AsyncGenerator[AsyncSession, None]:
        async with session_module.async_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_db_session

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture
def synthetic_page_image() -> np.ndarray:
    """
    Generate a 600x800 BGR synthetic handwritten page:
    Contains white paper background, a darker gradient shadow, and multiple distinct text lines.
    """
    h, w = 800, 600
    # Paper background with slight off-white color
    img = np.full((h, w, 3), 245, dtype=np.uint8)

    # Add uneven lighting / shadow gradient across bottom right
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    gradient = ((x_coords / float(w)) + (y_coords / float(h))) * 0.35
    shadow_mask = np.clip(1.0 - gradient, 0.4, 1.0)
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * shadow_mask).astype(np.uint8)

    # Draw simulated handwritten lines with proper UTF-8 Cyrillic font
    lines_text = [
        "1. Лекция: Математический анализ",
        "Функция f(x) непрерывна на отрезке [a, b]",
        "Тогда существует интеграл \\int f(x) dx",
        "- Свойство 1: линейность оператора",
        "- Свойство 2: аддитивность меры",
        "Формула Ньютона-Лейбница: F(b) - F(a) = I",
    ]

    # Use TrueType font for genuine Cyrillic letters without '????' artifacts
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = None
    for fp in [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, 22)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    start_y = 100
    spacing = 70
    for i, text in enumerate(lines_text):
        draw.text((60, start_y + i * spacing), text, font=font, fill=(30, 20, 15))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@pytest.fixture
def synthetic_line_crop() -> np.ndarray:
    """Generate a single cropped text line image of size 40x280 with readable Cyrillic."""
    h, w = 40, 280
    line_img = np.full((h, w, 3), 250, dtype=np.uint8)
    pil_img = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = None
    for fp in [
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(fp).exists():
            try:
                font = ImageFont.truetype(fp, 20)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    draw.text((10, 8), "Тестовая строка 123", font=font, fill=(20, 20, 20))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
