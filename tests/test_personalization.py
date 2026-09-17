# ### FILE: tests/test_personalization.py
import pytest
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from app.ml.personalization import UserPersonalizationProfile
from app.ml.handwriting_confusion import HandwritingConfusionCorrector
from app.api.main import app


@pytest.fixture
def temp_profile(tmp_path):
    profile = UserPersonalizationProfile(user_id="test_user", storage_dir=tmp_path)
    yield profile
    profile.reset()


def test_profile_record_correction(temp_profile):
    profile = temp_profile

    # User corrects 'у' to 'д' in 'уокумент' -> 'документ'
    pairs = profile.record_correction("уокумент", "документ")
    assert ("у", "д") in pairs
    assert profile.total_corrections == 1
    assert "документ" in profile.custom_vocabulary
    assert profile.character_confusions.get("у->д") == 1

    # Record another correction with the same substitution
    profile.record_correction("уело", "дело")
    assert profile.character_confusions.get("у->д") == 2
    assert profile.total_corrections == 2


def test_profile_apply_to_corrector(temp_profile):
    profile = temp_profile
    profile.record_correction("уокумент", "документ")
    profile.record_correction("уело", "дело")

    corrector = HandwritingConfusionCorrector()
    base_penalty = corrector.confusion_matrix.get(("у", "д"), 0.20)
    assert base_penalty == 0.20

    profile.apply_to_confusion_corrector(corrector)
    updated_penalty = corrector.confusion_matrix.get(("у", "д"))
    assert updated_penalty < base_penalty
    assert "документ" in corrector.vocabulary


def test_profile_persistence_and_reset(temp_profile):
    profile = temp_profile
    profile.record_correction("слово", "слава")
    assert profile.profile_path.exists()

    # Load into a new object from same path
    reloaded = UserPersonalizationProfile(user_id="test_user", storage_dir=profile.storage_dir)
    assert reloaded.total_corrections == 1
    assert "о->а" in reloaded.character_confusions

    # Reset
    reloaded.reset()
    assert reloaded.total_corrections == 0
    assert not reloaded.character_confusions


def test_personalization_api():
    client = TestClient(app)

    # 1. Learn correction
    resp = client.post(
        "/api/v1/recognition/personalization/learn",
        json={"original": "неразборчива", "corrected": "неразборчиво", "user_id": "api_test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "learned"

    # 2. Get profile
    resp = client.get("/api/v1/recognition/personalization/profile", params={"user_id": "api_test"})
    assert resp.status_code == 200
    pdata = resp.json()
    assert pdata["total_corrections"] >= 1
    assert pdata["user_id"] == "api_test"

    # 3. Reset profile
    resp = client.post("/api/v1/recognition/personalization/reset", params={"user_id": "api_test"})
    assert resp.status_code == 200
