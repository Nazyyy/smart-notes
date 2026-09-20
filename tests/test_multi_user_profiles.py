# ### FILE: tests/test_multi_user_profiles.py
"""
Unit tests for Multi-User Handwriting Personalization and Profile Isolation.
Verifies that individual writers have separated confusion matrices, vocabulary, and statistics.
"""

import pytest
import tempfile
from pathlib import Path
from app.ml.personalization import (
    UserPersonalizationProfile,
    list_all_profiles,
    create_user_profile,
    get_user_profile,
    sanitize_user_id,
)


def test_sanitize_user_id():
    assert sanitize_user_id("Дмитрий") == "дмитрий"
    assert sanitize_user_id("Student 1 / Test") == "student_1_test"
    assert sanitize_user_id("   ") == "user"


def test_user_profile_isolation():
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = Path(tmp_dir)

        # Profile A
        prof_a = UserPersonalizationProfile(user_id="user_a", display_name="Автор А", storage_dir=storage)
        pairs_a = prof_a.record_correction("собака", "сабака")  # learned о->а
        prof_a.record_correction("конспект", "канспект")

        # Profile B
        prof_b = UserPersonalizationProfile(user_id="user_b", display_name="Автор Б", storage_dir=storage)
        pairs_b = prof_b.record_correction("физика", "физика")  # identical, no learned pairs
        prof_b.record_correction("документ", "токумент")  # learned д->т

        assert prof_a.total_corrections == 2
        assert prof_b.total_corrections == 1

        stats_a = prof_a.get_stats()
        stats_b = prof_b.get_stats()

        assert stats_a["display_name"] == "Автор А"
        assert stats_b["display_name"] == "Автор Б"
        assert stats_a["total_corrections"] == 2
        assert stats_b["total_corrections"] == 1

        # Check that user_a confusions are NOT present in user_b
        assert "д->т" not in prof_a.character_confusions
        assert "о->а" in prof_a.character_confusions
        assert "д->т" in prof_b.character_confusions
        assert "о->а" not in prof_b.character_confusions


def test_list_all_profiles():
    with tempfile.TemporaryDirectory() as tmp_dir:
        storage = Path(tmp_dir)
        create_user_profile("dmitriy", display_name="Дмитрий", storage_dir=storage)
        create_user_profile("alice", display_name="Алиса", storage_dir=storage)

        profiles = list_all_profiles(storage_dir=storage)
        user_ids = [p["user_id"] for p in profiles]

        assert "default" in user_ids  # Always initialized
        assert "dmitriy" in user_ids
        assert "alice" in user_ids
