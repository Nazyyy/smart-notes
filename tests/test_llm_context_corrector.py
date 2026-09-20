# ### FILE: tests/test_llm_context_corrector.py
"""
Unit tests for LLM Context Corrector.
Verifies offline heuristic corrections, KaTeX formula preservation,
and external LLM client mocking.
"""

import pytest
from unittest.mock import patch, AsyncMock
from app.ml.llm_context_corrector import (
    LLMContextCorrector,
    LLMProviderConfig,
    get_llm_context_corrector,
)


@pytest.mark.asyncio
async def test_llm_corrector_offline_heuristic():
    corrector = get_llm_context_corrector()
    lines = [
        {"line_index": 0, "text": "Антропосоциогенез - пронсхожденне", "confidence": 0.70},
        {"line_index": 1, "text": "Формула $x^2 + y^2 = r^2$ и $H_2SO_4$", "confidence": 0.95},
    ]

    cfg = LLMProviderConfig(provider="heuristic")
    result = await corrector.correct_page_lines(lines, config=cfg, user_id="test_user")

    assert result["status"] == "success"
    assert result["provider"] == "heuristic_offline"
    assert len(result["lines"]) == 2

    # Verify math formula is preserved
    line1 = result["lines"][1]
    assert "$x^2 + y^2 = r^2$" in line1["corrected_text"]
    assert "$H_2SO_4$" in line1["corrected_text"]


@pytest.mark.asyncio
async def test_llm_corrector_external_api_mock():
    corrector = get_llm_context_corrector()
    lines = [
        {"line_index": 0, "text": "Человек - биогоугольное существо", "confidence": 0.75},
        {"line_index": 1, "text": "создал труд", "confidence": 0.80},
    ]

    mock_llm_response = {
        "choices": [
            {
                "message": {
                    "content": '{"corrections": [{"line_index": 0, "corrected_text": "Человек - биосоциальное существо", "explanation": "биогоугольное -> биосоциальное"}, {"line_index": 1, "corrected_text": "Человека создал труд.", "explanation": "восстановлена цитата"}]}'
                }
            }
        ]
    }

    cfg = LLMProviderConfig(
        provider="openrouter",
        api_key="mock_key",
        model="qwen/qwen-2.5-7b-instruct",
    )

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        mock_resp.json = lambda: mock_llm_response
        mock_post.return_value = mock_resp

        result = await corrector.correct_page_lines(lines, config=cfg, user_id="default")

        assert result["status"] == "success"
        assert result["provider"] == "openrouter"
        assert result["corrected_count"] == 2
        assert result["lines"][0]["corrected_text"] == "Человек - биосоциальное существо"
        assert result["lines"][1]["corrected_text"] == "Человека создал труд."


@pytest.mark.asyncio
async def test_llm_batch_reindexing_protection():
    corrector = get_llm_context_corrector()
    # 5 lines so it splits into 2 batches (batch 0: lines 0..3, batch 1: line 4)
    lines = [
        {"line_index": 0, "text": "строка 0", "confidence": 0.8},
        {"line_index": 1, "text": "строка 1", "confidence": 0.8},
        {"line_index": 2, "text": "строка 2", "confidence": 0.8},
        {"line_index": 3, "text": "строка 3", "confidence": 0.8},
        {"line_index": 4, "text": "строка 4", "confidence": 0.8},
    ]

    # Mock responses where both batches return local index 0
    resp_batch_0 = {
        "choices": [{"message": {"content": '{"corrections": [{"line_index": 0, "corrected_text": "исправленная 0"}]}'}}]
    }
    resp_batch_1 = {
        "choices": [{"message": {"content": '{"corrections": [{"line_index": 0, "corrected_text": "исправленная 4"}]}'}}]
    }

    call_count = 0

    async def mock_post(*args, **kwargs):
        nonlocal call_count
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = lambda: None
        if call_count == 0:
            mock_resp.json = lambda: resp_batch_0
        else:
            mock_resp.json = lambda: resp_batch_1
        call_count += 1
        return mock_resp

    cfg = LLMProviderConfig(provider="openrouter", api_key="test_key", chunk_size=4)

    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        result = await corrector.correct_page_lines(lines, config=cfg)

        assert result["status"] == "success"
        res_lines = {l["line_index"]: l["corrected_text"] for l in result["lines"]}
        # Verify batch 1 (with local index 0) did NOT overwrite line 0!
        assert res_lines[0] == "исправленная 0"
        assert res_lines[4] == "исправленная 4"
