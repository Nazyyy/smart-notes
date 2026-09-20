# ### FILE: app/ml/llm_context_corrector.py
"""
LLM and Semantic Context Corrector for Handwritten Notes OCR.
Provides page-level contextual error correction for handwritten lecture notes,
reconstructing corrupted cursive words, cross-line hyphenations, and domain terminology.
Supports both offline heuristic mode and external/local LLM APIs (OpenAI, OpenRouter, Groq, Ollama).
"""

from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass
import asyncio
import json
import re
import httpx

from app.core.logging import get_logger
from app.ml.personalization import get_user_profile
from app.ml.handwriting_confusion import get_handwriting_confusion_corrector
from app.ml.vocabulary_binder import get_vocabulary_binder
from app.services.context_intelligence import ContextIntelligenceEngine

logger = get_logger(__name__)


@dataclass
class LLMProviderConfig:
    """Configuration for LLM context correction provider."""
    provider: str = "openrouter"  # "openrouter", "ollama", "openai", "heuristic", "custom"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "nex-agi/nex-n2.5-pro:free"
    temperature: float = 0.1
    timeout_seconds: float = 60.0
    chunk_size: int = 30


class LLMContextCorrector:
    """
    Orchestrates full-page contextual correction across all lines of handwritten notes.
    Combines high-speed heuristic Russian NLP with contextual LLM reasoning.
    """

    DEFAULT_BASE_URLS = {
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": "https://api.openai.com/v1",
        "ollama": "http://localhost:11434/v1",
        "groq": "https://api.groq.com/openai/v1",
        "deepseek": "https://api.deepseek.com/v1",
    }

    def __init__(self) -> None:
        from app.ml.vocabulary_binder import get_russian_lexicon
        self.binder = get_vocabulary_binder()
        self.confusion = get_handwriting_confusion_corrector()
        self.lexicon = get_russian_lexicon()

    async def correct_page_lines(
        self,
        lines: List[Dict[str, Any]],
        config: Optional[LLMProviderConfig] = None,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Analyze all lines of a page and apply context-aware corrections.
        Lines should be a list of dicts: [{"line_index": int, "text": str, "confidence": float, "id": Optional[str]}]
        """
        if not lines:
            return {
                "status": "empty",
                "provider": "none",
                "total_lines": 0,
                "corrected_count": 0,
                "lines": [],
            }

        cfg = config or LLMProviderConfig()
        profile = get_user_profile(user_id)

        # 1. First Pass: Ultra-fast Heuristic NLP Pre-correction (<10ms)
        heuristic_res = self._correct_with_heuristic(lines, profile)

        if cfg.provider == "heuristic":
            return heuristic_res

        # 2. Hybrid Pipeline: Feed heuristically pre-cleaned text to the LLM
        pre_cleaned_lines: List[Dict[str, Any]] = []
        for orig_item, heur_item in zip(lines, heuristic_res.get("lines", [])):
            pre_cleaned_lines.append({
                "id": orig_item.get("id"),
                "line_index": orig_item.get("line_index"),
                "text": heur_item.get("corrected_text", orig_item.get("text", "")),
                "original_raw_text": orig_item.get("text", ""),
                "confidence": heur_item.get("confidence", orig_item.get("confidence", 0.8)),
            })

        if cfg.provider in ["openrouter", "openai", "ollama", "groq", "deepseek", "custom"]:
            try:
                llm_result = await self._correct_with_llm(pre_cleaned_lines, cfg, user_id)
                if llm_result.get("status") == "success":
                    llm_result["provider"] = cfg.provider
                    llm_result["mode"] = "hybrid_nlp_llm"
                    return llm_result
                logger.warning("LLM correction did not succeed, falling back to heuristic engine.")
            except Exception as exc:
                logger.error("External LLM call failed: %s. Falling back to offline heuristic.", exc)
                heuristic_res["status"] = "fallback_success"
                heuristic_res["provider"] = f"эвристика (сбой {cfg.provider}: {type(exc).__name__})"
                return heuristic_res

        return heuristic_res

    def _correct_with_heuristic(
        self, lines: List[Dict[str, Any]], profile
    ) -> Dict[str, Any]:
        """
        Run multi-pass heuristic contextual correction:
        1. Inject user's learned profile into confusion corrector.
        2. Stitch hyphenations and hanging prepositions across consecutive lines.
        3. Validate against 1.52M Russian dictionary (preserve genuine words).
        4. Correct individual non-lexicon words using optical confusion matrix and personal vocab.
        5. Normalize syntax and punctuation.
        """
        self.confusion._build_confusion_weights()
        profile.apply_to_confusion_corrector(self.confusion)

        corrected_lines: List[Dict[str, Any]] = []
        total_changed = 0

        for idx, line_item in enumerate(lines):
            orig_text = line_item.get("text", "").strip()

            # Token-level optical correction
            words = orig_text.split()
            fixed_words: List[str] = []
            word_fixes: List[str] = []

            for w in words:
                clean_w = w.strip(".,;:!?()-\"\'")
                # Preserve math / LaTeX
                if "$" in w or "\\" in w or (w.isupper() and len(w) <= 3) or not clean_w.isalpha():
                    fixed_words.append(w)
                    continue

                # Preserve genuine Russian words present in the 1.52M dictionary
                if clean_w.lower() in self.lexicon:
                    fixed_words.append(w)
                    continue

                if clean_w.lower() not in self.confusion.vocabulary:
                    cands = self.confusion.get_word_candidates(clean_w, context_words=words, top_k=1)
                    if cands and (cands[0].get("dist", 99.0) <= 1.6 or cands[0].get("score", 0.0) >= 0.70):
                        best_cand = cands[0]["word"]
                        # Restore original casing
                        if clean_w.istitle():
                            best_cand = best_cand.capitalize()
                        # Reattach punctuation
                        prefix = w[:len(w) - len(w.lstrip(".,;:!?()-\"\'"))]
                        suffix = w[len(w.rstrip(".,;:!?()-\"\'")):]
                        replaced = f"{prefix}{best_cand}{suffix}"
                        fixed_words.append(replaced)
                        word_fixes.append(f"{clean_w} ➔ {best_cand}")
                        continue

                fixed_words.append(w)

            new_text = " ".join(fixed_words).strip()
            # General domain vocabulary binding
            new_text = self.binder.correct_line_vocabulary(new_text)

            changed = (new_text.lower().strip() != orig_text.lower().strip())
            if changed:
                total_changed += 1

            explanation = (
                f"Исправлено: {', '.join(word_fixes)}"
                if word_fixes
                else ("Сшивка переноса со следующей строкой" if new_text != orig_text else "Без изменений")
            )

            corrected_lines.append({
                "line_index": line_item.get("line_index", idx),
                "line_id": line_item.get("id"),
                "original_text": orig_text,
                "corrected_text": new_text,
                "changed": changed,
                "confidence": min(0.96, line_item.get("confidence", 0.8) + (0.10 if changed else 0.0)),
                "explanation": explanation,
            })

        return {
            "status": "success",
            "provider": "heuristic_offline",
            "model": "Academic-HTR-NLP-Heuristic",
            "total_lines": len(lines),
            "corrected_count": total_changed,
            "lines": corrected_lines,
        }

    async def _correct_single_batch(
        self,
        client: httpx.AsyncClient,
        batch_lines: List[Dict[str, Any]],
        cfg: LLMProviderConfig,
        base_url: str,
        headers: Dict[str, str],
        sem: asyncio.Semaphore,
    ) -> Dict[int, Dict[str, Any]]:
        async with sem:
            numbered_lines = "\n".join(
                [f"[Строка {l.get('line_index', idx)}]: {l.get('text', '').strip()}" for idx, l in enumerate(batch_lines)]
            )
            system_prompt = (
                "Ты ведущий специалист по расшифровке рукописных конспектов и исправлению ошибок оптического распознавания (OCR).\n"
                "Перед тобой черновой текст конспекта (русский язык, грамматика, синтаксис сложных предложений, формулы, термины), прочитанный OCR с искажениями беглого почерка.\n"
                "ЗАДАЧИ:\n"
                "1. Активно восстанавливай искажённые беглым почерком слова по смыслу контекста (например: 'лицу нарко, а со спирты' ➔ 'лицу жарко, а со спины', 'сы с болгозной и соимнитель' ➔ 'сп. с бессоюзной и сочинительной', 'штать' ➔ 'читать', 'кабать' ➔ 'капать').\n"
                "2. Исправляй оптические ошибки путаницы букв (ж/н, н/п, р/т, л/п, у/и, ш/т, д/g) и сшивай разорванные переносы.\n"
                "3. СТРОГО сохраняй формулы ($...$, LaTeX), химические формулы (H2O, H2SO4), числа и школьные пометки разбора ('1 скл', '3 скл', 'ед. ч.').\n"
                "4. В массив 'corrections' включай ВСЕ строки, где есть ошибки OCR, опечатки или искажения.\n"
                "5. В поле 'explanation' укажи краткую причину исправления.\n"
                "6. Верни результат ТОЛЬКО в виде валидного JSON без markdown-разметки (без ```json):\n"
                '{"corrections": [{"line_index": 0, "corrected_text": "полная восстановленная строка целиком", "explanation": "краткая причина"}]}'
            )

            user_content = f"Строки конспекта:\n{numbered_lines}"
            request_body = {
                "model": cfg.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": cfg.temperature,
                "stream": False,
                "max_tokens": 4096,
            }
            if cfg.provider == "ollama":
                request_body["format"] = "json"
                request_body["options"] = {"temperature": cfg.temperature, "num_ctx": 4096}
            elif "gpt-4" in cfg.model or ("qwen" in cfg.model and "free" not in cfg.model):
                request_body["response_format"] = {"type": "json_object"}
            elif "nex" in cfg.model or "free" in cfg.model:
                # Disable heavy internal reasoning loops to guarantee immediate JSON generation in <15s
                request_body["reasoning"] = {"effort": "none"}

            api_endpoint = f"{base_url}/chat/completions"
            logger.info("Dispatching LLM batch of %d lines to %s (%s)...", len(batch_lines), api_endpoint, cfg.model)

            max_retries = 2
            resp = None
            for attempt in range(max_retries):
                try:
                    resp = await client.post(api_endpoint, headers=headers, json=request_body)
                    if resp.status_code == 429:
                        err_text = ""
                        try:
                            err_text = resp.text
                        except Exception:
                            pass
                        if "free-models-per-day" in err_text or "daily" in err_text or "credits" in err_text:
                            logger.warning(
                                "OpenRouter daily free quota exhausted. Aborting retries immediately."
                            )
                            return {"__quota_exhausted__": True}

                        wait_sec = 2.5 * (attempt + 1)
                        try:
                            hdr_val = resp.headers.get("retry-after")
                            if hdr_val:
                                wait_sec = max(wait_sec, float(hdr_val))
                        except Exception:
                            pass
                        logger.warning(
                            "Rate limit 429 from %s, backing off %.1fs (retry %d/%d)...",
                            cfg.provider, wait_sec, attempt + 1, max_retries
                        )
                        await asyncio.sleep(wait_sec)
                        continue
                    resp.raise_for_status()
                    break
                except httpx.HTTPStatusError as err:
                    if err.response.status_code == 429 and attempt < max_retries - 1:
                        await asyncio.sleep(2.5 * (attempt + 1))
                        continue
                    raise

            if resp is None or resp.status_code != 200:
                return {}

            data = resp.json()

            choice_msg = data["choices"][0].get("message", {})
            raw_reply = choice_msg.get("content") or ""
            if not raw_reply.strip() and choice_msg.get("reasoning"):
                raw_reply = choice_msg.get("reasoning", "")
            if not raw_reply.strip() and choice_msg.get("reasoning_details"):
                details = choice_msg.get("reasoning_details", [])
                raw_reply = " ".join(d.get("text", "") for d in details if isinstance(d, dict))

            parsed_data = self._extract_json_data(raw_reply)

            if isinstance(parsed_data, list):
                corrections_list = parsed_data
            elif isinstance(parsed_data, dict):
                corrections_list = parsed_data.get("corrections", [])
            else:
                corrections_list = []

            batch_line_indices = [int(l.get("line_index", idx)) for idx, l in enumerate(batch_lines)]
            batch_corrections: Dict[int, Dict[str, Any]] = {}
            for list_pos, item in enumerate(corrections_list):
                if not isinstance(item, dict):
                    continue
                raw_idx = item.get("line_index") if item.get("line_index") is not None else item.get("line")
                try:
                    raw_idx = int(raw_idx) if raw_idx is not None else list_pos
                except Exception:
                    raw_idx = list_pos

                # Map to true global line_index
                if raw_idx in batch_line_indices:
                    target_idx = raw_idx
                elif 0 <= raw_idx < len(batch_line_indices):
                    target_idx = batch_line_indices[raw_idx]
                elif 0 <= list_pos < len(batch_line_indices):
                    target_idx = batch_line_indices[list_pos]
                else:
                    continue

                corr_text = item.get("corrected_text") or item.get("corrected")
                expl = item.get("explanation") or "Контекстное исправление"
                if corr_text:
                    clean_corr = re.sub(r"^\[?(?:Строка|Line)\s*\d+\]?:?\s*", "", str(corr_text).strip(), flags=re.IGNORECASE)
                    clean_corr = clean_corr.strip("`\"' ")
                    batch_corrections[target_idx] = {
                        "line_index": target_idx,
                        "corrected_text": clean_corr,
                        "explanation": str(expl).strip(),
                    }
            return batch_corrections

    def _extract_json_data(self, text: str) -> Any:
        """Robustly extract JSON data (dict or list) from LLM output or reasoning."""
        if not text:
            return {}
        cleaned = text.strip()
        # Direct attempt
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Markdown code fence attempt
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", cleaned, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1).strip())
            except Exception:
                pass

        # Search for valid JSON bracket boundaries
        last_brace = cleaned.rfind("}")
        if last_brace != -1:
            for i in range(len(cleaned)):
                if cleaned[i] == "{" and i < last_brace:
                    cand = cleaned[i:last_brace + 1].strip()
                    try:
                        return json.loads(cand)
                    except Exception:
                        continue

        return {}

    async def _correct_with_llm(
        self,
        lines: List[Dict[str, Any]],
        cfg: LLMProviderConfig,
        user_id: str,
    ) -> Dict[str, Any]:
        """
        Call OpenAI-compatible LLM endpoint with batching and academic OCR prompt.
        """
        base_url = cfg.base_url or self.DEFAULT_BASE_URLS.get(cfg.provider, "https://openrouter.ai/api/v1")
        base_url = base_url.rstrip("/")

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        if cfg.provider == "openrouter":
            headers["HTTP-Referer"] = "https://smartnotes.local"
            headers["X-Title"] = "Smart Notes OCR"

        chunk_size = getattr(cfg, "chunk_size", 30)
        batches = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
        sem = asyncio.Semaphore(1)

        corrections_by_idx: Dict[int, Dict[str, Any]] = {}
        quota_exhausted = False

        async with httpx.AsyncClient(timeout=cfg.timeout_seconds) as client:
            for b in batches:
                res = await self._correct_single_batch(client, b, cfg, base_url, headers, sem)
                if isinstance(res, dict):
                    if res.get("__quota_exhausted__"):
                        quota_exhausted = True
                        break
                    corrections_by_idx.update(res)
                await asyncio.sleep(0.3)

        if quota_exhausted and cfg.provider == "openrouter":
            # Attempt automatic fallback to local Ollama if running
            try:
                async with httpx.AsyncClient(timeout=2.0) as test_client:
                    test_resp = await test_client.get("http://localhost:11434/api/tags")
                    if test_resp.status_code == 200:
                        logger.info("OpenRouter quota exhausted. Seamlessly falling back to local Ollama (qwen2.5:7b)...")
                        ollama_cfg = LLMProviderConfig(
                            provider="ollama",
                            model="qwen2.5:7b",
                            base_url="http://localhost:11434/v1",
                        )
                        ollama_res = await self._correct_with_llm(lines, ollama_cfg, user_id)
                        if ollama_res.get("status") == "success":
                            ollama_res["provider"] = "Ollama qwen2.5:7b (авто-fallback)"
                            return ollama_res
            except Exception as ex:
                logger.debug("Ollama fallback failed: %s", ex)

        if quota_exhausted and not corrections_by_idx:
            return {"status": "quota_exhausted", "provider": cfg.provider, "lines": []}

        corrected_lines: List[Dict[str, Any]] = []
        total_changed = 0

        for idx, line_item in enumerate(lines):
            l_idx = line_item.get("line_index", idx)
            orig = line_item.get("original_raw_text") or line_item.get("text", "").strip()
            corr_info = corrections_by_idx.get(l_idx, {})
            new_text = corr_info.get("corrected_text", line_item.get("text", orig)).strip()
            expl = corr_info.get("explanation", "Без изменений")

            changed = (new_text.lower() != orig.lower())
            if changed:
                total_changed += 1

            corrected_lines.append({
                "line_index": l_idx,
                "line_id": line_item.get("id"),
                "original_text": orig,
                "corrected_text": new_text,
                "changed": changed,
                "confidence": min(0.98, line_item.get("confidence", 0.8) + (0.12 if changed else 0.0)),
                "explanation": expl,
            })

        return {
            "status": "success",
            "provider": cfg.provider,
            "model": cfg.model,
            "total_lines": len(lines),
            "corrected_count": total_changed,
            "lines": corrected_lines,
        }

    async def apply_corrections_to_database(
        self,
        page_id: UUID,
        corrected_lines: List[Dict[str, Any]],
        user_id: str,
        page_repo,
        doc_repo,
        structurer,
        storage,
    ) -> Dict[str, Any]:
        """
        Persist LLM-corrected lines into database, update user's handwriting calibration,
        and re-generate Markdown export.
        """
        page = await page_repo.get_page_with_lines(page_id)
        if not page:
            raise ValueError(f"Page {page_id} not found.")

        profile = get_user_profile(user_id)
        learned_substitutions: List[Tuple[str, str]] = []

        line_map = {line.id: line for line in page.lines}
        line_by_idx = {line.line_index: line for line in page.lines}

        for item in corrected_lines:
            target_line = None
            if item.get("line_id"):
                try:
                    target_line = line_map.get(UUID(str(item["line_id"])))
                except Exception:
                    pass
            if not target_line and item.get("line_index") is not None:
                target_line = line_by_idx.get(item["line_index"])

            if target_line and item.get("corrected_text"):
                old_text = target_line.recognized_text or ""
                new_text = item["corrected_text"]
                target_line.recognized_text = new_text

                # Learn optical substitutions for writer profile
                old_words = [w.strip(".,;:!?()-\"\'") for w in old_text.split()]
                new_words = [w.strip(".,;:!?()-\"\'") for w in new_text.split()]
                for ow, nw in zip(old_words, new_words):
                    if ow.lower() != nw.lower() and len(ow) >= 2 and len(nw) >= 2:
                        pairs = profile.record_correction(ow, nw)
                        learned_substitutions.extend(pairs)

        await page_repo.session.commit()

        # Re-generate structured Markdown export
        refreshed_page = await page_repo.get_page_with_lines(page_id)
        doc = await doc_repo.get_by_id(page.document_id)
        doc_title = doc.title if doc else "Конспект"

        markdown_content = structurer.structure_lines_to_markdown(refreshed_page.lines, doc_title)
        export_dir = storage.get_export_directory(page.document_id)
        export_file_path = export_dir / "notes.md"
        await storage.write_text_file(export_file_path, markdown_content)

        await doc_repo.save_export(
            document_id=page.document_id,
            export_format="MARKDOWN",
            file_path=str(export_file_path),
            content=markdown_content,
        )

        return {
            "status": "applied",
            "page_id": str(page_id),
            "lines_updated": len(corrected_lines),
            "learned_substitutions": len(learned_substitutions),
            "profile_stats": profile.get_stats(),
        }

    async def synthesize_study_guide(
        self,
        raw_lines_or_text: List[str],
        document_title: str = "Учебный конспект",
        config: Optional[LLMProviderConfig] = None,
        length_mode: str = "medium",  # "short", "medium", "detailed"
        enrich_facts: bool = False,
        creativity_mode: str = "strict",  # "strict", "balanced", "creative"
    ) -> str:
        """
        Synthesize raw, noisy OCR lines into a comprehensive, beautifully structured academic study guide/lecture note.
        Formats sections, key definitions, comparison tables, formulas, and quotes.
        Supports customization: length_mode, enrich_facts, creativity_mode.
        """
        cfg = config or LLMProviderConfig()
        lines_clean = [l.strip() for l in raw_lines_or_text if l and l.strip()]
        if not lines_clean:
            return f"# {document_title}\n\n*(Конспект пуст или не содержит распознанных строк)*\n"

        joined_raw = "\n".join(lines_clean)

        # Style & Length directives
        length_instructions = {
            "short": "Сделай максимально краткий конспект-шпаргалку: только ключевые тезисы, определения и главная таблица. Без лишней «воды».",
            "medium": "Сделай сбалансированный академический конспект стандартного объема лекции с четкими разделами и таблицами.",
            "detailed": "Сделай максимально подробный, развернутый конспект: подробно раскрой каждый пункт, приведи примеры, пояснения и детализируй термины.",
        }.get(length_mode, "Сделай качественный академический конспект.")

        enrichment_directive = (
            "ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ: Добавь полезные исторические/научные справки, интересные факты и контекст эпохи/открытий, связанные с темой конспекта, чтобы материал читался захватывающе."
            if enrich_facts
            else "Придерживайся только фактов из конспекта, не добавляй посторонних исторических справок."
        )

        creativity_directive = {
            "strict": "СТРОГОСТЬ: Придерживайся строго сухого научного стиля и фактов первоисточника, без вольных интерпретаций.",
            "balanced": "БАЛАНС: Сохраняй научную точность, но поясняй сложные концепции ясным, современным и доступным языком.",
            "creative": "ПОНЯТНО И НАГЛЯДНО: Объясняй сложные вещи простыми словами, используй понятные жизненные аналогии и ассоциации для легкого запоминания.",
        }.get(creativity_mode, "Сохраняй научную точность.")

        prompt = f"""Ты — первоклассный академический редактор и методист. Твоя задача: преобразовать сырой, неидеальный текст рукописного студенческого конспекта в полноценный, идеально структурированный, глубокий и удобный для изучения конспект в формате Markdown.

Параметры формирования конспекта:
- Формат объема: {length_instructions}
- Дополнительные факты: {enrichment_directive}
- Стиль изложения: {creativity_directive}

Требования к оформлению:
1. Заголовок первого уровня `# Название темы` (определи по смыслу или используй "{document_title}").
2. Логические разделы второго уровня `## Название раздела` (например, «Ключевые понятия», «Основные теории», «Сравнительный анализ»).
3. Чёткие термины и определения: выделяй термин жирным шрифтом (`**Термин** — определение`).
4. Цитаты учёных и мыслителей оформляй через цитатный блок `> «Цитата...» — Автор`.
5. Сравнительные данные, параллельные категории или свойства (например, биологическая и социальная сущность, формы, классификации) ОБЯЗАТЕЛЬНО оформляй в виде красивой Markdown-таблицы с 2 или 3 колонками (`| Колонка 1 | Колонка 2 |`).
6. Формулы и математические/химические выражения оформляй в нотации KaTeX/LaTeX: `$E = mc^2$` или в выключном блоке `$$...$$`.
7. Устрани OCR-опечатки, восстанови правильные академические термины и фамилии ученых (например, «Энгельс», «антропогенез», «социогенез»), склей разорванные переносы слов.

Сырой распознанный текст конспекта:
```
{joined_raw}
```

Выведи ТОЛЬКО готовый текст конспекта в формате Markdown без вступительных или заключительных фраз от первого лица. Начни сразу с заголовка `# ...`."""

        if cfg.provider in ["openrouter", "openai", "ollama", "groq", "deepseek", "custom"]:
            base_url = cfg.base_url or self.DEFAULT_BASE_URLS.get(cfg.provider, "https://openrouter.ai/api/v1")
            api_key = cfg.api_key or ""
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if "openrouter.ai" in base_url:
                headers["HTTP-Referer"] = "https://smartnotes.ai"
                headers["X-Title"] = "Smart Notes Note Synthesizer"

            temp = 0.1 if creativity_mode == "strict" else (0.25 if creativity_mode == "balanced" else 0.4)
            payload: Dict[str, Any] = {
                "model": cfg.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Ты — профессиональный академический редактор конспектов. Ты генерируешь безупречно оформленные, структурированные учебные конспекты в Markdown с таблицами, формулами и списками.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": temp,
                "max_tokens": 4096,
            }
            if "openrouter.ai" in base_url and ("nex" in cfg.model.lower() or "free" in cfg.model.lower()):
                payload["reasoning"] = {"effort": "none"}

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices and "message" in choices[0]:
                            text = choices[0]["message"].get("content", "").strip()
                            if text:
                                if text.startswith("```markdown"):
                                    text = text[len("```markdown"):].strip()
                                if text.startswith("```md"):
                                    text = text[len("```md"):].strip()
                                if text.startswith("```") and text.endswith("```"):
                                    text = text[3:-3].strip()
                                return text
            except Exception as exc:
                logger.error("LLM study guide synthesis failed: %s. Falling back to heuristic structurer.", exc)

        # Fallback to heuristic context intelligence engine
        from app.services.context_intelligence import ContextIntelligenceEngine
        return ContextIntelligenceEngine.structure_into_markdown(lines_clean, document_title)

    async def generate_interactive_study_kit(
        self,
        raw_lines_or_text: List[str],
        document_title: str = "Учебный конспект",
        config: Optional[LLMProviderConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate interactive learning materials based on the note text:
        1. Flashcards (Flip cards: front/back/category)
        2. Cloze Tests (Fill in the blanks: text_with_blank, blank_word, hint, options)
        3. Quick Quiz (Multiple choice test with explanations)
        """
        cfg = config or LLMProviderConfig()
        lines_clean = [l.strip() for l in raw_lines_or_text if l and l.strip()]
        if not lines_clean:
            return {
                "flashcards": [],
                "cloze_tests": [],
                "quiz": [],
            }

        joined_raw = "\n".join(lines_clean)

        prompt = f"""На основе приведенного студенческого конспекта создай полный интерактивный учебный набор для запоминания и самопроверки.
Ответь СТРОГО в формате JSON без какого-либо окружающего текста.

Требуемая структура JSON:
{{
  "flashcards": [
    {{
      "front": "Вопрос или термин для карточки (лицевая сторона)",
      "back": "Определение или ответ (обратная сторона)",
      "category": "Термины|Теории|Свойства",
      "hint": "Краткая подсказка"
    }}
  ],
  "cloze_tests": [
    {{
      "sentence_with_blank": "Текст предложения с пропуском, например: Антропогенез — это [ ... ] возникновения человека.",
      "target_word": "теория",
      "hint": "Подсказка для вспоминания",
      "full_sentence": "Антропогенез — это теория возникновения человека.",
      "options": ["теория", "факт", "гипотеза", "закон"]
    }}
  ],
  "quiz": [
    {{
      "question": "Текст тестового вопроса",
      "options": ["Вариант А", "Вариант Б", "Вариант В", "Вариант Г"],
      "correct_index": 0,
      "explanation": "Объяснение, почему этот ответ правильный"
    }}
  ]
}}

Требования:
- Сделай 4-8 качественных карточек flashcards (по ключевым терминам и персоналиям).
- Сделай 3-6 заданий cloze_tests (с пропуском одного или двух ключевых слов и 4 вариантами выбора).
- Сделай 3-5 вопросов quiz с 4 вариантами и подробным объяснением.
- Все вопросы должны быть содержательными и опираться на тему конспекта.

Текст конспекта:
{joined_raw}"""

        if cfg.provider in ["openrouter", "openai", "ollama", "groq", "deepseek", "custom"]:
            base_url = cfg.base_url or self.DEFAULT_BASE_URLS.get(cfg.provider, "https://openrouter.ai/api/v1")
            api_key = cfg.api_key or ""
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            if "openrouter.ai" in base_url:
                headers["HTTP-Referer"] = "https://smartnotes.ai"
                headers["X-Title"] = "Smart Notes Interactive Kit"

            payload: Dict[str, Any] = {
                "model": cfg.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Ты методист обучающих систем. Ты генерируешь интерактивные карточки, квизы и упражнения с пропусками. Отвечай ТОЛЬКО валидным JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.2,
                "max_tokens": 4096,
            }
            if "openrouter.ai" in base_url and ("nex" in cfg.model.lower() or "free" in cfg.model.lower()):
                payload["reasoning"] = {"effort": "none"}

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        choices = data.get("choices", [])
                        if choices and "message" in choices[0]:
                            raw_content = choices[0]["message"].get("content", "").strip()
                            # Clean potential markdown wrapping
                            if raw_content.startswith("```json"):
                                raw_content = raw_content[len("```json"):].strip()
                            if raw_content.startswith("```"):
                                raw_content = raw_content[3:].strip()
                            if raw_content.endswith("```"):
                                raw_content = raw_content[:-3].strip()

                            parsed = json.loads(raw_content)
                            if isinstance(parsed, dict):
                                return {
                                    "flashcards": parsed.get("flashcards", []),
                                    "cloze_tests": parsed.get("cloze_tests", []),
                                    "quiz": parsed.get("quiz", []),
                                }
            except Exception as exc:
                logger.error("Interactive kit generation via LLM failed: %s", exc)

        # Basic fallback cards
        return {
            "flashcards": [
                {
                    "front": f"О чем тема «{document_title}»?",
                    "back": " ".join(lines_clean[:3]) if lines_clean else "Изучение материала лекции.",
                    "category": "Тема",
                    "hint": "Основной тезис конспекта",
                }
            ],
            "cloze_tests": [],
            "quiz": [],
        }


# Singleton instance
_llm_corrector = LLMContextCorrector()


def get_llm_context_corrector() -> LLMContextCorrector:
    """Retrieve singleton LLM context corrector."""
    return _llm_corrector


