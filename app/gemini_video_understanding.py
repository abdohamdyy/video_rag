from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


class GeminiResponseParseError(ValueError):
    pass


class GeminiAPIError(RuntimeError):
    def __init__(self, *, status_code: int, message: str, retry_after_seconds: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class VideoUnderstandingResult:
    raw_text: str
    data: Dict[str, Any]


_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)


def _strip_code_fences(s: str) -> str:
    return re.sub(_FENCE_RE, "", s).strip()


def _best_effort_json_parse(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception as e:  # noqa: BLE001
        # Try to salvage the largest JSON object in the response
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:  # noqa: BLE001
                pass
        raise GeminiResponseParseError(str(e)) from e


def _extract_status_code_and_retry_after(err: Exception) -> tuple[int, Optional[int]]:
    # google-genai errors often have `status_code`; if not, we parse from the string.
    status_code = getattr(err, "status_code", None)
    if isinstance(status_code, int):
        code = status_code
    else:
        m = re.search(r"\b(\d{3})\b", str(err))
        code = int(m.group(1)) if m else 500

    retry_after = None
    # The error payload often includes retryDelay like "30s"
    m2 = re.search(r"retryDelay'\s*:\s*'(\d+)s'", str(err))
    if m2:
        retry_after = int(m2.group(1))
    else:
        m3 = re.search(r"retry in\s+(\d+)\.?(\d+)?s", str(err), flags=re.IGNORECASE)
        if m3:
            retry_after = int(m3.group(1))
    return code, retry_after


def build_prompt(*, language: str) -> str:
    # Project-wide convention: prompts are written in English.
    # The `language` parameter is treated as a preference for the *user-facing* text in the JSON fields.
    # If you want everything strictly in English, send language=en (default behavior can still be enforced here).
    preferred_language = (language or "en").strip().lower()

    return (
        "You are a senior home-appliance support technician.\n"
        "Analyze the video to identify the appliance type and the most likely issue.\n"
        "Return ONLY valid JSON (no markdown, no code fences, no extra text).\n"
        f'All text values (EXCEPT "transcript") should be written in this language: "{preferred_language}".\n'
        'For "transcript": extract the spoken audio as verbatim text in the original spoken language; do NOT translate.\n'
        'If some words are unclear, write "[غير واضح]" for that portion.\n'
        'If there is no speech, return an empty string for "transcript".\n'
        "If the video is unclear, say so in issue_summary and ask focused follow-up questions.\n"
        "Schema:\n"
        "{\n"
        '  "appliance_type": "string",\n'
        '  "brand_or_model": "string | null",\n'
        '  "transcript": "string",\n'
        '  "issue_summary": "string",\n'
        '  "likely_root_causes": ["string"],\n'
        '  "recommended_fix_steps": ["string"],\n'
        '  "tools_or_parts_needed": ["string"],\n'
        '  "safety_warnings": ["string"],\n'
        '  "questions_to_confirm": ["string"],\n'
        '  "confidence": 0.0\n'
        "}"
    )


def analyze_video_with_gemini(
    *,
    api_key: str,
    model: str,
    video_bytes: bytes,
    video_mime_type: str,
    language: str = "ar",
    user_hint: Optional[str] = None,
) -> VideoUnderstandingResult:
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        from google.genai import errors as genai_errors  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError('Missing dependency "google-genai". Install it with: pip install google-genai') from e

    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Set it in environment or .env.")

    client = genai.Client(api_key=api_key)

    prompt = build_prompt(language=language)
    if user_hint:
        prompt = f"{prompt}\n\nUser hint: {user_hint}"

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(data=video_bytes, mime_type=video_mime_type),
            ],
        )
    ]

    try:
        resp = client.models.generate_content(model=model, contents=contents)
    except Exception as e:  # noqa: BLE001
        # Map known API errors to a structured exception so the API layer can return proper HTTP status codes.
        if isinstance(e, getattr(genai_errors, "APIError", Exception)):
            status_code, retry_after = _extract_status_code_and_retry_after(e)
            raise GeminiAPIError(status_code=status_code, message=str(e), retry_after_seconds=retry_after) from e
        raise
    raw_text = (resp.text or "").strip()
    if not raw_text:
        raise RuntimeError("Gemini returned empty response text")

    data = _best_effort_json_parse(raw_text)

    # Minimal normalization
    data.setdefault("appliance_type", None)
    data.setdefault("transcript", "")
    data.setdefault("issue_summary", raw_text)
    data.setdefault("confidence", None)

    return VideoUnderstandingResult(raw_text=raw_text, data=data)


