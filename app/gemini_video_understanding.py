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


def build_prompt(*, language: str, is_image: bool = False) -> str:
    # Project-wide convention: prompts are written in English.
    # The `language` parameter is treated as a preference for the *user-facing* text in the JSON fields.
    # If you want everything strictly in English, send language=en (default behavior can still be enforced here).
    preferred_language = (language or "en").strip().lower()
    media_type = "image" if is_image else "video"

    return (
        "You are a senior home-appliance support technician.\n"
        f"Analyze the {media_type} to identify the appliance type and the most likely issue.\n"
        "Return ONLY valid JSON (no markdown, no code fences, no extra text).\n"
        f'All text values (EXCEPT "transcript") should be written in this language: "{preferred_language}".\n'
        'For "transcript": extract the spoken audio as verbatim text in the original spoken language; do NOT translate.\n'
        'If some words are unclear, write "[غير واضح]" for that portion.\n'
        'If there is no speech, return an empty string for "transcript".\n'
        "If the media is unclear, say so in issue_summary and ask focused follow-up questions.\n"
        "\n"
        "IMPORTANT - Part Number Extraction:\n"
        "- Carefully examine ALL visible text in the image/video, especially:\n"
        "  * Nameplates, labels, stickers, or tags on the appliance (check all sides)\n"
        "  * Model numbers, serial numbers, or identification codes\n"
        "  * Any alphanumeric codes that look like part numbers\n"
        "  * Text on packaging, manuals, or documentation visible in the frame\n"
        "  * QR codes or barcodes (sometimes contain part numbers)\n"
        "- Look for patterns like:\n"
        "  * CHS followed by numbers (e.g., CHS199100RECiN, CHS-199100-RECiN)\n"
        "  * Alphanumeric codes with 10+ characters\n"
        "  * Codes that appear on nameplates or specification labels\n"
        "- For water heaters specifically, look for:\n"
        "  * CHS199100RECiN (Demand Duo R-Series REC)\n"
        "  * Similar patterns: CHS followed by numbers and letters\n"
        "  * Codes on installation manuals or technical data sheets visible in frame\n"
        "- Extract the EXACT part number as it appears (case-sensitive, with all characters, no spaces)\n"
        "- If you see multiple codes, prioritize:\n"
        "  1. Codes on nameplates or official labels\n"
        "  2. Codes that match known part number patterns (CHS...)\n"
        "  3. The longest/most complete alphanumeric code\n"
        "- If no part number is visible after thorough examination, return null for part_number\n"
        "\n"
        "Schema:\n"
        "{\n"
        '  "appliance_type": "string",\n'
        '  "brand_or_model": "string | null",\n'
        '  "part_number": "string | null",\n'
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


def predict_part_number(appliance_type: Optional[str], brand_or_model: Optional[str]) -> Optional[str]:
    """
    Predict Part Number based on appliance type and brand/model.
    Currently only supports water heaters with known Part Number.
    
    This function predicts the Part Number when it cannot be extracted from the video/image.
    For water heaters, it defaults to CHS199100RECiN (the only Part Number available in the database).
    
    Args:
        appliance_type: Type of appliance (e.g., "water heater", "heater")
        brand_or_model: Brand or model name (optional, used for better matching)
        
    Returns:
        Predicted Part Number or None if cannot predict
    """
    if not appliance_type:
        return None
    
    appliance_lower = appliance_type.lower()
    
    # Water heater prediction - works for all water heaters
    # Currently, the database only contains one Part Number for water heaters: CHS199100RECiN
    if "water heater" in appliance_lower or "heater" in appliance_lower:
        # For water heaters, we default to the only Part Number available in the database
        # This covers Demand Duo R-Series (REC) and other water heater models
        return "CHS199100RECiN"
    
    return None


def analyze_video_with_gemini(
    *,
    api_key: str,
    model: str,
    video_bytes: bytes,
    video_mime_type: str,
    language: str = "ar",
    user_hint: Optional[str] = None,
    is_image: bool = False,
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

    prompt = build_prompt(language=language, is_image=is_image)
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
    data.setdefault("part_number", None)
    data.setdefault("transcript", "")
    data.setdefault("issue_summary", raw_text)
    data.setdefault("confidence", None)
    
    # If part_number was not extracted, try to predict it
    extracted_part_number = data.get("part_number")
    if not extracted_part_number or (isinstance(extracted_part_number, str) and not extracted_part_number.strip()):
        predicted_part_number = predict_part_number(
            appliance_type=data.get("appliance_type"),
            brand_or_model=data.get("brand_or_model")
        )
        if predicted_part_number:
            data["part_number"] = predicted_part_number
            data["part_number_source"] = "predicted"  # Mark as predicted
        else:
            data["part_number_source"] = "not_found"
    else:
        data["part_number_source"] = "extracted"  # Mark as extracted

    return VideoUnderstandingResult(raw_text=raw_text, data=data)


