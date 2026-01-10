from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple


_QUESTION_ENDINGS = ("?", "؟")
_SPLIT_SENTENCE_RE = re.compile(r"[\.!\?\u061F]+[\s\n]+")  # includes Arabic question mark \u061F


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def extract_user_questions(transcript: str, max_questions: int = 8) -> List[str]:
    """
    Extract user-spoken questions from a transcript.
    - Captures sentences ending with ? or Arabic ؟
    - Also heuristically captures interrogatives that may miss a question mark
    """
    if not transcript:
        return []

    # Split into sentences
    candidates: List[str] = []
    for sent in _SPLIT_SENTENCE_RE.split(transcript):
        s = _normalize_whitespace(sent)
        if not s:
            continue
        candidates.append(s)

    questions: List[str] = []
    for s in candidates:
        s_norm = s.strip()
        if not s_norm:
            continue
        # Ends with question mark?
        if s_norm.endswith(_QUESTION_ENDINGS):
            questions.append(s_norm)
            continue
        # Heuristic: interrogative start words (Arabic/English)
        if re.match(
            r"^(هل|لماذا|ليه|كيف|أين|متى|ما|ماذا|which|what|why|how|where|when)\b",
            s_norm,
            flags=re.IGNORECASE,
        ):
            questions.append(s_norm if s_norm.endswith("?") else f"{s_norm}?")

        if len(questions) >= max_questions:
            break

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for q in questions:
        qn = q.lower()
        if qn in seen:
            continue
        seen.add(qn)
        unique.append(q)
    return unique[:max_questions]


def generate_clarifying_questions(
    *,
    appliance_type: Optional[str],
    brand_or_model: Optional[str],
    issue_summary: Optional[str],
    language: str = "ar",
    max_questions: int = 8,
) -> List[str]:
    """
    Generate focused clarifying questions to diagnose appliance issues.
    Output language defaults to Arabic; switch to English if language != 'ar'.
    """
    ap = appliance_type or "الجهاز"
    br = brand_or_model or "الماركة/الموديل"
    is_ar = (language or "ar").lower().startswith("ar")

    if is_ar:
        base = [
            f"ما هو موديل {ap} بالتحديد من {br}؟",
            "منذ متى ظهرت المشكلة؟ وهل حدثت تغييرات قبلها (تنظيف، صيانة، نقل)؟",
            "هل تظهر أي رسائل خطأ أو مؤشرات غير طبيعية (أصوات، أضواء)؟",
            "هل قمت مؤخرًا بإجراء صيانة دورية أو استبدال فلاتر/جوانات؟",
            "ما هي ظروف التشغيل (الجهد الكهربائي، مصدر الماء/الغاز، الحرارة المحيطة)؟",
            "هل المشكلة مستمرة أم متقطعة؟ وما هي الخطوات التي تزيد/تقلل حدتها؟",
            "هل تم اتباع تعليمات التشغيل في دليل {brand} الخاصة بهذا الموديل؟".replace("{brand}", br),
            "هل لديك الرقم التسلسلي أو صورة لملصق البيانات؟",
        ]
    else:
        base = [
            f"What is the exact model of the {ap} from {br}?",
            "When did the issue start? Any changes prior (cleaning, service, moving)?",
            "Are there any error codes or unusual indicators (sounds, lights)?",
            "Have you recently performed maintenance or replaced filters/gaskets?",
            "What are the operating conditions (voltage, water/gas source, ambient heat)?",
            "Is the problem persistent or intermittent? What increases/decreases it?",
            f"Have you followed the official {br} operating instructions for this model?",
            "Do you have the serial number or a photo of the data plate?",
        ]

    # Tailor with known issue keywords if present
    if issue_summary:
        issue_summary = _normalize_whitespace(issue_summary)
        if is_ar:
            base.insert(0, f"هل المشكلة التالية دقيقة: \"{issue_summary}\"؟ هل يمكن وصف الأعراض بدقة أكبر؟")
        else:
            base.insert(0, f"Is this issue accurate: \"{issue_summary}\"? Can you describe symptoms more precisely?")

    # Trim to requested count
    return base[:max_questions]


def build_retrieval_queries(
    *,
    appliance_type: Optional[str],
    brand_or_model: Optional[str],
    issue_summary: Optional[str],
    user_questions: Iterable[str] = (),
    max_queries: int = 8,
) -> List[str]:
    """
    Build short retrieval queries for Chroma/BGE-M3.
    Mix of brand+appliance+issue seeds and top user questions.
    """
    ap = _normalize_whitespace(appliance_type or "")
    br = _normalize_whitespace(brand_or_model or "")
    iss = _normalize_whitespace(issue_summary or "")

    seeds: List[str] = []
    if br or ap:
        seeds.append(" ".join(x for x in [br, ap, "troubleshooting"] if x))
        seeds.append(" ".join(x for x in [br, ap, "maintenance"] if x))
        seeds.append(" ".join(x for x in [br, ap, "error codes"] if x))
        seeds.append(" ".join(x for x in [br, ap, "specifications"] if x))
        if iss:
            seeds.append(" ".join(x for x in [br, ap, iss, "diagnosis"] if x))
            seeds.append(" ".join(x for x in [br, ap, iss, "repair steps"] if x))
    else:
        if iss:
            seeds.append(" ".join(x for x in [iss, "troubleshooting"] if x))

    # Add top N user questions as-is (shortened)
    for q in user_questions:
        qt = _normalize_whitespace(q)
        if qt:
            seeds.append(qt[:140])

    # Deduplicate while preserving order and drop empties
    seen = set()
    queries: List[str] = []
    for q in seeds:
        qn = q.lower()
        if qn and qn not in seen:
            seen.add(qn)
            queries.append(q)
        if len(queries) >= max_queries:
            break
    return queries


