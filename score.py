from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Tuple

DEFAULT_WEIGHTS = {
    "fluency": 0.30,
    "grammar": 0.25,
    "vocabulary": 0.20,
    "coherence": 0.20,
    "clarity_proxy": 0.05,
}

LANGUAGE_NAMES = {
    "en": "English",
    "ja": "Japanese",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ms": "Malay",
    "ta": "Tamil",
}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _extract_pause_spans(pauses: Any) -> List[Tuple[float, float]]:
    if pauses is None:
        return []

    spans: List[Tuple[float, float]] = []
    items: Iterable[Any]

    if isinstance(pauses, dict):
        items = pauses.values()
    elif isinstance(pauses, list):
        items = pauses
    else:
        return spans

    for item in items:
        start = None
        end = None
        if isinstance(item, dict):
            start = item.get("start")
            end = item.get("end")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]

        if start is None or end is None:
            continue

        try:
            s_val = float(start)
            e_val = float(end)
        except (TypeError, ValueError):
            continue

        if e_val > s_val >= 0:
            spans.append((s_val, e_val))

    return spans


def _tokenize(transcript: str) -> List[str]:
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+|[\u0B80-\u0BFF]+|[\u3040-\u30ff\u3400-\u9fff]+|\d+", transcript.lower())


def _metrics_from_payload(transcript: str, pauses: List[Tuple[float, float]], total_duration: float) -> Dict[str, Any]:
    tokens = _tokenize(transcript)
    num_words = len(tokens)
    total_pause_seconds = sum(end - start for start, end in pauses)
    total_pause_seconds = min(total_pause_seconds, max(total_duration, 0.0))
    duration_minutes = max(total_duration / 60.0, 1e-9)

    return {
        "num_words": num_words,
        "total_duration_seconds": round(total_duration, 3),
        "pause_count": len(pauses),
        "total_pause_seconds": round(total_pause_seconds, 3),
        "wpm": round(num_words / duration_minutes if total_duration > 0 else 0.0, 2),
        "pause_ratio": round(total_pause_seconds / total_duration if total_duration > 0 else 0.0, 4),
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise json.JSONDecodeError("No JSON object found", text, 0)


def _build_llm_prompt(
    *,
    transcript: str,
    prompt: str | None,
    language: str,
    metrics: Dict[str, Any],
    weights: Dict[str, float],
) -> str:
    language_name = LANGUAGE_NAMES.get(language.split("-")[0], language)
    rubric = {
        "fluency": "flow, pauses, pace consistency",
        "grammar": "syntactic correctness and sentence well-formedness",
        "vocabulary": "range and appropriateness of word choice",
        "coherence": "relevance and logical structure",
        "clarity_proxy": "how clear/confident delivery appears from transcript and timing",
    }

    schema = {
        "overall_score": "number 0-100",
        "subscores": {
            "fluency": "number 0-100",
            "grammar": "number 0-100",
            "vocabulary": "number 0-100",
            "coherence": "number 0-100",
            "clarity_proxy": "number 0-100",
        },
        "category_feedback": {
            "fluency": ["short reason strings"],
            "grammar": ["short reason strings"],
            "vocabulary": ["short reason strings"],
            "coherence": ["short reason strings"],
            "clarity_proxy": ["short reason strings"],
        },
        "issues": [
            {
                "id": "snake_case",
                "category": "fluency|grammar|vocabulary|coherence|clarity_proxy",
                "severity": "number 0-1",
                "message": "short issue summary",
                "evidence": {"any": "json"},
                "suggestion_hint": "short actionable hint",
            }
        ],
        "strengths": [
            {
                "category": "fluency|grammar|vocabulary|coherence|clarity_proxy",
                "message": "short strength statement",
            }
        ],
        "feedback_summary": ["up to 5 concise bullets"],
    }

    return (
        "You are an expert language proficiency evaluator. "
        "Grade the learner response ONLY by qualitative judgment. "
        "Do not use deterministic formulas to compute scores.\n\n"
        f"Language: {language_name}\n"
        f"Prompt question: {prompt or '(none provided)'}\n"
        f"Learner transcript: {transcript or '(empty)'}\n"
        f"Timing context: {json.dumps(metrics, ensure_ascii=False)}\n"
        f"Weights (for overall score): {json.dumps(weights)}\n"
        f"Rubric notes: {json.dumps(rubric)}\n\n"
        "Return STRICT JSON only, matching this schema and field names exactly:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "1) Keep scores between 0 and 100.\n"
        "2) Keep severity between 0 and 1.\n"
        "3) category_feedback must explain score decisions with specific evidence from transcript/timing.\n"
        "4) issues should be sorted by severity descending.\n"
        "5) Keep reasons concise and actionable."
    )


def _call_ollama_json(prompt_text: str) -> Dict[str, Any]:
    model = os.getenv("SCORING_MODEL", "gemma3:4b")
    endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate").strip()
    if endpoint.endswith("/"):
        endpoint = endpoint[:-1]
    if "/api/" not in endpoint:
        endpoint = endpoint + "/api/generate"
    timeout_seconds = float(os.getenv("SCORING_TIMEOUT_SECONDS", "90"))

    body = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.2,
        },
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
        api_error = payload.get("error")
        if api_error:
            raise ValueError(f"Ollama API error: {api_error}")
        text = str(payload.get("response", "")).strip()
        if not text:
            raise ValueError("LLM response was empty")
        return _extract_json_object(text)


def _normalize_llm_output(llm_output: Dict[str, Any], weights: Dict[str, float]) -> Dict[str, Any]:
    subscores_in = llm_output.get("subscores") or {}
    subscores = {
        "fluency": _clamp(float(subscores_in.get("fluency", 0))),
        "grammar": _clamp(float(subscores_in.get("grammar", 0))),
        "vocabulary": _clamp(float(subscores_in.get("vocabulary", 0))),
        "coherence": _clamp(float(subscores_in.get("coherence", 0))),
        "clarity_proxy": _clamp(float(subscores_in.get("clarity_proxy", 0))),
    }

    reported_overall = llm_output.get("overall_score")
    if reported_overall is None:
        overall = sum(subscores[k] * float(weights[k]) for k in weights)
    else:
        overall = _clamp(float(reported_overall))

    weighted_breakdown = {k: round(subscores[k] * float(weights[k]), 2) for k in weights}

    category_feedback_in = llm_output.get("category_feedback") or {}
    category_feedback = {
        "fluency": list(category_feedback_in.get("fluency") or []),
        "grammar": list(category_feedback_in.get("grammar") or []),
        "vocabulary": list(category_feedback_in.get("vocabulary") or []),
        "coherence": list(category_feedback_in.get("coherence") or []),
        "clarity_proxy": list(category_feedback_in.get("clarity_proxy") or []),
    }

    issues: List[Dict[str, Any]] = []
    for item in list(llm_output.get("issues") or []):
        if not isinstance(item, dict):
            continue
        issues.append(
            {
                "id": str(item.get("id", "issue_unknown")),
                "category": str(item.get("category", "coherence")),
                "severity": round(max(0.0, min(1.0, float(item.get("severity", 0.0)))), 3),
                "message": str(item.get("message", "")),
                "evidence": item.get("evidence", {}),
                "suggestion_hint": str(item.get("suggestion_hint", "")),
            }
        )

    issues.sort(key=lambda x: x.get("severity", 0.0), reverse=True)

    strengths: List[Dict[str, Any]] = []
    for item in list(llm_output.get("strengths") or []):
        if isinstance(item, dict):
            strengths.append(
                {
                    "category": str(item.get("category", "coherence")),
                    "message": str(item.get("message", "")),
                }
            )

    feedback_summary = [str(x) for x in list(llm_output.get("feedback_summary") or [])]

    return {
        "overall_score": round(overall, 2),
        "subscores": {k: round(v, 2) for k, v in subscores.items()},
        "weighted_breakdown": weighted_breakdown,
        "category_feedback": category_feedback,
        "issues": issues,
        "strengths": strengths,
        "feedback_summary": feedback_summary,
    }


def _build_suggestion_generator_input(
    *,
    language: str,
    prompt: str | None,
    transcript: str,
    normalized: Dict[str, Any],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    top_issues = list(normalized["issues"])[:5]

    return {
        "version": "v2-llm-grader",
        "task": "generate_learner_feedback_suggestions",
        "language": language,
        "prompt": prompt,
        "learner_transcript": transcript,
        "overall_score": normalized["overall_score"],
        "subscores": normalized["subscores"],
        "metrics": metrics,
        "top_issues": top_issues,
        "strengths": list(normalized["strengths"])[:3],
        "generator_guidance": {
            "tone": "encouraging and specific",
            "max_suggestions": 3,
            "must_reference_evidence": True,
            "prioritize_categories": ["fluency", "grammar", "vocabulary", "coherence", "clarity_proxy"],
        },
        "suggestion_candidates": [
            {"focus_issue_id": issue["id"], "hint": issue.get("suggestion_hint", "")}
            for issue in top_issues
        ],
    }


def evaluate_pronunciation(
    input_payload: Dict[str, Any],
    *,
    prompt: str | None = None,
    language: str | None = None,
    weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    LLM-driven grading.

    input_payload expected keys:
    - transcript: str
    - pauses: list/dict of pause spans
    - total_duration: float (seconds)
    - language: optional BCP-47-ish code (e.g. en, ja, ms, ta)
    """
    weights = weights or DEFAULT_WEIGHTS

    transcript = str(input_payload.get("transcript", "")).strip()
    total_duration = float(input_payload.get("total_duration", 0.0) or 0.0)
    pause_spans = _extract_pause_spans(input_payload.get("pauses", []))

    resolved_language = (language or input_payload.get("language") or "en").lower()
    metrics = _metrics_from_payload(transcript, pause_spans, total_duration)

    llm_prompt = _build_llm_prompt(
        transcript=transcript,
        prompt=prompt,
        language=resolved_language,
        metrics=metrics,
        weights=weights,
    )

    llm_error = None
    model_name = os.getenv("SCORING_MODEL", "gemma3:4b")
    endpoint_env = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate").strip()
    endpoint_for_meta = endpoint_env.rstrip("/")
    if "/api/" not in endpoint_for_meta:
        endpoint_for_meta = endpoint_for_meta + "/api/generate"
    timeout_for_meta = float(os.getenv("SCORING_TIMEOUT_SECONDS", "90"))

    try:
        llm_raw = _call_ollama_json(llm_prompt)
        normalized = _normalize_llm_output(llm_raw, weights)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, socket.timeout, ValueError, json.JSONDecodeError) as exc:
        if isinstance(exc, urllib.error.HTTPError):
            try:
                err_body = exc.read().decode("utf-8")
                llm_error = f"HTTP {exc.code}: {err_body}"
            except Exception:
                llm_error = f"HTTP {exc.code}: {exc.reason}"
        else:
            llm_error = str(exc)
        normalized = {
            "overall_score": 0.0,
            "subscores": {
                "fluency": 0.0,
                "grammar": 0.0,
                "vocabulary": 0.0,
                "coherence": 0.0,
                "clarity_proxy": 0.0,
            },
            "weighted_breakdown": {k: 0.0 for k in weights},
            "category_feedback": {
                "fluency": ["LLM grader unavailable."],
                "grammar": ["LLM grader unavailable."],
                "vocabulary": ["LLM grader unavailable."],
                "coherence": ["LLM grader unavailable."],
                "clarity_proxy": ["LLM grader unavailable."],
            },
            "issues": [
                {
                    "id": "llm_unavailable",
                    "category": "coherence",
                    "severity": 1.0,
                    "message": "LLM grading service is unavailable.",
                    "evidence": {"error": llm_error},
                    "suggestion_hint": "Start the local model service and retry grading.",
                }
            ],
            "strengths": [],
            "feedback_summary": ["Unable to generate LLM feedback because grading service is unavailable."],
        }

    return {
        "overall_score": normalized["overall_score"],
        "language": resolved_language,
        "subscores": normalized["subscores"],
        "weights": weights,
        "weighted_breakdown": normalized["weighted_breakdown"],
        "metrics": metrics,
        "category_feedback": normalized["category_feedback"],
        "feedback_summary": normalized["feedback_summary"],
        "issues": normalized["issues"],
        "strengths": normalized["strengths"],
        "llm_grading": {
            "provider": "ollama",
            "model": model_name,
            "endpoint": endpoint_for_meta,
            "timeout_seconds": timeout_for_meta,
            "success": llm_error is None,
            "error": llm_error,
        },
        "suggestion_generator_input": _build_suggestion_generator_input(
            language=resolved_language,
            prompt=prompt,
            transcript=transcript,
            normalized=normalized,
            metrics=metrics,
        ),
    }


def evaluate_pronunciation_llm(transcribe_output: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_pronunciation(transcribe_output)
