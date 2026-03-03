from __future__ import annotations

import json
import os
import re
import socket
from typing import Any, Dict, Iterable, List, Tuple
from urllib.parse import urlparse

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency in some environments
    ollama = None

# Default category weights (sum should be 1.0)
DEFAULT_WEIGHTS = {
    "fluency": 0.30,
    "grammar": 0.25,
    "vocabulary": 0.20,
    "coherence": 0.20,
    "clarity_proxy": 0.05,
}

LANGUAGE_PROFILES: Dict[str, Dict[str, Any]] = {
    "en": {
        "token_mode": "word",
        "wpm_target_min": 90,
        "wpm_target_max": 150,
        "filler_words": {"um", "uh", "erm", "hmm", "like", "you know", "sort of", "kind of"},
        "linking_words": {"because", "so", "therefore", "however", "first", "then", "finally"},
    },
    "ja": {
        "token_mode": "cjk_char",
        "wpm_target_min": 120,
        "wpm_target_max": 220,
        "filler_words": {"えっと", "あの", "その", "まあ"},
        "linking_words": {"だから", "でも", "そして", "まず", "次に", "最後に"},
    },
    "zh": {
        "token_mode": "cjk_char",
        "wpm_target_min": 120,
        "wpm_target_max": 220,
        "filler_words": {"嗯", "那个", "这个"},
        "linking_words": {"因为", "所以", "但是", "首先", "然后", "最后"},
    },
    "es": {
        "token_mode": "word",
        "wpm_target_min": 95,
        "wpm_target_max": 165,
        "filler_words": {"eh", "este", "pues", "o sea"},
        "linking_words": {"porque", "entonces", "sin embargo", "primero", "luego", "finalmente"},
    },
    "fr": {
        "token_mode": "word",
        "wpm_target_min": 95,
        "wpm_target_max": 165,
        "filler_words": {"euh", "ben"},
        "linking_words": {"parce que", "donc", "cependant", "d'abord", "puis", "finalement"},
    },
    "de": {
        "token_mode": "word",
        "wpm_target_min": 90,
        "wpm_target_max": 155,
        "filler_words": {"äh", "hm"},
        "linking_words": {"weil", "also", "jedoch", "zuerst", "dann", "schließlich"},
    },
    "ms": {
        "token_mode": "word",
        "wpm_target_min": 95,
        "wpm_target_max": 160,
        "filler_words": {"erm", "eee", "apa", "macam", "kan"},
        "linking_words": {"kerana", "jadi", "tetapi", "pertama", "kemudian", "akhirnya"},
    },
    "ta": {
        "token_mode": "tamil_word",
        "wpm_target_min": 90,
        "wpm_target_max": 155,
        "filler_words": {"அம்", "அஹ்", "அப்படின்னா", "ம்ம்ம்"},
        "linking_words": {"ஏனெனில்", "அதனால்", "ஆனால்", "முதலில்", "பிறகு", "இறுதியாக"},
    },
}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _extract_pause_spans(pauses: Any) -> List[Tuple[float, float]]:
    """
    Accepts:
    - list[{"start": x, "end": y}]
    - dict[str, {"start": x, "end": y}] or dict[str, [x, y]]
    - list[[x, y]]
    """
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


def _get_language_profile(language: str | None) -> Dict[str, Any]:
    if not language:
        return LANGUAGE_PROFILES["en"]
    normalized = language.lower().split("-")[0].strip()
    return LANGUAGE_PROFILES.get(normalized, LANGUAGE_PROFILES["en"])


def _tokenize(transcript: str, token_mode: str = "word") -> List[str]:
    """
    Tokenizer that works for Latin words and also captures CJK runs.
    For CJK runs, split to characters as a rough "word-like" unit.
    """
    rough_tokens = re.findall(
        r"[A-Za-zÀ-ÖØ-öø-ÿ']+|[\u0B80-\u0BFF]+|[\u3040-\u30ff\u3400-\u9fff]+|\d+",
        transcript.lower(),
    )
    tokens: List[str] = []
    for token in rough_tokens:
        if token_mode == "cjk_char" and re.fullmatch(r"[\u3040-\u30ff\u3400-\u9fff]+", token):
            tokens.extend(list(token))
        else:
            tokens.append(token)
    return tokens


def _score_fluency(
    wpm: float,
    pause_ratio: float,
    filler_ratio: float,
    wpm_target_min: float,
    wpm_target_max: float,
) -> Tuple[float, List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    issues: List[Dict[str, Any]] = []
    score = 100.0

    if wpm < wpm_target_min:
        penalty = min(35.0, (wpm_target_min - wpm) * 0.45)
        score -= penalty
        reasons.append(f"wpm was {wpm:.1f} (target {wpm_target_min}-{wpm_target_max})")
        issues.append(
            {
                "id": "fluency_wpm_low",
                "category": "fluency",
                "severity": round(min(1.0, (wpm_target_min - wpm) / max(1.0, wpm_target_min)), 3),
                "message": "Speaking pace is below target.",
                "evidence": {"actual": round(wpm, 2), "target_min": wpm_target_min, "target_max": wpm_target_max},
                "suggestion_hint": "Practice short timed answers and aim for one complete sentence every 5-7 seconds.",
            }
        )
    elif wpm > wpm_target_max:
        penalty = min(20.0, (wpm - wpm_target_max) * 0.25)
        score -= penalty
        reasons.append(f"wpm was {wpm:.1f} (target {wpm_target_min}-{wpm_target_max})")
        issues.append(
            {
                "id": "fluency_wpm_high",
                "category": "fluency",
                "severity": round(min(1.0, (wpm - wpm_target_max) / max(1.0, wpm_target_max)), 3),
                "message": "Speaking pace is above target.",
                "evidence": {"actual": round(wpm, 2), "target_min": wpm_target_min, "target_max": wpm_target_max},
                "suggestion_hint": "Slow down slightly and add clearer phrase boundaries.",
            }
        )

    if pause_ratio > 0.12:
        penalty = min(45.0, (pause_ratio - 0.12) * 220)
        score -= penalty
        reasons.append(f"pause ratio was {pause_ratio * 100:.1f}% (target < 12%)")
        issues.append(
            {
                "id": "fluency_pause_ratio_high",
                "category": "fluency",
                "severity": round(min(1.0, (pause_ratio - 0.12) / 0.12), 3),
                "message": "Too much silence between chunks of speech.",
                "evidence": {"actual": round(pause_ratio, 4), "target_max": 0.12},
                "suggestion_hint": "Use simple templates and pre-plan sentence openings to reduce long pauses.",
            }
        )

    if filler_ratio > 0.03:
        penalty = min(20.0, (filler_ratio - 0.03) * 200)
        score -= penalty
        reasons.append(f"filler ratio was {filler_ratio * 100:.1f}% (target < 3%)")
        issues.append(
            {
                "id": "fluency_filler_ratio_high",
                "category": "fluency",
                "severity": round(min(1.0, (filler_ratio - 0.03) / 0.07), 3),
                "message": "Frequent filler words reduce fluency.",
                "evidence": {"actual": round(filler_ratio, 4), "target_max": 0.03},
                "suggestion_hint": "Replace fillers with short silent planning pauses (<1s).",
            }
        )

    if not reasons:
        reasons.append("speech pace and pause control were in target range")
    return _clamp(score), reasons, issues


def _score_grammar(transcript: str, tokens: List[str]) -> Tuple[float, List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    issues: List[Dict[str, Any]] = []
    score = 100.0

    if not transcript.strip():
        return 0.0, ["no transcript detected"], [
            {
                "id": "grammar_no_transcript",
                "category": "grammar",
                "severity": 1.0,
                "message": "No transcript detected for grammar scoring.",
                "evidence": {},
                "suggestion_hint": "Capture a clearer recording and retry.",
            }
        ]

    # Proxy checks only; replace with LLM/parser later for deeper accuracy.
    sentence_end_ok = transcript.strip().endswith((".", "?", "!", "。", "？", "！"))
    if not sentence_end_ok:
        score -= 10
        reasons.append("sentence end punctuation is missing")
        issues.append(
            {
                "id": "grammar_sentence_boundary",
                "category": "grammar",
                "severity": 0.25,
                "message": "Sentence boundary markers are unclear.",
                "evidence": {"sentence_end_punctuation": False},
                "suggestion_hint": "Practice producing complete sentence endings.",
            }
        )

    repeated_token_count = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            repeated_token_count += 1
    if repeated_token_count > 0:
        penalty = min(25.0, repeated_token_count * 8.0)
        score -= penalty
        reasons.append(f"found {repeated_token_count} repeated adjacent token(s)")
        issues.append(
            {
                "id": "grammar_repetition",
                "category": "grammar",
                "severity": round(min(1.0, repeated_token_count / 3), 3),
                "message": "Repeated adjacent words suggest disfluency or correction loops.",
                "evidence": {"repeated_adjacent_tokens": repeated_token_count},
                "suggestion_hint": "Use shorter clauses and pause at phrase boundaries instead of repeating words.",
            }
        )

    if len(tokens) < 3:
        score -= 18
        reasons.append("answer is very short for grammar assessment")
        issues.append(
            {
                "id": "grammar_too_short",
                "category": "grammar",
                "severity": 0.5,
                "message": "Insufficient length to demonstrate grammar control.",
                "evidence": {"token_count": len(tokens)},
                "suggestion_hint": "Aim for at least one full sentence with a subject and predicate.",
            }
        )

    if not reasons:
        reasons.append("basic sentence form looked consistent")
    return _clamp(score), reasons, issues


def _score_vocabulary(tokens: List[str]) -> Tuple[float, List[str], Dict[str, float], List[Dict[str, Any]]]:
    reasons: List[str] = []
    issues: List[Dict[str, Any]] = []
    score = 100.0
    metrics: Dict[str, float] = {}

    if not tokens:
        return 0.0, ["no words detected"], {"lexical_diversity": 0.0}, [
            {
                "id": "vocab_no_words",
                "category": "vocabulary",
                "severity": 1.0,
                "message": "No words detected for vocabulary scoring.",
                "evidence": {},
                "suggestion_hint": "Ensure microphone input is captured correctly.",
            }
        ]

    unique_count = len(set(tokens))
    token_count = len(tokens)
    lexical_diversity = unique_count / token_count
    metrics["lexical_diversity"] = round(lexical_diversity, 4)

    # Range proxy.
    if lexical_diversity < 0.45:
        score -= min(35.0, (0.45 - lexical_diversity) * 120)
        reasons.append(
            f"lexical diversity was {lexical_diversity:.2f} (target >= 0.45 for simple answers)"
        )
        issues.append(
            {
                "id": "vocab_range_low",
                "category": "vocabulary",
                "severity": round(min(1.0, (0.45 - lexical_diversity) / 0.30), 3),
                "message": "Vocabulary range is limited for this response.",
                "evidence": {"actual": round(lexical_diversity, 4), "target_min": 0.45},
                "suggestion_hint": "Introduce one or two precise content words related to the prompt topic.",
            }
        )
    elif lexical_diversity > 0.9 and token_count < 8:
        # Very short responses can look diverse but still weak.
        score -= 8
        reasons.append("vocabulary range looked inflated by short answer length")

    avg_token_len = sum(len(t) for t in tokens) / token_count
    metrics["avg_token_length"] = round(avg_token_len, 2)
    if avg_token_len < 3 and token_count >= 8:
        score -= 8
        reasons.append("word choices were mostly very short/simple")
        issues.append(
            {
                "id": "vocab_word_complexity_low",
                "category": "vocabulary",
                "severity": 0.25,
                "message": "Word complexity is low for the response length.",
                "evidence": {"avg_token_length": round(avg_token_len, 2)},
                "suggestion_hint": "Use at least one descriptive adjective or specific noun.",
            }
        )

    if not reasons:
        reasons.append("word variety was appropriate for the response length")
    return _clamp(score), reasons, metrics, issues


def _score_coherence(
    transcript: str,
    tokens: List[str],
    linking_words: set[str],
    prompt: str | None = None,
    prompt_tokens: List[str] | None = None,
) -> Tuple[float, List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    issues: List[Dict[str, Any]] = []
    score = 100.0

    if not transcript.strip():
        return 0.0, ["no transcript detected"], [
            {
                "id": "coherence_no_transcript",
                "category": "coherence",
                "severity": 1.0,
                "message": "No transcript detected for coherence scoring.",
                "evidence": {},
                "suggestion_hint": "Capture a clearer recording and retry.",
            }
        ]

    # Length proxy for completeness.
    if len(tokens) < 4:
        score -= 25
        reasons.append("answer length is too short to fully develop an idea")
        issues.append(
            {
                "id": "coherence_too_short",
                "category": "coherence",
                "severity": 0.6,
                "message": "Answer is too short to show idea development.",
                "evidence": {"token_count": len(tokens)},
                "suggestion_hint": "Respond with at least two connected clauses.",
            }
        )

    # Basic connector usage proxy for logic flow.
    connector_hits = sum(1 for t in tokens if t in linking_words)
    if len(tokens) >= 10 and connector_hits == 0:
        score -= 10
        reasons.append("few linking words were used to connect ideas")
        issues.append(
            {
                "id": "coherence_linkers_missing",
                "category": "coherence",
                "severity": 0.3,
                "message": "Few explicit connectors were detected.",
                "evidence": {"connector_hits": connector_hits},
                "suggestion_hint": "Use a connector such as 'because/therefore/however' to join ideas.",
            }
        )

    # Prompt overlap proxy for relevance.
    if prompt:
        prompt_token_set = set(prompt_tokens or [])
        if prompt_token_set:
            overlap = len(set(tokens) & prompt_token_set) / max(1, len(prompt_token_set))
            if overlap < 0.05:
                score -= 20
                reasons.append("low overlap with prompt keywords suggests weak relevance")
                issues.append(
                    {
                        "id": "coherence_prompt_relevance_low",
                        "category": "coherence",
                        "severity": round(min(1.0, (0.05 - overlap) / 0.05), 3),
                        "message": "Response appears weakly tied to the prompt.",
                        "evidence": {"keyword_overlap_ratio": round(overlap, 4), "target_min": 0.05},
                        "suggestion_hint": "Reuse 1-2 prompt keywords in your answer and expand on them.",
                    }
                )

    if not reasons:
        reasons.append("answer stayed on-topic and sufficiently developed")
    return _clamp(score), reasons, issues


def _score_clarity_proxy(filler_ratio: float, pause_ratio: float) -> Tuple[float, List[str], List[Dict[str, Any]]]:
    reasons: List[str] = []
    issues: List[Dict[str, Any]] = []
    score = 100.0

    if filler_ratio > 0.05:
        score -= min(55.0, (filler_ratio - 0.05) * 300)
        reasons.append(f"filler ratio was {filler_ratio * 100:.1f}% (target < 5%)")
        issues.append(
            {
                "id": "clarity_filler_high",
                "category": "clarity_proxy",
                "severity": round(min(1.0, (filler_ratio - 0.05) / 0.10), 3),
                "message": "Frequent fillers reduce perceived clarity/confidence.",
                "evidence": {"actual": round(filler_ratio, 4), "target_max": 0.05},
                "suggestion_hint": "Pause silently for a brief moment instead of using fillers.",
            }
        )

    if pause_ratio > 0.2:
        score -= min(45.0, (pause_ratio - 0.2) * 180)
        reasons.append(f"long pauses reduced clarity proxy (pause ratio {pause_ratio * 100:.1f}%)")
        issues.append(
            {
                "id": "clarity_pause_high",
                "category": "clarity_proxy",
                "severity": round(min(1.0, (pause_ratio - 0.2) / 0.2), 3),
                "message": "Long pauses reduce clarity/confidence signal.",
                "evidence": {"actual": round(pause_ratio, 4), "target_max": 0.2},
                "suggestion_hint": "Chunk your answer into two short planned phrases.",
            }
        )

    if not reasons:
        reasons.append("delivery sounded consistently clear by pause/filler proxies")
    return _clamp(score), reasons, issues


def _build_suggestion_generator_input(
    *,
    language: str,
    prompt: str | None,
    transcript: str,
    overall_score: float,
    subscores: Dict[str, float],
    category_feedback: Dict[str, List[str]],
    issues: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    ordered_issues = sorted(issues, key=lambda item: item.get("severity", 0.0), reverse=True)
    top_issues = ordered_issues[:5]
    strengths: List[Dict[str, Any]] = []
    for cat, score in subscores.items():
        if score >= 85:
            strengths.append(
                {
                    "category": cat,
                    "score": round(score, 2),
                    "signal": (category_feedback.get(cat) or ["strong performance"])[0],
                }
            )

    return {
        "version": "v1",
        "task": "generate_learner_feedback_suggestions",
        "language": language,
        "prompt": prompt,
        "learner_transcript": transcript,
        "overall_score": round(overall_score, 2),
        "subscores": {k: round(v, 2) for k, v in subscores.items()},
        "metrics": metrics,
        "top_issues": top_issues,
        "strengths": strengths[:3],
        "generator_guidance": {
            "tone": "encouraging and specific",
            "max_suggestions": 3,
            "must_reference_evidence": True,
            "prioritize_categories": ["fluency", "grammar", "vocabulary", "coherence", "clarity_proxy"],
        },
        "suggestion_candidates": [
            {"focus_issue_id": issue["id"], "hint": issue["suggestion_hint"]}
            for issue in top_issues
        ],
    }


def _evaluate_pronunciation_deterministic(
    input_payload: Dict[str, Any],
    *,
    prompt: str | None = None,
    language: str | None = None,
    weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Deterministic, explainable scoring for hackathon prototyping.

    Expected input_payload:
    {
      "transcript": "text user said",
      "pauses": [{"start": 1.2, "end": 1.7}, ...]  # or equivalent dict/list format
      "total_duration": 12.4,
      "language": "en"  # optional, can also be passed as function argument
    }
    """
    weights = weights or DEFAULT_WEIGHTS
    resolved_language = (language or input_payload.get("language") or "en").lower()
    profile = _get_language_profile(resolved_language)
    token_mode = profile["token_mode"]
    wpm_target_min = float(profile["wpm_target_min"])
    wpm_target_max = float(profile["wpm_target_max"])
    filler_words = profile["filler_words"]
    linking_words = profile["linking_words"]

    transcript = str(input_payload.get("transcript", "")).strip()
    total_duration = float(input_payload.get("total_duration", 0.0) or 0.0)
    pause_spans = _extract_pause_spans(input_payload.get("pauses", []))

    tokens = _tokenize(transcript, token_mode=token_mode)
    num_words = len(tokens)

    total_pause_seconds = sum(end - start for start, end in pause_spans)
    total_pause_seconds = min(total_pause_seconds, max(total_duration, 0.0))
    duration_minutes = max(total_duration / 60.0, 1e-9)

    wpm = num_words / duration_minutes if total_duration > 0 else 0.0
    pause_ratio = total_pause_seconds / total_duration if total_duration > 0 else 0.0

    filler_count = 0
    lowered_transcript = transcript.lower()
    for filler in filler_words:
        if " " in filler:
            filler_count += lowered_transcript.count(filler)
        else:
            filler_count += tokens.count(filler)
    filler_ratio = filler_count / max(num_words, 1)

    prompt_tokens = _tokenize(prompt or "", token_mode=token_mode) if prompt else []

    fluency_score, fluency_reasons, fluency_issues = _score_fluency(
        wpm, pause_ratio, filler_ratio, wpm_target_min, wpm_target_max
    )
    grammar_score, grammar_reasons, grammar_issues = _score_grammar(transcript, tokens)
    vocabulary_score, vocabulary_reasons, vocab_metrics, vocab_issues = _score_vocabulary(tokens)
    coherence_score, coherence_reasons, coherence_issues = _score_coherence(
        transcript,
        tokens,
        linking_words=linking_words,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
    )
    clarity_score, clarity_reasons, clarity_issues = _score_clarity_proxy(filler_ratio, pause_ratio)

    weighted = {
        "fluency": fluency_score * weights["fluency"],
        "grammar": grammar_score * weights["grammar"],
        "vocabulary": vocabulary_score * weights["vocabulary"],
        "coherence": coherence_score * weights["coherence"],
        "clarity_proxy": clarity_score * weights["clarity_proxy"],
    }
    overall_score = _clamp(sum(weighted.values()))

    category_feedback = {
        "fluency": fluency_reasons,
        "grammar": grammar_reasons,
        "vocabulary": vocabulary_reasons,
        "coherence": coherence_reasons,
        "clarity_proxy": clarity_reasons,
    }

    summary_feedback = []
    for category, reasons in category_feedback.items():
        if reasons:
            summary_feedback.append(f"{category}: {reasons[0]}")

    all_issues = fluency_issues + grammar_issues + vocab_issues + coherence_issues + clarity_issues
    prioritized_issues = sorted(all_issues, key=lambda item: item.get("severity", 0.0), reverse=True)

    subscores = {
        "fluency": round(fluency_score, 2),
        "grammar": round(grammar_score, 2),
        "vocabulary": round(vocabulary_score, 2),
        "coherence": round(coherence_score, 2),
        "clarity_proxy": round(clarity_score, 2),
    }
    metrics = {
        "num_words": num_words,
        "total_duration_seconds": round(total_duration, 3),
        "wpm": round(wpm, 2),
        "wpm_target_min": wpm_target_min,
        "wpm_target_max": wpm_target_max,
        "pause_count": len(pause_spans),
        "total_pause_seconds": round(total_pause_seconds, 3),
        "pause_ratio": round(pause_ratio, 4),
        "filler_count": filler_count,
        "filler_ratio": round(filler_ratio, 4),
        **vocab_metrics,
    }

    return {
        "overall_score": round(overall_score, 2),
        "language": resolved_language,
        "subscores": subscores,
        "weights": weights,
        "weighted_breakdown": {k: round(v, 2) for k, v in weighted.items()},
        "metrics": metrics,
        "category_feedback": category_feedback,
        "feedback_summary": summary_feedback,
        "issues": prioritized_issues,
        "suggestion_generator_input": _build_suggestion_generator_input(
            language=resolved_language,
            prompt=prompt,
            transcript=transcript,
            overall_score=overall_score,
            subscores=subscores,
            category_feedback=category_feedback,
            issues=prioritized_issues,
            metrics=metrics,
        ),
    }


SCORE_CATEGORIES = ("fluency", "grammar", "vocabulary", "coherence", "clarity_proxy")


def _ollama_is_reachable() -> bool:
    host_raw = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    parsed = urlparse(host_raw if "://" in host_raw else f"http://{host_raw}")
    hostname = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    try:
        with socket.create_connection((hostname, port), timeout=0.25):
            return True
    except OSError:
        return False


def _parse_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _coerce_llm_report(
    llm_data: Dict[str, Any],
    base_report: Dict[str, Any],
) -> Dict[str, Any]:
    weights = base_report["weights"]
    subscores = dict(base_report["subscores"])

    raw_subscores = llm_data.get("subscores", {})
    if isinstance(raw_subscores, dict):
        for cat in SCORE_CATEGORIES:
            val = raw_subscores.get(cat)
            if isinstance(val, (int, float)):
                subscores[cat] = round(_clamp(float(val)), 2)

    overall = llm_data.get("overall_score")
    if not isinstance(overall, (int, float)):
        overall = sum(subscores[cat] * weights[cat] for cat in SCORE_CATEGORIES)
    overall = round(_clamp(float(overall)), 2)

    category_feedback = dict(base_report.get("category_feedback", {}))
    raw_feedback = llm_data.get("category_feedback", {})
    if isinstance(raw_feedback, dict):
        for cat in SCORE_CATEGORIES:
            val = raw_feedback.get(cat)
            if isinstance(val, list):
                cleaned = [str(item).strip() for item in val if str(item).strip()]
                if cleaned:
                    category_feedback[cat] = cleaned

    issues = base_report.get("issues", [])
    raw_issues = llm_data.get("issues", [])
    if isinstance(raw_issues, list):
        cleaned_issues: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_issues[:10], start=1):
            if not isinstance(item, dict):
                continue
            category = str(item.get("category", "fluency")).strip().lower()
            if category not in SCORE_CATEGORIES:
                category = "fluency"
            severity_raw = item.get("severity", 0.3)
            severity = float(severity_raw) if isinstance(severity_raw, (int, float)) else 0.3
            cleaned_issues.append(
                {
                    "id": str(item.get("id", f"llm_issue_{idx}")).strip() or f"llm_issue_{idx}",
                    "category": category,
                    "severity": round(max(0.0, min(1.0, severity)), 3),
                    "message": str(item.get("message", "Improvement needed.")).strip() or "Improvement needed.",
                    "evidence": item.get("evidence", {}) if isinstance(item.get("evidence"), dict) else {},
                    "suggestion_hint": (
                        str(item.get("suggestion_hint", "Practice this category with targeted drills.")).strip()
                        or "Practice this category with targeted drills."
                    ),
                }
            )
        if cleaned_issues:
            issues = sorted(cleaned_issues, key=lambda x: x["severity"], reverse=True)

    summary = llm_data.get("feedback_summary")
    if isinstance(summary, list):
        feedback_summary = [str(x).strip() for x in summary if str(x).strip()]
    else:
        feedback_summary = []
    if not feedback_summary:
        feedback_summary = [
            f"{cat}: {(category_feedback.get(cat) or ['no additional feedback'])[0]}"
            for cat in SCORE_CATEGORIES
        ]

    weighted_breakdown = {
        cat: round(subscores[cat] * weights[cat], 2) for cat in SCORE_CATEGORIES
    }

    return {
        "overall_score": overall,
        "language": base_report["language"],
        "subscores": subscores,
        "weights": weights,
        "weighted_breakdown": weighted_breakdown,
        "metrics": base_report["metrics"],
        "category_feedback": category_feedback,
        "feedback_summary": feedback_summary,
        "issues": issues,
        "suggestion_generator_input": _build_suggestion_generator_input(
            language=base_report["language"],
            prompt=base_report["suggestion_generator_input"]["prompt"],
            transcript=base_report["suggestion_generator_input"]["learner_transcript"],
            overall_score=overall,
            subscores=subscores,
            category_feedback=category_feedback,
            issues=issues,
            metrics=base_report["metrics"],
        ),
    }


def _evaluate_with_ollama(
    *,
    input_payload: Dict[str, Any],
    prompt: str | None,
    base_report: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    if ollama is None:
        raise RuntimeError("ollama package is not available")

    scorer_prompt = f"""
You are an expert spoken-language scorer.
Evaluate the learner response and return only valid JSON.

Return schema:
{{
  "overall_score": number 0-100,
  "subscores": {{
    "fluency": number 0-100,
    "grammar": number 0-100,
    "vocabulary": number 0-100,
    "coherence": number 0-100,
    "clarity_proxy": number 0-100
  }},
  "category_feedback": {{
    "fluency": ["short reason"],
    "grammar": ["short reason"],
    "vocabulary": ["short reason"],
    "coherence": ["short reason"],
    "clarity_proxy": ["short reason"]
  }},
  "feedback_summary": ["up to 5 bullets"],
  "issues": [
    {{
      "id": "string",
      "category": "fluency|grammar|vocabulary|coherence|clarity_proxy",
      "severity": number between 0 and 1,
      "message": "short issue statement",
      "evidence": {{}},
      "suggestion_hint": "specific action"
    }}
  ]
}}

Context:
- target_language: {base_report["language"]}
- prompt: {prompt or ""}
- learner_input: {json.dumps(input_payload, ensure_ascii=False)}
- baseline_report_for_reference: {json.dumps(base_report, ensure_ascii=False)}

Requirements:
- Use the baseline only as reference; provide your own evaluation.
- Keep issues evidence-grounded to transcript/metrics.
- Return JSON only.
""".strip()

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": scorer_prompt}],
        format="json",
        options={"temperature": 0.1},
    )
    content = str(response.get("message", {}).get("content", ""))
    parsed = _parse_json_object(content)
    if not parsed:
        raise RuntimeError("Failed to parse scorer JSON from Ollama output")
    return _coerce_llm_report(parsed, base_report)


def evaluate_pronunciation(
    input_payload: Dict[str, Any],
    *,
    prompt: str | None = None,
    language: str | None = None,
    weights: Dict[str, float] | None = None,
    use_ollama: bool = True,
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Scoring entry point.
    1) Build deterministic baseline metrics/report
    2) Ask Ollama model to evaluate and return scoring report
    3) Fall back to deterministic report if Ollama is unavailable/unparseable
    """
    base_report = _evaluate_pronunciation_deterministic(
        input_payload,
        prompt=prompt,
        language=language,
        weights=weights,
    )

    if not use_ollama:
        base_report["metrics"]["scorer"] = "deterministic"
        return base_report

    selected_model = model or os.getenv("OLLAMA_SCORER_MODEL", "gemma2:2b")
    if not _ollama_is_reachable():
        base_report["metrics"]["scorer"] = "deterministic_fallback"
        base_report["metrics"]["scorer_model"] = selected_model
        base_report["metrics"]["scorer_error"] = "Ollama server is not reachable"
        return base_report

    try:
        llm_report = _evaluate_with_ollama(
            input_payload=input_payload,
            prompt=prompt,
            base_report=base_report,
            model=selected_model,
        )
        llm_report["metrics"]["scorer"] = "ollama"
        llm_report["metrics"]["scorer_model"] = selected_model
        return llm_report
    except Exception as exc:
        base_report["metrics"]["scorer"] = "deterministic_fallback"
        base_report["metrics"]["scorer_model"] = selected_model
        base_report["metrics"]["scorer_error"] = str(exc)
        return base_report


def evaluate_pronunciation_llm(transcribe_output: Dict[str, Any]) -> Dict[str, Any]:
    return evaluate_pronunciation(transcribe_output, use_ollama=True)
