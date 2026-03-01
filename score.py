from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

# Default category weights (sum should be 1.0)
DEFAULT_WEIGHTS = {
    "fluency": 0.30,
    "grammar": 0.25,
    "vocabulary": 0.20,
    "coherence": 0.20,
    "clarity_proxy": 0.05,
}

# Filler words for confidence proxy.
FILLER_WORDS = {
    "um",
    "uh",
    "erm",
    "hmm",
    "like",
    "you know",
    "sort of",
    "kind of",
}

# Simple coherence anchors for short-answer prompts.
LINKING_WORDS = {
    "because",
    "so",
    "therefore",
    "however",
    "first",
    "then",
    "finally",
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


def _tokenize(transcript: str) -> List[str]:
    """
    Tokenizer that works for Latin words and also captures CJK runs.
    For CJK runs, split to characters as a rough "word-like" unit.
    """
    rough_tokens = re.findall(r"[A-Za-z']+|[\u3040-\u30ff\u3400-\u9fff]+|\d+", transcript.lower())
    tokens: List[str] = []
    for token in rough_tokens:
        if re.fullmatch(r"[\u3040-\u30ff\u3400-\u9fff]+", token):
            tokens.extend(list(token))
        else:
            tokens.append(token)
    return tokens


def _score_fluency(wpm: float, pause_ratio: float, filler_ratio: float) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 100.0

    # Target zone 90-150 WPM for simple conversation.
    if wpm < 90:
        penalty = min(35.0, (90 - wpm) * 0.45)
        score -= penalty
        reasons.append(f"wpm was {wpm:.1f} (target 90-150)")
    elif wpm > 150:
        penalty = min(20.0, (wpm - 150) * 0.25)
        score -= penalty
        reasons.append(f"wpm was {wpm:.1f} (target 90-150)")

    # Pause ratio target under 12%.
    if pause_ratio > 0.12:
        penalty = min(45.0, (pause_ratio - 0.12) * 220)
        score -= penalty
        reasons.append(f"pause ratio was {pause_ratio * 100:.1f}% (target < 12%)")

    # Filler ratio target under 3%.
    if filler_ratio > 0.03:
        penalty = min(20.0, (filler_ratio - 0.03) * 200)
        score -= penalty
        reasons.append(f"filler ratio was {filler_ratio * 100:.1f}% (target < 3%)")

    if not reasons:
        reasons.append("speech pace and pause control were in target range")
    return _clamp(score), reasons


def _score_grammar(transcript: str, tokens: List[str]) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 100.0

    if not transcript.strip():
        return 0.0, ["no transcript detected"]

    # Proxy checks only; replace with LLM/parser later for deeper accuracy.
    sentence_end_ok = transcript.strip().endswith((".", "?", "!", "。", "？", "！"))
    if not sentence_end_ok:
        score -= 10
        reasons.append("sentence end punctuation is missing")

    repeated_token_count = 0
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            repeated_token_count += 1
    if repeated_token_count > 0:
        penalty = min(25.0, repeated_token_count * 8.0)
        score -= penalty
        reasons.append(f"found {repeated_token_count} repeated adjacent token(s)")

    if len(tokens) < 3:
        score -= 18
        reasons.append("answer is very short for grammar assessment")

    if not reasons:
        reasons.append("basic sentence form looked consistent")
    return _clamp(score), reasons


def _score_vocabulary(tokens: List[str]) -> Tuple[float, List[str], Dict[str, float]]:
    reasons: List[str] = []
    score = 100.0
    metrics: Dict[str, float] = {}

    if not tokens:
        return 0.0, ["no words detected"], {"lexical_diversity": 0.0}

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
    elif lexical_diversity > 0.9 and token_count < 8:
        # Very short responses can look diverse but still weak.
        score -= 8
        reasons.append("vocabulary range looked inflated by short answer length")

    avg_token_len = sum(len(t) for t in tokens) / token_count
    metrics["avg_token_length"] = round(avg_token_len, 2)
    if avg_token_len < 3 and token_count >= 8:
        score -= 8
        reasons.append("word choices were mostly very short/simple")

    if not reasons:
        reasons.append("word variety was appropriate for the response length")
    return _clamp(score), reasons, metrics


def _score_coherence(transcript: str, tokens: List[str], prompt: str | None = None) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 100.0

    if not transcript.strip():
        return 0.0, ["no transcript detected"]

    # Length proxy for completeness.
    if len(tokens) < 4:
        score -= 25
        reasons.append("answer length is too short to fully develop an idea")

    # Basic connector usage proxy for logic flow.
    connector_hits = sum(1 for t in tokens if t in LINKING_WORDS)
    if len(tokens) >= 10 and connector_hits == 0:
        score -= 10
        reasons.append("few linking words were used to connect ideas")

    # Prompt overlap proxy for relevance.
    if prompt:
        prompt_tokens = set(_tokenize(prompt))
        if prompt_tokens:
            overlap = len(set(tokens) & prompt_tokens) / max(1, len(prompt_tokens))
            if overlap < 0.05:
                score -= 20
                reasons.append("low overlap with prompt keywords suggests weak relevance")

    if not reasons:
        reasons.append("answer stayed on-topic and sufficiently developed")
    return _clamp(score), reasons


def _score_clarity_proxy(filler_ratio: float, pause_ratio: float) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    score = 100.0

    if filler_ratio > 0.05:
        score -= min(55.0, (filler_ratio - 0.05) * 300)
        reasons.append(f"filler ratio was {filler_ratio * 100:.1f}% (target < 5%)")

    if pause_ratio > 0.2:
        score -= min(45.0, (pause_ratio - 0.2) * 180)
        reasons.append(f"long pauses reduced clarity proxy (pause ratio {pause_ratio * 100:.1f}%)")

    if not reasons:
        reasons.append("delivery sounded consistently clear by pause/filler proxies")
    return _clamp(score), reasons


def evaluate_pronunciation(
    input_payload: Dict[str, Any],
    *,
    prompt: str | None = None,
    weights: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """
    Deterministic, explainable scoring for hackathon prototyping.

    Expected input_payload:
    {
      "transcript": "text user said",
      "pauses": [{"start": 1.2, "end": 1.7}, ...]  # or equivalent dict/list format
      "total_duration": 12.4
    }
    """
    weights = weights or DEFAULT_WEIGHTS
    transcript = str(input_payload.get("transcript", "")).strip()
    total_duration = float(input_payload.get("total_duration", 0.0) or 0.0)
    pause_spans = _extract_pause_spans(input_payload.get("pauses", []))

    tokens = _tokenize(transcript)
    num_words = len(tokens)

    total_pause_seconds = sum(end - start for start, end in pause_spans)
    total_pause_seconds = min(total_pause_seconds, max(total_duration, 0.0))
    duration_minutes = max(total_duration / 60.0, 1e-9)

    wpm = num_words / duration_minutes if total_duration > 0 else 0.0
    pause_ratio = total_pause_seconds / total_duration if total_duration > 0 else 0.0

    filler_count = 0
    lowered_transcript = transcript.lower()
    for filler in FILLER_WORDS:
        if " " in filler:
            filler_count += lowered_transcript.count(filler)
        else:
            filler_count += tokens.count(filler)
    filler_ratio = filler_count / max(num_words, 1)

    fluency_score, fluency_reasons = _score_fluency(wpm, pause_ratio, filler_ratio)
    grammar_score, grammar_reasons = _score_grammar(transcript, tokens)
    vocabulary_score, vocabulary_reasons, vocab_metrics = _score_vocabulary(tokens)
    coherence_score, coherence_reasons = _score_coherence(transcript, tokens, prompt=prompt)
    clarity_score, clarity_reasons = _score_clarity_proxy(filler_ratio, pause_ratio)

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

    return {
        "overall_score": round(overall_score, 2),
        "subscores": {
            "fluency": round(fluency_score, 2),
            "grammar": round(grammar_score, 2),
            "vocabulary": round(vocabulary_score, 2),
            "coherence": round(coherence_score, 2),
            "clarity_proxy": round(clarity_score, 2),
        },
        "weights": weights,
        "weighted_breakdown": {k: round(v, 2) for k, v in weighted.items()},
        "metrics": {
            "num_words": num_words,
            "total_duration_seconds": round(total_duration, 3),
            "wpm": round(wpm, 2),
            "pause_count": len(pause_spans),
            "total_pause_seconds": round(total_pause_seconds, 3),
            "pause_ratio": round(pause_ratio, 4),
            "filler_count": filler_count,
            "filler_ratio": round(filler_ratio, 4),
            **vocab_metrics,
        },
        "category_feedback": category_feedback,
        "feedback_summary": summary_feedback,
    }


def evaluate_pronunciation_llm(transcribe_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compatibility wrapper.
    Keep this function name so callers don't break while using deterministic scoring.
    """
    return evaluate_pronunciation(transcribe_output)
