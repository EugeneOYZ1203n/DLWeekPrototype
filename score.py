from __future__ import annotations

from typing import Any, Dict

from suggestion import SuggestionGenerator


def evaluate_pronunciation_llm(transcribe_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Placeholder scoring step that normalizes values from the transcription step.
    Replace this with your full scoring logic/LLM scorer when ready.
    """
    transcript = transcribe_output.get("transcript", "")
    pause_count = int(transcribe_output.get("pause_count", 0) or 0)
    hesitation_count = int(transcribe_output.get("hesitation_count", 0) or 0)

    fluency_penalty = min(pause_count * 3 + hesitation_count * 4, 60)
    fluency_score = max(40, 100 - fluency_penalty)

    score_report = {
        "overall_score": fluency_score,
        "relevance_score": 70,
        "vocabulary_score": 70,
        "grammar_score": 70,
        "pronunciation_score": 70,
        "fluency_score": fluency_score,
        "feedback": f'Transcript captured: "{transcript}"',
    }
    return score_report


def evaluate_pronunciation(transcribe_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point used by the app:
    1) Build score report from prior step
    2) Generate speaking suggestions with Ollama using transcript + nuance metrics
    """
    score_report = evaluate_pronunciation_llm(transcribe_output)

    speech_metrics = {
        "pause_count": transcribe_output.get("pause_count", 0),
        "avg_pause_ms": transcribe_output.get("avg_pause_ms", 0),
        "hesitation_count": transcribe_output.get("hesitation_count", 0),
        "tone_consistency": transcribe_output.get("tone_consistency", 1.0),
        "clip_duration_sec": transcribe_output.get("duration_sec", 0),
    }

    generator = SuggestionGenerator()
    suggestions = generator.generate(
        transcript=transcribe_output.get("transcript", ""),
        score_report=score_report,
        speech_metrics=speech_metrics,
        target_sentence=transcribe_output.get("target_sentence", ""),
        language=transcribe_output.get("language_name", "Japanese"),
    )

    return {
        "score_report": score_report,
        "speaking_suggestions": suggestions,
    }
