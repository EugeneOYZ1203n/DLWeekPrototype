from __future__ import annotations

import json
import re
from typing import Any, Dict, List

try:
    import ollama
except Exception:  # pragma: no cover - optional dependency in some environments
    ollama = None


class SuggestionGenerator:
    """
    Generates speaking improvement suggestions from transcript, score metrics,
    and speech nuances (pauses, hesitation, tone) using a local Ollama model.
    """

    def __init__(
        self,
        model: str = "gemma2:2b",
        timeout_seconds: int = 60,
    ) -> None:
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        transcript: str,
        score_report: Dict[str, Any],
        speech_metrics: Dict[str, Any],
        target_sentence: str = "",
        language: str = "Japanese",
    ) -> Dict[str, Any]:
        prompt = self._build_prompt(
            transcript=transcript,
            score_report=score_report,
            speech_metrics=speech_metrics,
            target_sentence=target_sentence,
            language=language,
        )

        try:
            raw = self._call_ollama(prompt)
            parsed = self._parse_json(raw)
            if parsed:
                return parsed
        except Exception:
            pass

        return self._fallback(score_report, speech_metrics)

    def _build_prompt(
        self,
        transcript: str,
        score_report: Dict[str, Any],
        speech_metrics: Dict[str, Any],
        target_sentence: str,
        language: str,
    ) -> str:
        return f"""
You are a strict but helpful {language} speaking coach.
Use the transcript, score metrics, and speech nuance metrics to generate actionable speaking suggestions.

Target sentence (if provided): {target_sentence}
Student transcript: {transcript}
Score report: {json.dumps(score_report, ensure_ascii=False)}
Speech nuance metrics: {json.dumps(speech_metrics, ensure_ascii=False)}

Focus on:
1) pause patterns and gap timing
2) hesitation markers (fillers, restarts, unfinished phrases)
3) tone/intonation consistency
4) pronunciation clarity
5) fluency and pacing

Return only valid JSON with this schema:
{{
  "overall_assessment": "short string",
  "suggestions": [
    {{
      "category": "pauses|hesitation|tone|pronunciation|fluency|grammar|vocabulary",
      "issue": "short string",
      "action": "specific actionable coaching step",
      "practice_example": "short drill or sentence practice"
    }}
  ],
  "next_practice_focus": ["item1", "item2", "item3"]
}}
""".strip()

    def _call_ollama(self, prompt: str) -> str:
        if ollama is None:
            raise RuntimeError("ollama package is not available")
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.2},
        )
        message = response.get("message", {})
        return message.get("content", "")

    def _parse_json(self, text: str) -> Dict[str, Any]:
        if not text:
            return {}

        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}

        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return {}

        return {}

    def _fallback(
        self, score_report: Dict[str, Any], speech_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        suggestions: List[Dict[str, str]] = []

        pause_count = int(speech_metrics.get("pause_count", 0) or 0)
        avg_pause_ms = float(speech_metrics.get("avg_pause_ms", 0.0) or 0.0)
        hesitation_count = int(speech_metrics.get("hesitation_count", 0) or 0)
        tone_consistency = float(speech_metrics.get("tone_consistency", 1.0) or 1.0)
        raw_fluency = score_report.get("fluency_score")
        if raw_fluency is None and isinstance(score_report.get("subscores"), dict):
            raw_fluency = score_report["subscores"].get("fluency")
        fluency_score = int(raw_fluency or 0)

        if pause_count > 6 or avg_pause_ms > 900:
            suggestions.append(
                {
                    "category": "pauses",
                    "issue": "Frequent or long pauses break sentence flow.",
                    "action": "Chunk the sentence into 2-3 phrase groups and speak each group in one breath.",
                    "practice_example": "Read once slowly, then repeat at natural speed without stopping mid-phrase.",
                }
            )

        if hesitation_count > 2:
            suggestions.append(
                {
                    "category": "hesitation",
                    "issue": "Hesitation markers suggest low automatic recall.",
                    "action": "Do 30-second shadowing loops with the target sentence until smooth.",
                    "practice_example": "Repeat the same line 5 times while reducing fillers each repetition.",
                }
            )

        if tone_consistency < 0.5:
            suggestions.append(
                {
                    "category": "tone",
                    "issue": "Pitch movement is inconsistent across the sentence.",
                    "action": "Practice with a model recording and mirror intonation contour phrase by phrase.",
                    "practice_example": "Record two versions and compare pitch rise/fall against the model.",
                }
            )

        if fluency_score < 60:
            suggestions.append(
                {
                    "category": "fluency",
                    "issue": "Delivery speed and rhythm are not yet stable.",
                    "action": "Use a metronome pace and increase speed in 10% increments when accuracy holds.",
                    "practice_example": "Speak at 70%, 80%, 90%, then full speed with no extra pauses.",
                }
            )

        if not suggestions:
            suggestions.append(
                {
                    "category": "fluency",
                    "issue": "No major speech-pattern issues detected from current metrics.",
                    "action": "Continue daily short speaking drills and increase sentence complexity.",
                    "practice_example": "Add one new sentence each day and keep pronunciation consistent.",
                }
            )

        return {
            "overall_assessment": "Fallback coaching generated locally because model output was unavailable.",
            "suggestions": suggestions,
            "next_practice_focus": [s["category"] for s in suggestions[:3]],
        }
