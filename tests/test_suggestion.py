import unittest
from unittest.mock import patch

from suggestion import SuggestionGenerator


class TestSuggestionGenerator(unittest.TestCase):
    def test_fallback_reads_fluency_from_subscores(self):
        generator = SuggestionGenerator()
        score_report = {
            "subscores": {"fluency": 42.0},
            "metrics": {"pause_count": 0, "total_pause_seconds": 0.0, "filler_count": 0},
        }
        speech_metrics = {
            "pause_count": 1,
            "avg_pause_ms": 400.0,
            "hesitation_count": 0,
            "tone_consistency": 0.9,
        }

        report = generator._fallback(score_report, speech_metrics)
        categories = [item["category"] for item in report["suggestions"]]
        self.assertIn("fluency", categories)

    @patch.object(SuggestionGenerator, "_call_ollama", side_effect=RuntimeError("offline"))
    def test_generate_falls_back_when_model_call_fails(self, _mock_call):
        generator = SuggestionGenerator()
        score_report = {"subscores": {"fluency": 55.0}, "metrics": {"filler_count": 3}}
        speech_metrics = {"pause_count": 3, "avg_pause_ms": 900.0, "hesitation_count": 3, "tone_consistency": 0.4}

        report = generator.generate(
            transcript="sample transcript",
            score_report=score_report,
            speech_metrics=speech_metrics,
            target_sentence="サンプル文です。",
            language="Japanese",
        )

        self.assertIn("overall_assessment", report)
        self.assertIn("suggestions", report)
        self.assertGreater(len(report["suggestions"]), 0)


if __name__ == "__main__":
    unittest.main()
