import unittest
from unittest.mock import patch

from score import evaluate_pronunciation


class TestScoreEngineLLM(unittest.TestCase):
    def _payload(self, language="en"):
        return {
            "transcript": "I think we should plant more trees because cities are hotter now.",
            "pauses": [{"start": 1.0, "end": 1.2}],
            "total_duration": 8.0,
            "language": language,
        }

    @patch("score._call_ollama_json")
    def test_llm_output_is_mapped_to_contract(self, mock_llm):
        mock_llm.return_value = {
            "overall_score": 82,
            "subscores": {
                "fluency": 80,
                "grammar": 84,
                "vocabulary": 79,
                "coherence": 85,
                "clarity_proxy": 83,
            },
            "category_feedback": {
                "fluency": ["Mostly smooth with occasional pauses."],
                "grammar": ["Sentence forms are mostly correct."],
                "vocabulary": ["Good range for a short response."],
                "coherence": ["Answer remains on-topic."],
                "clarity_proxy": ["Delivery appears clear."],
            },
            "issues": [
                {
                    "id": "fluency_pause_control",
                    "category": "fluency",
                    "severity": 0.42,
                    "message": "Pauses occasionally interrupt flow.",
                    "evidence": {"pause_ratio": 0.12},
                    "suggestion_hint": "Practice linking clauses before speaking.",
                }
            ],
            "strengths": [{"category": "coherence", "message": "Strong topic relevance."}],
            "feedback_summary": ["Good structure overall; reduce pauses for higher fluency."],
        }

        report = evaluate_pronunciation(self._payload(), prompt="How can we improve the environment?")

        self.assertTrue(report["llm_grading"]["success"])
        self.assertEqual(report["overall_score"], 82.0)
        self.assertEqual(report["subscores"]["grammar"], 84.0)
        self.assertIn("suggestion_generator_input", report)
        self.assertEqual(
            report["suggestion_generator_input"]["task"],
            "generate_learner_feedback_suggestions",
        )
        self.assertEqual(report["issues"][0]["id"], "fluency_pause_control")

    @patch("score._call_ollama_json")
    def test_malay_and_tamil_language_passthrough(self, mock_llm):
        mock_llm.return_value = {
            "overall_score": 70,
            "subscores": {
                "fluency": 70,
                "grammar": 70,
                "vocabulary": 70,
                "coherence": 70,
                "clarity_proxy": 70,
            },
            "category_feedback": {
                "fluency": ["ok"],
                "grammar": ["ok"],
                "vocabulary": ["ok"],
                "coherence": ["ok"],
                "clarity_proxy": ["ok"],
            },
            "issues": [],
            "strengths": [],
            "feedback_summary": ["ok"],
        }

        ms_report = evaluate_pronunciation(self._payload(language="ms"))
        ta_report = evaluate_pronunciation(self._payload(language="ta"))

        self.assertEqual(ms_report["language"], "ms")
        self.assertEqual(ta_report["language"], "ta")

    @patch("score._call_ollama_json", side_effect=TimeoutError("timeout"))
    def test_llm_failure_returns_structured_fallback(self, _mock_llm):
        report = evaluate_pronunciation(self._payload(), prompt="test")

        self.assertFalse(report["llm_grading"]["success"])
        self.assertEqual(report["overall_score"], 0.0)
        self.assertEqual(report["issues"][0]["id"], "llm_unavailable")
        self.assertIn("suggestion_generator_input", report)


if __name__ == "__main__":
    unittest.main()
