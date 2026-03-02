import unittest

from score import evaluate_pronunciation


class TestScoreEngine(unittest.TestCase):
    def test_output_contains_expected_top_level_fields(self):
        payload = {
            "transcript": "I think we should plant more trees because cities are hotter now.",
            "pauses": [{"start": 1.0, "end": 1.2}],
            "total_duration": 8.0,
            "language": "en",
        }
        report = evaluate_pronunciation(payload, prompt="How can we improve the environment?", use_ollama=False)

        self.assertIn("overall_score", report)
        self.assertIn("language", report)
        self.assertIn("subscores", report)
        self.assertIn("metrics", report)
        self.assertIn("issues", report)
        self.assertIn("suggestion_generator_input", report)
        self.assertEqual(report["language"], "en")
        self.assertGreaterEqual(report["overall_score"], 0.0)
        self.assertLessEqual(report["overall_score"], 100.0)

    def test_pause_ratio_and_wpm_metrics_are_calculated(self):
        payload = {
            "transcript": "one two three four five six",
            "pauses": [{"start": 1.0, "end": 2.0}],
            "total_duration": 6.0,
            "language": "en",
        }
        report = evaluate_pronunciation(payload, prompt="Say six words.", use_ollama=False)
        metrics = report["metrics"]

        # 6 words in 0.1 minutes => 60 WPM
        self.assertAlmostEqual(metrics["wpm"], 60.0, places=2)
        # 1 second pause over 6 seconds total => 0.1667 pause ratio
        self.assertAlmostEqual(metrics["pause_ratio"], 1.0 / 6.0, places=4)

    def test_suggestion_generator_input_structure(self):
        payload = {
            "transcript": "um I think um the city should add more trees.",
            "pauses": [{"start": 0.7, "end": 1.3}, {"start": 2.0, "end": 2.4}],
            "total_duration": 7.0,
            "language": "en",
        }
        report = evaluate_pronunciation(payload, prompt="How to improve city life?", use_ollama=False)
        sgi = report["suggestion_generator_input"]

        self.assertEqual(sgi["task"], "generate_learner_feedback_suggestions")
        self.assertIn("top_issues", sgi)
        self.assertIn("suggestion_candidates", sgi)
        self.assertIn("generator_guidance", sgi)
        self.assertIsInstance(sgi["top_issues"], list)
        self.assertIsInstance(sgi["suggestion_candidates"], list)

    def test_malay_language_profile_is_applied(self):
        payload = {
            "transcript": "Saya belajar bahasa setiap hari kerana saya mahu bercakap dengan lebih lancar.",
            "pauses": [{"start": 0.8, "end": 1.1}],
            "total_duration": 8.0,
            "language": "ms",
        }
        report = evaluate_pronunciation(payload, prompt="Ceritakan tabiat belajar anda.", use_ollama=False)

        self.assertEqual(report["language"], "ms")
        self.assertEqual(report["metrics"]["wpm_target_min"], 95.0)
        self.assertEqual(report["metrics"]["wpm_target_max"], 160.0)
        self.assertGreater(report["metrics"]["num_words"], 0)

    def test_tamil_tokenization_counts_words(self):
        payload = {
            "transcript": "à®¨à®¾à®©à¯ à®¤à®¿à®©à®®à¯à®®à¯ à®•à®¾à®²à¯ˆ à®“à®Ÿà¯à®•à®¿à®±à¯‡à®©à¯ à®à®©à¯†à®©à®¿à®²à¯ à®…à®¤à¯ à®‰à®Ÿà®²à¯à®¨à®²à®¤à¯à®¤à®¿à®±à¯à®•à¯ à®¨à®²à¯à®²à®¤à¯",
            "pauses": [{"start": 0.6, "end": 1.0}],
            "total_duration": 8.0,
            "language": "ta",
        }
        report = evaluate_pronunciation(payload, prompt="à®‰à®™à¯à®•à®³à¯ à®ªà®´à®•à¯à®•à®¤à¯à®¤à¯ˆ à®µà®¿à®³à®•à¯à®•à¯à®™à¯à®•à®³à¯.", use_ollama=False)

        self.assertEqual(report["language"], "ta")
        # Confirms Tamil script is tokenized rather than dropped.
        self.assertGreaterEqual(report["metrics"]["num_words"], 6)
        self.assertEqual(report["metrics"]["wpm_target_min"], 90.0)
        self.assertEqual(report["metrics"]["wpm_target_max"], 155.0)

    def test_fallback_to_english_for_unknown_language(self):
        payload = {
            "transcript": "I answer with plain English words.",
            "pauses": [],
            "total_duration": 5.0,
            "language": "xx",
        }
        report = evaluate_pronunciation(payload, use_ollama=False)

        # Report keeps requested language while thresholds use fallback profile.
        self.assertEqual(report["language"], "xx")
        self.assertEqual(report["metrics"]["wpm_target_min"], 90.0)
        self.assertEqual(report["metrics"]["wpm_target_max"], 150.0)

    def test_high_pause_ratio_creates_fluency_issue(self):
        payload = {
            "transcript": "I think we can help by recycling.",
            "pauses": [{"start": 0.5, "end": 1.8}],  # 1.3s pause over 4s total
            "total_duration": 4.0,
            "language": "en",
        }
        report = evaluate_pronunciation(payload, prompt="How can we help the environment?", use_ollama=False)
        issue_ids = [item["id"] for item in report["issues"]]

        self.assertIn("fluency_pause_ratio_high", issue_ids)
        self.assertTrue(
            any(
                "pause ratio was" in reason
                for reason in report["category_feedback"]["fluency"]
            )
        )


if __name__ == "__main__":
    unittest.main()
