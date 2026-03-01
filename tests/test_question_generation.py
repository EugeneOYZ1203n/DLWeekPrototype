import unittest
from unittest.mock import patch

import requests

import question


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class TestQuestionGeneration(unittest.TestCase):
    @patch("question.requests.post")
    def test_generate_question_success(self, mock_post):
        mock_post.return_value = FakeResponse({"response": "  今日はいい天気ですね。  "})

        result = question.generate_question_with_ollama("any prompt")

        self.assertEqual(result, "今日はいい天気ですね。")

    @patch("question.requests.post")
    def test_generate_question_empty_response_falls_back(self, mock_post):
        mock_post.return_value = FakeResponse({"response": "   "})

        result = question.generate_question_with_ollama("any prompt")

        self.assertEqual(result, question.FALLBACK_QUESTION)

    @patch("question.requests.post")
    def test_generate_question_request_error_falls_back(self, mock_post):
        mock_post.side_effect = requests.RequestException("connection error")

        result = question.generate_question_with_ollama("any prompt")

        self.assertEqual(result, question.FALLBACK_QUESTION)

    @patch("question.requests.post")
    def test_get_question_uses_custom_prompt_when_provided(self, mock_post):
        mock_post.return_value = FakeResponse({"response": "テスト文です。"})
        custom_prompt = "Return a short Japanese sentence about food."

        _ = question.get_question(prompt_key="beginner", custom_prompt=custom_prompt)

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["prompt"], custom_prompt)

    @patch("question.requests.post")
    def test_get_question_uses_beginner_prompt_for_unknown_key(self, mock_post):
        mock_post.return_value = FakeResponse({"response": "テスト文です。"})

        _ = question.get_question(prompt_key="unknown-key")

        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["json"]["prompt"], question.SAMPLE_PROMPTS["beginner"])


if __name__ == "__main__":
    unittest.main()
