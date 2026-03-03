import pathlib
import unittest


class TestAppRegressions(unittest.TestCase):
    def test_attempt_history_uses_append_not_insert(self):
        app_text = pathlib.Path("app.py").read_text(encoding="utf-8")
        self.assertIn("st.session_state.history.append(", app_text)
        self.assertNotIn("st.session_state.history.insert(", app_text)

    def test_question_text_is_html_escaped_before_render(self):
        app_text = pathlib.Path("app.py").read_text(encoding="utf-8")
        self.assertIn("from html import escape", app_text)
        self.assertIn("escape(st.session_state.practice_question)", app_text)


if __name__ == "__main__":
    unittest.main()
