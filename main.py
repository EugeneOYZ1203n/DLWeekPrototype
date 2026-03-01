# main.py
# Entry point for the Japanese pronunciation grading app

from transcribe import transcribe_audio
from score import evaluate_pronunciation
from question import get_question

def run_session():
    """
    Main loop for a single practice session:
    1. Generate or select a question
    2. Record or get user audio
    3. Transcribe audio
    4. Score pronunciation
    5. Display feedback
    """
    # TODO: implement session workflow
    pass

if __name__ == "__main__":
    run_session()