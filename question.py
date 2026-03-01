# question.py
# Generates questions or sentences for user to read

import random
from typing import Dict

# Example question bank
QUESTION_BANK = [
    "私は学校へ行きます。",
    "今日の天気はいいですね。",
    "明日、友達と映画を見に行きます。",
    "昨日、寿司を食べました。"
]

def get_question() -> str:
    """
    Returns a random question/sentence for the user to read
    """
    return random.choice(QUESTION_BANK)

def add_question(question: str):
    """
    Adds a new question to the bank (optional)
    """
    QUESTION_BANK.append(question)