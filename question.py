# question.py
# Generates questions or sentences for user to read

import os
from typing import Dict, Optional, Any

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "translategemma:4b")
FALLBACK_QUESTION = "私は学校へ行きます。"
LAST_OLLAMA_DEBUG: Dict[str, Any] = {}

SAMPLE_PROMPTS: Dict[str, str] = {
    "beginner": (
        "Create one short and natural Japanese practice sentence for a beginner learner. "
        "Use only N5-level grammar, 6-12 characters if possible, and return only the Japanese sentence."
    ),
    "travel": (
        "Create one Japanese speaking-practice sentence for travel situations. "
        "Keep it polite and practical. Return only the Japanese sentence."
    ),
    "business": (
        "Create one Japanese speaking-practice sentence for a business meeting context. "
        "Use polite Japanese. Return only the Japanese sentence."
    ),
}


def generate_question_with_ollama(
    prompt: str,
    model: Optional[str] = None,
    timeout_seconds: int = 30,
) -> str:
    """
    Generate a single Japanese practice sentence using a local Ollama model.
    """
    payload = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout_seconds)
        status_code = response.status_code
        response.raise_for_status()
        data = response.json()
        text = data.get("response", "").strip()
        global LAST_OLLAMA_DEBUG
        LAST_OLLAMA_DEBUG = {
            "ok": bool(text),
            "status_code": status_code,
            "model": payload["model"],
            "url": OLLAMA_URL,
            "error": None,
            "response_preview": text[:200],
        }
        return text if text else FALLBACK_QUESTION
    except (requests.RequestException, ValueError, KeyError) as error:
        LAST_OLLAMA_DEBUG = {
            "ok": False,
            "status_code": None,
            "model": payload["model"],
            "url": OLLAMA_URL,
            "error": f"{type(error).__name__}: {error}",
            "response_preview": "",
        }
        return FALLBACK_QUESTION


def get_last_ollama_debug() -> Dict[str, Any]:
    return dict(LAST_OLLAMA_DEBUG)


def get_question(prompt_key: str = "beginner", custom_prompt: Optional[str] = None) -> str:
    """
    Returns a generated Japanese practice sentence.

    Use `custom_prompt` to override built-in sample prompts.
    """
    prompt = custom_prompt or SAMPLE_PROMPTS.get(prompt_key, SAMPLE_PROMPTS["beginner"])
    return generate_question_with_ollama(prompt=prompt)