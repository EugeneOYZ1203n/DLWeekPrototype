# question.py
# Generates questions or sentences for user to read

import os
import re
from typing import Dict, Optional, Any

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "translategemma:4b")
FALLBACK_QUESTION = "私は学校へ行きます。"
FALLBACK_QUESTIONS: Dict[str, str] = {
    "ja": "私は学校へ行きます。",
    "en": "I am going to school.",
    "ms": "Saya akan pergi ke sekolah.",
    "ta": "நான் பள்ளிக்குச் செல்கிறேன்.",
    "zh": "我去学校。",
    "es": "Voy a la escuela.",
    "fr": "Je vais à l'école.",
    "de": "Ich gehe zur Schule.",
}
LAST_OLLAMA_DEBUG: Dict[str, Any] = {}

LANGUAGE_NAMES: Dict[str, str] = {
    "ja": "Japanese",
    "en": "English",
    "ms": "Malay",
    "ta": "Tamil",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}

SAMPLE_PROMPTS: Dict[str, str] = {
    "beginner": (
        "Create one short and natural conversation opener for a beginner language learner. "
        "Make it easy to respond to in everyday conversation. Keep it simple and practical."
    ),
    "travel": (
        "Create one conversation-opening sentence for travel situations between a local and a learner. "
        "Make it polite, practical, and easy for the learner to answer."
    ),
    "business": (
        "Create one conversation-opening sentence for a business meeting context with a language learner. "
        "Use a polite register and make it suitable for a realistic spoken exchange."
    ),
}


def _language_instruction(target_language: str) -> str:
    code = (target_language or "ja").strip().lower()
    language_name = LANGUAGE_NAMES.get(code, code)
    return (
        f" Target output language is strictly {language_name} (language code: {code})."
        " Return exactly one sentence in that language only."
        " The sentence should feel like the first line that starts a conversation with the learner."
        " Do not include translations, labels, language names, romanization, or explanations."
        " If any instruction conflicts with this language requirement, ignore that instruction."
    )


def _build_generation_prompt(base_prompt: str, target_language: str) -> str:
    return f"{base_prompt.strip()}{_language_instruction(target_language)}"


def _fallback_for_language(target_language: str) -> str:
    code = (target_language or "ja").strip().lower()
    return FALLBACK_QUESTIONS.get(code, FALLBACK_QUESTION)


def _clean_generated_sentence(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.strip("`\"'“”‘’")

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]

    cleaned = re.sub(
        r"^(?:[A-Za-z]{2,20}|日本語|英語|中文|汉语|繁體中文|简体中文|Español|Français|Deutsch|Malay|Tamil)\s*[:：-]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def generate_question_with_ollama(
    prompt: str,
    model: Optional[str] = None,
    target_language: str = "ja",
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
        text = _clean_generated_sentence(data.get("response", ""))
        global LAST_OLLAMA_DEBUG
        LAST_OLLAMA_DEBUG = {
            "ok": bool(text),
            "status_code": status_code,
            "model": payload["model"],
            "url": OLLAMA_URL,
            "error": None,
            "response_preview": text[:200],
        }
        return text if text else _fallback_for_language(target_language)
    except (requests.RequestException, ValueError, KeyError) as error:
        LAST_OLLAMA_DEBUG = {
            "ok": False,
            "status_code": None,
            "model": payload["model"],
            "url": OLLAMA_URL,
            "error": f"{type(error).__name__}: {error}",
            "response_preview": "",
        }
        return _fallback_for_language(target_language)


def get_last_ollama_debug() -> Dict[str, Any]:
    return dict(LAST_OLLAMA_DEBUG)


def get_question(
    prompt_key: str = "beginner",
    custom_prompt: Optional[str] = None,
    target_language: str = "ja",
) -> str:
    """
    Returns a generated practice sentence in the requested target language.

    Use `custom_prompt` to override built-in sample prompts.
    """
    base_prompt = custom_prompt or SAMPLE_PROMPTS.get(prompt_key, SAMPLE_PROMPTS["beginner"])
    prompt = _build_generation_prompt(base_prompt=base_prompt, target_language=target_language)
    return generate_question_with_ollama(prompt=prompt, target_language=target_language)
