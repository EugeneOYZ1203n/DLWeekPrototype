import json
from typing import Dict, Any, List
import re
from ollama import chat

from commons import clean_json_string


system_prompt = """
You are an expert Japanese language-proficiency scorer.

You must output structured sections in a streaming-friendly format.

Do NOT output JSON.

Instead, output content in clearly separated sections using this exact format:

SECTION: SCORES
fluency: <0-100>
grammar: <0-100>
vocabulary: <0-100>
relavence: <0-100>

SECTION: CATEGORY_FEEDBACK
fluency: <short reason, MUST BE RELATED TO FLUENCY LIKE PAUSES, FILLER WORDS>
grammar: <short reason, MUST BE RELATED TO GRAMMAR LIKE PARTICLE USE and SENTENCE STRUCTURE AND TENSES>
vocabulary: <short reason, MUST BE RELATED TO VOCABULARY LIKE WORD CHOICE AND VARIETY>
relevance: <short reason, MUST BE RELATED TO HOW THE ANSWER IS RELEVANT TO THE QUESTION>

For each detected issue, output:

SECTION: ISSUE
id: <string_id>
category: fluency|grammar|vocabulary|coherence|clarity_proxy
severity: <0-1>
message: <short description>
evidence: <metrics or transcript snippet>
suggestion_hint: <short actionable tip>

At the end output:

SECTION: SUMMARY
- <point 1>
- <point 2>

Rules:
- Always follow the exact section labels.
- Keep sections separated.
- Be concise.
- Use transcript and metrics to justify issues.
"""

from ollama import chat

def evaluate_streaming(input_payload: Dict[str, Any], question, stats, on_chunk=None):
    transcript = input_payload.get("transcript", "")

    content = f"""
SYSTEM INSTRUCTIONS:
{system_prompt}

QUESTION:
{question}

TRANSCRIPT:
\"\"\"{transcript}\"\"\"

ESTIMATED METRICS:
{json.dumps(stats)}
"""

    response = chat(
        model="translategemma:4b",
        messages=[{"role": "user", "content": content}],
        stream=True,
    )

    full_text = ""

    for chunk in response:
        piece = chunk["message"]["content"]
        full_text += piece

        if on_chunk:
            on_chunk(piece)

    return full_text


from typing import Dict, Any, List
import re


def calculate_extra_stats(input_payload: Dict[str, Any]) -> Dict[str, Any]:
    transcript: str = input_payload.get("transcript", "") or ""
    segments: List[Dict[str, Any]] = input_payload.get("segments", []) or []

    # -----------------------------
    # 1. Duration Calculation
    # -----------------------------
    if segments:
        start_time = float(segments[0].get("start", 0.0))
        end_time = float(segments[-1].get("end", 0.0))
        total_speaking_time = max(0.0, end_time - start_time)
    else:
        total_speaking_time = 0.0

    # -----------------------------
    # 2. Pause Detection (from gaps)
    # -----------------------------
    pause_durations = []

    for i in range(1, len(segments)):
        prev_end = float(segments[i - 1].get("end", 0.0))
        curr_start = float(segments[i].get("start", 0.0))
        gap = curr_start - prev_end

        # Ignore micro-gaps under 200ms
        if gap > 0.2:
            pause_durations.append(gap)

    pause_count = len(pause_durations)
    total_pause_time = sum(pause_durations)
    avg_pause_duration = (
        total_pause_time / pause_count if pause_count > 0 else 0.0
    )

    long_pause_count = sum(1 for p in pause_durations if p >= 1.5)

    # -----------------------------
    # 3. Japanese Filler Detection
    # -----------------------------
    filler_words = [
        "えー", "ええ", "えっと", "ええと",
        "あの", "その", "まあ", "なんか",
        "うーん", "えーと"
    ]

    filler_pattern = r"|".join(map(re.escape, filler_words))
    filler_matches = re.findall(filler_pattern, transcript)
    filler_count = len(filler_matches)

    filler_per_minute = (
        (filler_count / total_speaking_time) * 60
        if total_speaking_time > 0
        else 0.0
    )

    # -----------------------------
    # 4. Speech Pace (Japanese heuristic)
    # -----------------------------
    # Count Japanese characters (Hiragana, Katakana, Kanji)
    jp_chars = re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", transcript)
    char_count = len(jp_chars)

    chars_per_minute = (
        (char_count / total_speaking_time) * 60
        if total_speaking_time > 0
        else 0.0
    )

    # -----------------------------
    # 5. Segment Duration Stats
    # -----------------------------
    segment_durations = [
        float(seg["end"]) - float(seg["start"])
        for seg in segments
        if "start" in seg and "end" in seg
    ]

    avg_segment_duration = (
        sum(segment_durations) / len(segment_durations)
        if segment_durations
        else 0.0
    )

    # -----------------------------
    # 6. Return Metrics
    # -----------------------------
    return {
        "speech_stats": {
            "total_speaking_seconds": round(total_speaking_time, 2),
            "chars_per_minute_estimate": round(chars_per_minute, 2),
            "avg_segment_duration": round(avg_segment_duration, 2),
        },
        "pause_stats": {
            "pause_count": pause_count,
            "total_pause_seconds": round(total_pause_time, 2),
            "avg_pause_seconds": round(avg_pause_duration, 2),
            "long_pause_count_1_5s": long_pause_count,
        },
        "filler_stats": {
            "filler_count": filler_count,
            "fillers_per_minute": round(filler_per_minute, 2),
            "detected_fillers": filler_matches,
        }
    }