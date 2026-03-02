import os
import re
import tempfile
import time
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import streamlit as st

from question import SAMPLE_PROMPTS, get_last_ollama_debug, get_question
from score import evaluate_pronunciation
from suggestion import SuggestionGenerator
from transcribe import transcribe_audio

SAMPLE_RATE = 16000
MAX_DURATION = 60
SILENCE_THRESHOLD = 0.01
SILENCE_LIMIT = 4
CSS_FILE = Path(__file__).parent / "styles" / "app.css"

st.set_page_config(page_title="Japanese Speaking Practice", page_icon="JP", layout="wide")


# Practice question controls
prompt_options = list(SAMPLE_PROMPTS.keys()) + ["custom"]
style_col, _ = st.columns([1, 3])
with style_col:
    selected_prompt_key = st.selectbox("Prompt style", prompt_options, index=0)

custom_prompt = None
if selected_prompt_key == "custom":
    custom_prompt = st.text_area(
        "Custom prompt",
        value="",
        placeholder="Provide some topics to generate a prompt!",
    )

if "practice_question" not in st.session_state:
    st.session_state.practice_question = get_question(prompt_key=selected_prompt_key)

if st.button("Generate Question"):
    st.session_state.practice_question = get_question(
        prompt_key=selected_prompt_key,
        custom_prompt=(custom_prompt.strip() if custom_prompt else None),
    )

if not str(st.session_state.practice_question or "").strip():
    st.session_state.practice_question = "Question generation failed. Please click Generate Question again."

question = st.session_state.practice_question

def init_state():
    if "practice_question" not in st.session_state:
        st.session_state.practice_question = get_question()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "theme" not in st.session_state:
        st.session_state.theme = "day"


def new_prompt():
    st.session_state.practice_question = get_question()
    st.session_state.history = []


def record_audio(timer_placeholder):
    st.info("Recording... Speak now.")
    frames = []
    start_time = time.time()
    last_sound_time = start_time

    def callback(indata, frames_count, time_info, status):
        nonlocal last_sound_time
        _ = frames_count
        _ = time_info
        _ = status
        frames.append(indata.copy())
        rms = np.sqrt(np.mean(indata**2))
        if rms > SILENCE_THRESHOLD:
            last_sound_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback):
        while True:
            elapsed = time.time() - start_time
            timer_placeholder.markdown(
                f'<div class="timer">Recording: {int(elapsed)}s</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.2)
            if (time.time() - last_sound_time > SILENCE_LIMIT) or elapsed > MAX_DURATION:
                break

    audio_array = np.concatenate(frames, axis=0)
    return audio_array, elapsed


def extract_pause_spans(segments):
    spans = []
    if not segments:
        return spans
    for i in range(1, len(segments)):
        prev_end = float(segments[i - 1].get("end", 0.0) or 0.0)
        curr_start = float(segments[i].get("start", 0.0) or 0.0)
        if curr_start > prev_end:
            spans.append({"start": prev_end, "end": curr_start})
    return spans


def build_speech_metrics(score_report):
    metrics = score_report.get("metrics", {}) if isinstance(score_report, dict) else {}
    pause_count = int(metrics.get("pause_count", 0) or 0)
    total_pause_seconds = float(metrics.get("total_pause_seconds", 0.0) or 0.0)
    avg_pause_ms = (total_pause_seconds / pause_count * 1000.0) if pause_count > 0 else 0.0
    return {
        "pause_count": pause_count,
        "avg_pause_ms": round(avg_pause_ms, 2),
        "hesitation_count": int(metrics.get("filler_count", 0) or 0),
        "tone_consistency": 1.0,
    }


def _tokenize_simple(text):
    return re.findall(r"[A-Za-z0-9]+|[\u3040-\u30ff\u3400-\u9fff]+", (text or "").lower())


def is_likely_japanese(text):
    if not text:
        return False
    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False
    jp_count = sum(1 for ch in chars if re.match(r"[\u3040-\u30ff\u3400-\u9fff]", ch))
    return (jp_count / len(chars)) >= 0.35


def is_relevant_to_question(question_text, transcript_text, score_report):
    issues = score_report.get("issues", []) if isinstance(score_report, dict) else []
    if any(issue.get("id") == "coherence_prompt_mismatch" for issue in issues if isinstance(issue, dict)):
        return False, "Low prompt overlap detected by scorer."

    q_tokens = set(_tokenize_simple(question_text))
    t_tokens = set(_tokenize_simple(transcript_text))
    if not q_tokens or not t_tokens:
        return False, "Insufficient text to validate relevance."
    overlap = len(q_tokens.intersection(t_tokens)) / max(1, len(q_tokens))
    if overlap >= 0.15:
        return True, f"Token overlap with question: {overlap:.2f}"
    return False, f"Low token overlap with question: {overlap:.2f}"


def load_css():
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def apply_theme():
    if st.session_state.theme == "night":
        st.markdown(
            """
            <style>
            :root {
                            --bg-start: var(--night-bg-start);
                            --bg-mid: var(--night-bg-mid);
                            --bg-end: var(--night-bg-end);
                            --text: var(--night-text);
                            --subtext: var(--night-subtext);
                            --border: var(--night-border);
                            --brand: var(--night-brand);
                            --brand-soft: var(--night-brand-soft);
                            --panel: var(--night-panel);
                            --result-bg: var(--night-result-bg);
                            --timer-bg: var(--night-timer-bg);
                            --timer-text: var(--night-timer-text);
                            --sidebar-bg: var(--night-sidebar-bg);
                            --sidebar-text: var(--night-sidebar-text);
                            --button-bg: var(--night-button-bg);
                            --button-border: var(--night-button-border);
                            --button-hover: var(--night-button-hover);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            :root {
                            --bg-start: var(--day-bg-start);
                            --bg-mid: var(--day-bg-mid);
                            --bg-end: var(--day-bg-end);
                            --text: var(--day-text);
                            --subtext: var(--day-subtext);
                            --border: var(--day-border);
                            --brand: var(--day-brand);
                            --brand-soft: var(--day-brand-soft);
                            --panel: var(--day-panel);
                            --result-bg: var(--day-result-bg);
                            --timer-bg: var(--day-timer-bg);
                            --timer-text: var(--day-timer-text);
                            --sidebar-bg: var(--day-sidebar-bg);
                            --sidebar-text: var(--day-sidebar-text);
                            --button-bg: var(--day-button-bg);
                            --button-border: var(--day-button-border);
                            --button-hover: var(--day-button-hover);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def toggle_theme():
    st.session_state.theme = "night" if st.session_state.theme == "day" else "day"


init_state()
load_css()
apply_theme()

with st.sidebar:
    st.header("Session")
    st.caption("Speaking practice controls")
    theme_label = (
        "Switch to Night Mode"
        if st.session_state.theme == "day"
        else "Switch to Day Mode"
    )
    st.button(theme_label, use_container_width=True, on_click=toggle_theme)
    st.divider()
    st.caption("Auto-stop settings")
    st.caption(f"Max duration: {MAX_DURATION}s")
    st.caption(f"Silence stop: {SILENCE_LIMIT}s")
    st.caption("Tip: Speak clearly and at a steady pace.")

st.markdown('<p class="title">Japanese Speaking Practice</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Record your voice, then review transcript and confidence.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="sentence">
      <div class="sentence-label">Practice Sentence</div>
            <div class="sentence-text">{escape(st.session_state.practice_question)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
q_debug = get_last_ollama_debug()
if q_debug:
    if q_debug.get("ok"):
        st.caption(f'Question source: Ollama ({q_debug.get("model", "unknown model")})')
    else:
        st.warning(
            "Question generator fallback in use. "
            f'Error: {q_debug.get("error", "Unknown error")}'
        )

timer_placeholder = st.empty()

if st.button("Record", use_container_width=True):
    audio_array, elapsed_time = record_audio(timer_placeholder)
    processing_status = st.empty()
    progress_bar = st.progress(0)
    processing_status.info("Processing recording...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, audio_array, SAMPLE_RATE)
        temp_path = tmpfile.name

    with open(temp_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    progress_bar.progress(15)
    processing_status.info("Transcribing audio...")
    with st.spinner("Transcribing..."):
        result = transcribe_audio(temp_path)
    pauses = extract_pause_spans(result["segments"])
    duration_seconds = float(elapsed_time)
    score_payload = {
        "transcript": result["transcript"],
        "pauses": pauses,
        "total_duration": duration_seconds,
        "language": "ja",
    }
    progress_bar.progress(55)
    processing_status.info("Scoring pronunciation...")
    with st.spinner("Scoring pronunciation..."):
        score_report = evaluate_pronunciation(
            score_payload,
            prompt=st.session_state.practice_question,
            language="ja",
            use_ollama=True,
        )
    suggestion_generator = SuggestionGenerator(
        model=os.getenv("OLLAMA_SUGGESTION_MODEL", "gemma2:2b")
    )
    progress_bar.progress(80)
    processing_status.info("Generating suggestions...")
    with st.spinner("Generating coaching suggestions..."):
        suggestions_report = suggestion_generator.generate(
            transcript=result["transcript"],
            score_report=score_report,
            speech_metrics=build_speech_metrics(score_report),
            target_sentence=st.session_state.practice_question,
            language="Japanese",
        )
    language_ok = is_likely_japanese(result["transcript"])
    relevance_ok, relevance_reason = is_relevant_to_question(
        st.session_state.practice_question,
        result["transcript"],
        score_report,
    )

    segment_confidences = [np.exp(seg["avg_logprob"]) for seg in result["segments"]]
    avg_confidence = float(np.mean(segment_confidences)) if segment_confidences else 0.0

    segment_data = pd.DataFrame(
        [
            {
                "Start (s)": seg["start"],
                "End (s)": seg["end"],
                "Text": seg["text"],
                "Confidence": np.exp(seg["avg_logprob"]),
            }
            for seg in result["segments"]
        ]
    )

    st.session_state.history.append(
        {
            "audio_bytes": audio_bytes,
            "transcript": result["transcript"],
            "avg_confidence": avg_confidence,
            "elapsed_time": elapsed_time,
            "segments_df": segment_data,
            "score_report": score_report,
            "suggestions_report": suggestions_report,
            "validation": {
                "language_ok": language_ok,
                "relevance_ok": relevance_ok,
                "relevance_reason": relevance_reason,
            },
        }
    )

    progress_bar.progress(100)
    processing_status.success("Processing complete.")
    os.remove(temp_path)
    timer_placeholder.empty()

for idx, item in enumerate(st.session_state.history, start=1):
    st.markdown(f"### Attempt {idx}")
    st.audio(item["audio_bytes"])
    st.download_button(
        "Download Recording (.wav)",
        data=item["audio_bytes"],
        file_name=f"recording_attempt_{idx}.wav",
        mime="audio/wav",
        key=f"download_{idx}",
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="result-card"><b>Audio Duration</b></div>', unsafe_allow_html=True)
        st.write(f'{item["elapsed_time"]:.2f} seconds')
    with col2:
        st.markdown(
            '<div class="result-card"><b>Pronunciation Confidence (Rough)</b></div>',
            unsafe_allow_html=True,
        )
        st.write(f'{item["avg_confidence"]:.2f}')

    st.markdown('<div class="result-card"><b>Full Transcription</b></div>', unsafe_allow_html=True)
    st.text_area(
        f"Attempt {idx} Transcript",
        value=(item["transcript"] or "No speech detected."),
        height=120,
        disabled=True,
        key=f"transcript_{idx}",
    )

    validation = item.get("validation", {})
    if validation:
        st.markdown('<div class="result-card"><b>Relevance & Language Check</b></div>', unsafe_allow_html=True)
        if validation.get("language_ok"):
            st.success("Language check passed: response appears to be in Japanese.")
        else:
            st.error("Language check failed: response does not look like Japanese.")
        if validation.get("relevance_ok"):
            st.success(f'Relevance check passed: {validation.get("relevance_reason", "")}')
        else:
            st.warning(f'Relevance check failed: {validation.get("relevance_reason", "")}')

    score_report = item.get("score_report", {})
    if score_report:
        score_metrics = score_report.get("metrics", {})
        st.markdown('<div class="result-card"><b>Score Summary</b></div>', unsafe_allow_html=True)
        st.write(f'Overall: {score_report.get("overall_score", 0)} / 100')
        st.write(
            "Scorer: "
            f'{score_metrics.get("scorer", "unknown")} '
            f'({score_metrics.get("scorer_model", "default")})'
        )
        subscores = score_report.get("subscores", {})
        if isinstance(subscores, dict) and subscores:
            st.dataframe(pd.DataFrame([subscores]), use_container_width=True)

    suggestions_report = item.get("suggestions_report", {})
    if suggestions_report:
        st.markdown('<div class="result-card"><b>Coaching Suggestions</b></div>', unsafe_allow_html=True)
        st.write(suggestions_report.get("overall_assessment", ""))
        for suggestion in suggestions_report.get("suggestions", []):
            category = suggestion.get("category", "general")
            issue = suggestion.get("issue", "")
            action = suggestion.get("action", "")
            practice_example = suggestion.get("practice_example", "")
            st.markdown(f"- **{category}**: {issue}")
            if action:
                st.write(f"Action: {action}")
            if practice_example:
                st.write(f"Practice: {practice_example}")

    if not item["segments_df"].empty:
        st.markdown('<div class="result-card"><b>Segments</b></div>', unsafe_allow_html=True)
        st.dataframe(item["segments_df"], use_container_width=True)
