import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import streamlit as st

from question import get_question, SAMPLE_PROMPTS
from score import evaluate_pronunciation
from transcribe import transcribe_audio

SAMPLE_RATE = 16000
MAX_DURATION = 60
SILENCE_THRESHOLD = 0.01
SILENCE_LIMIT = 4
CSS_FILE = Path(__file__).parent / "styles" / "app.css"
LANGUAGE_OPTIONS = ["ja", "en", "ms", "ta", "zh", "es", "fr", "de"]

st.set_page_config(page_title="Conversation Speaking Practice", page_icon="JP", layout="wide")

if "selected_language" not in st.session_state:
    st.session_state.selected_language = "ja"

if "selected_prompt_key" not in st.session_state:
    st.session_state.selected_prompt_key = "beginner"

if "custom_prompt_text" not in st.session_state:
    st.session_state.custom_prompt_text = ""

if "practice_question" not in st.session_state:
    st.session_state.practice_question = get_question(
        prompt_key=st.session_state.get("selected_prompt_key", "beginner"),
        custom_prompt=(st.session_state.get("custom_prompt_text", "").strip() or None),
        target_language=st.session_state.selected_language,
    )

def init_state():
    if "practice_question" not in st.session_state:
        st.session_state.practice_question = get_question(
            target_language=st.session_state.get("selected_language", "ja")
        )
    if "history" not in st.session_state:
        st.session_state.history = []
    if "theme" not in st.session_state:
        st.session_state.theme = "day"


def on_language_change():
    st.session_state.practice_question = get_question(
        prompt_key=st.session_state.get("selected_prompt_key", "beginner"),
        custom_prompt=(st.session_state.get("custom_prompt_text", "").strip() or None),
        target_language=st.session_state.get("selected_language", "ja"),
    )
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


def load_css():
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def build_pause_spans(segments):
    pauses = []
    if not segments:
        return pauses
    for idx in range(1, len(segments)):
        prev_seg = segments[idx - 1]
        curr_seg = segments[idx]
        gap = float(curr_seg["start"]) - float(prev_seg["end"])
        if gap > 0.15:
            pauses.append({
                "start": round(float(prev_seg["end"]), 3),
                "end": round(float(curr_seg["start"]), 3),
            })
    return pauses


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
    st.selectbox(
        "Response language",
        options=LANGUAGE_OPTIONS,
        key="selected_language",
        on_change=on_language_change,
        help="Language code used for transcription + scoring.",
    )
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

st.markdown('<p class="title">Conversation Speaking Practice</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Record your voice, then review transcript and confidence.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f"""
    <div class="sentence">
      <div class="sentence-label">Practice Sentence</div>
            <div class="sentence-text">{st.session_state.practice_question}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

record_clicked = st.button("Record", use_container_width=True)

prompt_options = list(SAMPLE_PROMPTS.keys()) + ["custom"]
style_col, _ = st.columns([1, 3])
with style_col:
    selected_prompt_key = st.selectbox(
        "Prompt style",
        prompt_options,
        key="selected_prompt_key",
    )

custom_prompt = None
if selected_prompt_key == "custom":
    custom_prompt = st.text_area(
        "Custom prompt",
        key="custom_prompt_text",
        placeholder="Provide some topics to generate a prompt!",
    )

if st.button("Generate Question"):
    st.session_state.practice_question = get_question(
        prompt_key=selected_prompt_key,
        custom_prompt=(custom_prompt.strip() if custom_prompt else None),
        target_language=st.session_state.get("selected_language", "ja"),
    )
    st.rerun()

timer_placeholder = st.empty()

if record_clicked:
    active_language = st.session_state.get("selected_language", "ja")
    audio_array, elapsed_time = record_audio(timer_placeholder)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, audio_array, SAMPLE_RATE)
        temp_path = tmpfile.name

    with open(temp_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    with st.spinner("Transcribing..."):
        result = transcribe_audio(temp_path, language=active_language)

    pauses = build_pause_spans(result["segments"])
    scoring_payload = {
        "transcript": result["transcript"],
        "pauses": pauses,
        "total_duration": float(elapsed_time),
        "language": active_language,
    }
    with st.spinner("Scoring response..."):
        score_report = evaluate_pronunciation(
            scoring_payload,
            prompt=st.session_state.practice_question,
            language=active_language,
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

    st.session_state.history.insert(
        0,
        {
            "audio_bytes": audio_bytes,
            "transcript": result["transcript"],
            "avg_confidence": avg_confidence,
            "elapsed_time": elapsed_time,
            "segments_df": segment_data,
            "score_report": score_report,
        },
    )

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

    score = item.get("score_report", {})
    if score:
        st.markdown('<div class="result-card"><b>Scoring Dashboard</b></div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            st.metric("Overall Score", f'{score.get("overall_score", 0):.1f}/100')
        with s2:
            llm_state = score.get("llm_grading", {}).get("success", False)
            st.metric("LLM Grader", "Online" if llm_state else "Offline")
        with s3:
            st.metric("Language", score.get("language", "n/a"))

        llm_meta = score.get("llm_grading", {})
        if llm_meta:
            st.caption(
                f"Model: {llm_meta.get('model', 'n/a')} | Endpoint: {llm_meta.get('endpoint', 'n/a')} | "
                f"Timeout: {llm_meta.get('timeout_seconds', 'n/a')}s"
            )
            if not llm_meta.get("success", False):
                st.error(f"LLM error: {llm_meta.get('error', 'Unknown error')}")

        subscores = score.get("subscores", {})
        if subscores:
            sub_df = pd.DataFrame(
                [{"Category": k, "Score": v} for k, v in subscores.items()]
            ).sort_values("Score", ascending=False)
            st.bar_chart(sub_df.set_index("Category"))

        feedback_summary = score.get("feedback_summary") or []
        if feedback_summary:
            st.markdown("**Feedback Summary**")
            for line in feedback_summary:
                st.write(f"- {line}")

        issues = score.get("issues") or []
        if issues:
            st.markdown("**Top Improvement Areas**")
            for issue in issues[:3]:
                evidence = issue.get("evidence", {})
                hint = issue.get("suggestion_hint", "")
                st.write(
                    f"- [{issue.get('category', 'general')}] {issue.get('message', '')} "
                    f"(severity {issue.get('severity', 0):.2f})"
                )
                if evidence:
                    st.caption(f"Evidence: {evidence}")
                if hint:
                    st.caption(f"Suggestion: {hint}")

        suggestion_input = score.get("suggestion_generator_input", {})
        if suggestion_input:
            with st.expander("Suggestion Generator Input (JSON)"):
                st.json(suggestion_input)

    st.markdown('<div class="result-card"><b>Full Transcription</b></div>', unsafe_allow_html=True)
    st.write(item["transcript"] or "_No speech detected._")

    if not item["segments_df"].empty:
        st.markdown('<div class="result-card"><b>Segments</b></div>', unsafe_allow_html=True)
        st.dataframe(item["segments_df"], use_container_width=True)
