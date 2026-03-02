# app.py

import os
import re
import tempfile
import time
from html import escape

import numpy as np
import sounddevice as sd
import soundfile as sf
import streamlit as st

from question import SAMPLE_PROMPTS, get_question
from score_simplified import calculate_extra_stats, evaluate_streaming
from transcribe import transcribe_audio
from styling import (
    configure_page,
    init_theme_state,
    toggle_theme,
    load_css,
    apply_theme
)

# -------------------------
# Constants
# -------------------------

SAMPLE_RATE = 16000
MAX_DURATION = 60
SILENCE_THRESHOLD = 0.01
SILENCE_LIMIT = 4


# -------------------------
# Session Initialization
# -------------------------

def init_state():
    if "practice_question" not in st.session_state:
        st.session_state.practice_question = get_question()
    if "history" not in st.session_state:
        st.session_state.history = []


# -------------------------
# Audio Recording
# -------------------------

def record_audio(timer_placeholder):
    frames = []
    start_time = time.time()
    last_sound_time = start_time

    def callback(indata, frames_count, time_info, status):
        nonlocal last_sound_time
        frames.append(indata.copy())
        rms = np.sqrt(np.mean(indata ** 2))
        if rms > SILENCE_THRESHOLD:
            last_sound_time = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        callback=callback
    ):
        while True:
            elapsed = time.time() - start_time
            timer_placeholder.markdown(
                f'<div class="timer">Recording: {int(elapsed)}s</div>',
                unsafe_allow_html=True,
            )
            time.sleep(0.2)

            if (
                time.time() - last_sound_time > SILENCE_LIMIT
                or elapsed > MAX_DURATION
            ):
                break

    audio_array = np.concatenate(frames, axis=0)
    return audio_array, elapsed


# -------------------------
# UI Setup
# -------------------------

configure_page()
init_theme_state()
init_state()
load_css()
apply_theme()

# -------------------------
# Sidebar
# -------------------------

with st.sidebar:
    st.header("Session")
    theme_label = (
        "Switch to Night Mode"
        if st.session_state.theme == "day"
        else "Switch to Day Mode"
    )
    st.button(theme_label, use_container_width=True, on_click=toggle_theme)

# -------------------------
# Prompt Selection
# -------------------------

prompt_options = list(SAMPLE_PROMPTS.keys()) + ["custom"]
selected_prompt_key = st.selectbox("Prompt style", prompt_options)

custom_prompt = None
if selected_prompt_key == "custom":
    custom_prompt = st.text_area("Custom prompt")

if st.button("Generate Question"):
    st.session_state.practice_question = get_question(
        prompt_key=selected_prompt_key,
        custom_prompt=(custom_prompt.strip() if custom_prompt else None),
    )

question = st.session_state.practice_question

# -------------------------
# Header
# -------------------------

st.markdown('<p class="title">Japanese Speaking Practice</p>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div class="sentence">
        <div class="sentence-label">Practice Sentence</div>
        <div class="sentence-text">{escape(question)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Recording Button
# -------------------------

if st.button("Record", use_container_width=True):

    timer_placeholder = st.empty()
    status_placeholder = st.empty()

    status_placeholder.info("🎙 Recording... Speak clearly.")

    audio_array, elapsed_time = record_audio(timer_placeholder)

    status_placeholder.info("⏹ Recording complete. Preparing audio...")

    # -------------------------
    # Save Temp Audio
    # -------------------------

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio_array, SAMPLE_RATE)
        temp_path = tmp.name

    with open(temp_path, "rb") as f:
        audio_bytes = f.read()

    # -------------------------
    # Transcription
    # -------------------------

    status_placeholder.info("📝 Transcribing audio...")
    with st.spinner("Transcribing..."):
        transcribe_result = transcribe_audio(temp_path)
        print(transcribe_result)

    # -------------------------
    # Metrics
    # -------------------------

    status_placeholder.info("📊 Analyzing speech metrics...")
    with st.spinner("Analyzing..."):
        stats = calculate_extra_stats(transcribe_result)

    st.markdown("### 📊 Speech Metrics")

    speech = stats["speech_stats"]
    pauses = stats["pause_stats"]
    fillers = stats["filler_stats"]

    st.markdown(
        f"""
        <div class="result-card sentence-text">
        <b>Speaking Time:</b> {speech['total_speaking_seconds']}s<br/>
        <b>Chars / Minute:</b> {speech['chars_per_minute_estimate']}<br/>
        <b>Avg Segment:</b> {speech['avg_segment_duration']}s<br/>
        <b>Pause Count:</b> {pauses['pause_count']}<br/>
        <b>Long Pauses (&gt;1.5s):</b> {pauses['long_pause_count_1_5s']}<br/>
        <b>Avg Pause:</b> {pauses['avg_pause_seconds']}s<br/>
        <b>Filler Count:</b> {fillers['filler_count']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    if fillers["detected_fillers"]:
        st.markdown(
            f"""
            <div class="result-card sentence-text">
            <b>Detected Fillers:</b> {", ".join(fillers["detected_fillers"])}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------
    # LLM Streaming Evaluation
    # -------------------------

    st.markdown("## 🤖 AI Evaluation")

    scores_box = st.empty()
    feedback_box = st.empty()
    issues_box = st.empty()
    summary_box = st.empty()

    streamed = {"text": ""}
    parsed = {
        "scores": {},
        "category_feedback": {},
        "issues": [],
        "summary": [],
    }

    def parse_stream(buffer: str):
        sections = re.split(r"\nSECTION: ", buffer)

        for sec in sections:
            sec = sec.strip()
            if not sec:
                continue

            if sec.startswith("SCORES"):
                for line in sec.splitlines()[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        if v.strip().isdigit():
                            parsed["scores"][k.strip()] = int(v.strip())

            elif sec.startswith("CATEGORY_FEEDBACK"):
                for line in sec.splitlines()[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parsed["category_feedback"][k.strip()] = v.strip()

            elif sec.startswith("ISSUE"):
                issue = {}
                for line in sec.splitlines()[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        issue[k.strip()] = v.strip()
                if issue and issue not in parsed["issues"]:
                    parsed["issues"].append(issue)

            elif sec.startswith("SUMMARY"):
                parsed["summary"] = [
                    l.strip("- ").strip()
                    for l in sec.splitlines()[1:]
                    if l.strip()
                ]

    def render():
        if parsed["scores"]:
            scores_box.markdown(
                f"""
                <div class="result-card sentence-text">
                <h3>📈 Scores</h3>
                {parsed["scores"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if parsed["category_feedback"]:
            html = "<div class='result-card sentence-text'><h3>🧠 Feedback</h3>"
            for k, v in parsed["category_feedback"].items():
                html += f"<b>{k.capitalize()}</b>: {v}<br/>"
            html += "</div>"
            feedback_box.markdown(html, unsafe_allow_html=True)

        if parsed["issues"]:
            html = "<div class='result-card sentence-text'><h3>⚠ Issues</h3>"
            for i in parsed["issues"]:
                html += f"""
                <b>{i.get('id')}</b><br/>
                Category: {i.get('category')}<br/>
                Severity: {i.get('severity')}<br/>
                {i.get('message')}<br/>
                <i>{i.get('suggestion_hint')}</i><br/><br/>
                """
            html += "</div>"
            issues_box.markdown(html, unsafe_allow_html=True)

        if parsed["summary"]:
            html = "<div class='result-card sentence-text'><h3>📝 Summary</h3><ul>"
            for s in parsed["summary"]:
                html += f"<li>{s}</li>"
            html += "</ul></div>"
            summary_box.markdown(html, unsafe_allow_html=True)

    def update_ui(chunk):
        print(chunk)
        streamed["text"] += chunk
        parse_stream(streamed["text"])
        render()

    evaluate_streaming(
        transcribe_result,
        question,
        stats,
        on_chunk=update_ui,
    )

    # -------------------------
    # Cleanup
    # -------------------------

    os.remove(temp_path)
    timer_placeholder.empty()
    status_placeholder.success("✅ Processing complete!")

    st.session_state.history.append({
        "audio_bytes": audio_bytes,
        "elapsed_time": elapsed_time,
        "transcript": transcribe_result.get("transcript", ""),
        "stats": stats,
        "evaluation": parsed,
    })