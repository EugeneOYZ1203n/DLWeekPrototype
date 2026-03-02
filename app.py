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
    apply_theme,
)

# -----------------------------------
# Constants
# -----------------------------------
SAMPLE_RATE = 16000
MAX_DURATION = 60
SILENCE_THRESHOLD = 0.01
SILENCE_LIMIT = 6

# -----------------------------------
# Session Initialization
# -----------------------------------
def init_state():
    if "practice_question" not in st.session_state:
        st.session_state.practice_question = get_question()
    if "history" not in st.session_state:
        st.session_state.history = []


# -------------------------
# Audio Recording
# -------------------------

def record_audio(timer_placeholder, max_duration=60):
    frames = []
    start_time = time.time()
    last_sound_time = start_time

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())
        rms = np.sqrt(np.mean(indata ** 2))
        if rms > SILENCE_THRESHOLD:
            nonlocal last_sound_time
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

            if elapsed > max_duration:
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

# -----------------------------------
# Sidebar
# -----------------------------------
with st.sidebar:
    st.header("Session")
    theme_label = (
        "Switch to Night Mode"
        if st.session_state.theme == "day"
        else "Switch to Day Mode"
    )
    st.button(theme_label, use_container_width=True, on_click=toggle_theme)

# -----------------------------------
# Prompt Selection
# -----------------------------------
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

# -----------------------------------
# Header
# -----------------------------------
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

# -----------------------------------
# Recording Duration Slider
# -----------------------------------
max_duration = st.slider(
    "Maximum Recording Duration (seconds)",
    min_value=5,
    max_value=120,
    value=60,
    step=1,
)

# -----------------------------------
# Record Button
# -----------------------------------
if st.button("Record", use_container_width=True):

    timer_placeholder = st.empty()
    status_placeholder = st.empty()

    status_placeholder.info("🎙 Recording... Speak clearly.")

    audio_array, elapsed_time = record_audio(timer_placeholder, max_duration=max_duration)

    status_placeholder.info("⏹ Recording complete. Preparing audio...")

    # -----------------------------------
    # Save Temporary Audio
    # -----------------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio_array, SAMPLE_RATE)
        temp_path = tmp.name

    with open(temp_path, "rb") as f:
        audio_bytes = f.read()

    # -----------------------------------
    # Transcription
    # -----------------------------------
    status_placeholder.info("📝 Transcribing audio...")

    with st.spinner("Transcribing..."):
        transcribe_result = transcribe_audio(temp_path)

    transcript_text = transcribe_result.get("transcript", "")

    st.markdown("### 🗒 Transcript")
    st.text_area(
        "Transcript",
        value=transcript_text,
        height=140,
        disabled=True,
    )

    # -----------------------------------
    # Audio Playback & Download
    # -----------------------------------
    st.markdown("### 🎧 Your Recording")

    st.audio(
        audio_bytes,
        format="audio/wav"
    )

    st.download_button(
        label="⬇️ Download Recording",
        data=audio_bytes,
        file_name=temp_path,
        mime="audio/wav",
        use_container_width=True,
    )

    # -----------------------------------
    # Metrics
    # -----------------------------------
    status_placeholder.info("📊 Analyzing speech metrics...")

    with st.spinner("Analyzing speech..."):
        stats = calculate_extra_stats(transcribe_result)

    speech = stats["speech_stats"]
    pauses = stats["pause_stats"]
    fillers = stats["filler_stats"]

    st.markdown("### 📊 Speech Metrics")

    st.markdown(
        f"""
<div class="result-card sentence-text">
<b>Speaking Time:</b> {speech['total_speaking_seconds']}s<br/>
<b>Chars / Minute:</b> {speech['chars_per_minute_estimate']}<br/>
<b>Avg Segment:</b> {speech['avg_segment_duration']}s<br/>
<b>Pause Count:</b> {pauses['pause_count']}<br/>
<b>Long Pauses (>1.5s):</b> {pauses['long_pause_count_1_5s']}<br/>
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

    # -----------------------------------
    # AI Evaluation (Streaming)
    # -----------------------------------
    st.markdown("### 🤖 AI Evaluation")

    scores_box = st.empty()
    feedback_box = st.empty()
    improvement_box = st.empty()
    summary_box = st.empty()

    streamed_text = ""
    parsed = {
        "scores": {},
        "category_feedback": {},
        "category_improvement": {},
        "summary": [],
    }

    def parse_stream(buffer: str):
        sections = re.split(r"SECTION: ", buffer)

        for sec in sections:
            sec = sec.strip()
            sec = sec.replace("Section:", "")
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

            elif sec.startswith("CATEGORY_IMPROVEMENT"):
                for line in sec.splitlines()[1:]:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        parsed["category_improvement"][k.strip()] = v.strip()

            elif sec.startswith("SUMMARY"):
                parsed["summary"] = [
                    l.strip("- ").strip()
                    for l in sec.splitlines()[1:]
                    if l.strip()
                ]

    def render():

        # Scores
        if parsed["scores"]:
            html = "<div class='result-card sentence-text'><h3>📈 Scores</h3>"
            for k, v in parsed["scores"].items():
                html += f"<b>{k.capitalize()}</b>: {v}<br/>"
            html += "</div>"
            scores_box.markdown(html, unsafe_allow_html=True)

        # Feedback
        if parsed["category_feedback"]:
            html = "<div class='result-card sentence-text'><h3>🧠 Feedback</h3>"
            for k, v in parsed["category_feedback"].items():
                html += f"<b>{k.capitalize()}</b>: {v}<br/>"
            html += "</div>"
            feedback_box.markdown(html, unsafe_allow_html=True)

        # Improvements (Suggestions Only)
        if parsed["category_improvement"]:
            html = "<div class='result-card sentence-text'><h3>⚡ Improvements</h3>"
            for k, v in parsed["category_improvement"].items():
                html += f"<b>{k.capitalize()}</b>: {v}<br/>"
            html += "</div>"
            improvement_box.markdown(html, unsafe_allow_html=True)

        # Summary
        if parsed["summary"]:
            html = "<div class='result-card sentence-text'><h3>📝 Summary</h3><ul>"
            for s in parsed["summary"]:
                html += f"<li>{s}</li>"
            html += "</ul></div>"
            summary_box.markdown(html, unsafe_allow_html=True)

    status_placeholder.info("🤖 AI evaluating your answer...")

    with st.spinner("Evaluating..."):
        evaluate_streaming(
            transcribe_result,
            question,
            stats,
            on_chunk=lambda chunk: (
                print(chunk),
                globals().update(streamed_text=streamed_text + chunk),
                parse_stream(streamed_text),
                render(),
            ),
        )

    status_placeholder.success("✅ Processing complete!")

    # -----------------------------------
    # Save History
    # -----------------------------------
    st.session_state.history.append(
        {
            "audio_bytes": audio_bytes,
            "elapsed_time": elapsed_time,
            "transcript": transcript_text,
            "stats": stats,
            "evaluation": parsed,
        }
    )

    # Cleanup
    os.remove(temp_path)
    timer_placeholder.empty()