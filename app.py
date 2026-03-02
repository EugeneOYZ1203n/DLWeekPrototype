# app.py

import os
import re
import tempfile
import time
from html import escape

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import streamlit as st

from question import SAMPLE_PROMPTS, get_last_ollama_debug, get_question
from score_simplified import calculate_extra_stats, evaluate_streaming
from transcribe import transcribe_audio
from styling import (
    configure_page,
    init_theme_state,
    toggle_theme,
    load_css,
    apply_theme
)

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


# -------------------------
# Main App
# -------------------------

configure_page()
init_theme_state()
init_state()
load_css()
apply_theme()

# Sidebar
with st.sidebar:
    st.header("Session")
    theme_label = (
        "Switch to Night Mode"
        if st.session_state.theme == "day"
        else "Switch to Day Mode"
    )
    st.button(theme_label, use_container_width=True, on_click=toggle_theme)

# Prompt selection
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

# Recording
if st.button("Record", use_container_width=True):

    # ----------------------------
    # RECORDING
    # ----------------------------
    timer_placeholder = st.empty()
    status_placeholder = st.empty()

    status_placeholder.info("🎙 Recording... Speak clearly.")

    audio_array, elapsed_time = record_audio(timer_placeholder)

    status_placeholder.info("⏹ Recording complete. Preparing audio...")

    # ----------------------------
    # SAVE TEMP AUDIO
    # ----------------------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, audio_array, SAMPLE_RATE)
        temp_path = tmpfile.name

    with open(temp_path, "rb") as f:
        audio_bytes = f.read()

    # ----------------------------
    # TRANSCRIPTION
    # ----------------------------
    status_placeholder.info("📝 Transcribing audio...")

    with st.spinner("..."):
        transcribe_result = transcribe_audio(temp_path)
        print(transcribe_result)

    # ----------------------------
    # SCORING
    # ----------------------------
    status_placeholder.info("📊 Calculating metrics...")
    
    with st.spinner("..."):
        stats = calculate_extra_stats(transcribe_result)
        print(stats)

    status_placeholder.info("📊 Generating report...")

    status_box = st.empty()

    streamed_text = ""

    def update_ui(piece):
        global streamed_text
        streamed_text += piece
        status_box.markdown(f"```\n{streamed_text}\n```")

    full_output = evaluate_streaming(
        transcribe_result,
        question,
        stats,
        on_chunk=update_ui
    )

    # ----------------------------
    # CLEANUP
    # ----------------------------
    os.remove(temp_path)
    timer_placeholder.empty()

    status_placeholder.success("✅ Processing complete!")

    # ----------------------------
    # SAVE TO HISTORY
    # ----------------------------
    st.session_state.history.append({
        "audio_bytes": audio_bytes,
        "elapsed_time": elapsed_time,
        "transcript": transcribe_result.get("transcript", ""),
        "avg_confidence": transcribe_result.get("avg_confidence", 0.0),
    })

    # Optional: auto-clear progress after short delay
    time.sleep(1)

# History display
for idx, item in enumerate(st.session_state.history, start=1):
    st.markdown(f"### Attempt {idx}")
    st.audio(item["audio_bytes"])

    col1, col2 = st.columns(2)
    with col1:
        st.write(f'Duration: {item["elapsed_time"]:.2f}s')
    with col2:
        st.write(f'Confidence: {item["avg_confidence"]:.2f}')

    st.text_area(
        f"Transcript {idx}",
        value=item["transcript"] or "No speech detected.",
        height=120,
        disabled=True,
        key=f"transcript_{idx}",
    )