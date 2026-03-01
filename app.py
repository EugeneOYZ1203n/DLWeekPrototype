import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import streamlit as st

from question import get_question
from transcribe import transcribe_audio

SAMPLE_RATE = 16000
MAX_DURATION = 60
SILENCE_THRESHOLD = 0.01
SILENCE_LIMIT = 4
CSS_FILE = Path(__file__).parent / "styles" / "app.css"


def init_state():
    if "question" not in st.session_state:
        st.session_state.question = get_question()
    if "history" not in st.session_state:
        st.session_state.history = []
    if "theme" not in st.session_state:
        st.session_state.theme = "day"


def new_prompt():
    st.session_state.question = get_question()
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


def apply_theme():
    if st.session_state.theme == "night":
        st.markdown(
            """
            <style>
            :root {
              --bg-start: #0b1020;
              --bg-mid: #0f172a;
              --bg-end: #111827;
              --text: #f3f4f6;
              --subtext: #9ca3af;
              --border: #374151;
              --brand: #34d399;
              --brand-soft: #102a27;
              --panel: #111827;
              --result-bg: #111827;
              --timer-bg: #1f2937;
              --timer-text: #f9fafb;
              --sidebar-bg: #030712;
              --sidebar-text: #e5e7eb;
              --button-bg: #34d399;
              --button-border: #2bbd87;
              --button-hover: #2bbd87;
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
              --bg-start: #eef4ff;
              --bg-mid: #f6f8fb;
              --bg-end: #f4f7fa;
              --text: #1f2937;
              --subtext: #6b7280;
              --border: #e5e7eb;
              --brand: #10a37f;
              --brand-soft: #e8f8f3;
              --panel: #ffffff;
              --result-bg: #ffffff;
              --timer-bg: #111827;
              --timer-text: #f9fafb;
              --sidebar-bg: #0f172a;
              --sidebar-text: #e5e7eb;
              --button-bg: #10a37f;
              --button-border: #0f8f70;
              --button-hover: #0f8f70;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )


def toggle_theme():
    st.session_state.theme = "night" if st.session_state.theme == "day" else "day"


st.set_page_config(page_title="Japanese Speaking Practice", page_icon="JP", layout="wide")
init_state()
load_css()
apply_theme()

with st.sidebar:
    st.header("Session")
    st.caption("Speaking practice controls")
    st.button("New Sentence", use_container_width=True, on_click=new_prompt)
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
      <div class="sentence-text">{st.session_state.question}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

timer_placeholder = st.empty()

if st.button("Record", use_container_width=True):
    audio_array, elapsed_time = record_audio(timer_placeholder)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, audio_array, SAMPLE_RATE)
        temp_path = tmpfile.name

    with open(temp_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    with st.spinner("Transcribing..."):
        result = transcribe_audio(temp_path)

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

    st.markdown('<div class="result-card"><b>Full Transcription</b></div>', unsafe_allow_html=True)
    st.write(item["transcript"] or "_No speech detected._")

    if not item["segments_df"].empty:
        st.markdown('<div class="result-card"><b>Segments</b></div>', unsafe_allow_html=True)
        st.dataframe(item["segments_df"], use_container_width=True)
