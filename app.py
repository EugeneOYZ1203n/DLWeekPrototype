import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import streamlit as st

from question import get_question,  SAMPLE_PROMPTS
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
            <div class="sentence-text">{st.session_state.practice_question}</div>
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
