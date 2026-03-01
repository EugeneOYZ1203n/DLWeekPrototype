# app.py
import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import soundfile as sf
import time
from transcribe import transcribe_audio
from question import get_question, SAMPLE_PROMPTS

SAMPLE_RATE = 16000
MAX_DURATION = 60         # seconds
SILENCE_THRESHOLD = 0.01  # RMS threshold for silence
SILENCE_LIMIT = 4         # seconds

st.title("🇯🇵 Japanese Speaking Practice")

# Practice question controls
prompt_options = list(SAMPLE_PROMPTS.keys()) + ["custom"]
style_col, _ = st.columns([1, 3])
with style_col:
    selected_prompt_key = st.selectbox("Prompt style", prompt_options, index=0)

custom_prompt = None
if selected_prompt_key == "custom":
    custom_prompt = st.text_area(
        "Custom prompt",
        value="Create one short Japanese speaking practice sentence. Return only Japanese text.",
    )

if "practice_question" not in st.session_state:
    st.session_state.practice_question = get_question(prompt_key=selected_prompt_key)

if st.button("Generate Question"):
    st.session_state.practice_question = get_question(
        prompt_key=selected_prompt_key,
        custom_prompt=(custom_prompt.strip() if custom_prompt else None),
    )

question = st.session_state.practice_question
st.subheader("Practice Sentence:")
st.write(question)

# Placeholders
timer_placeholder = st.empty()
audio_placeholder = st.empty()
download_placeholder = st.empty()
transcription_placeholder = st.empty()
segments_placeholder = st.empty()
confidence_placeholder = st.empty()
duration_placeholder = st.empty()

# ---------------------------
# Record audio with automatic silence detection
# ---------------------------
def record_audio():
    st.info("🎤 Recording... Speak now!")
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
            timer_placeholder.write(f"⏱ Recording: {int(elapsed)}s")
            time.sleep(0.2)
            if (time.time() - last_sound_time > SILENCE_LIMIT) or elapsed > MAX_DURATION:
                break

    audio_array = np.concatenate(frames, axis=0)
    return audio_array, elapsed

# ---------------------------
# Record button
# ---------------------------
if st.button("🎤 Record"):
    audio_array, elapsed_time = record_audio()

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, audio_array, SAMPLE_RATE)
        temp_path = tmpfile.name

    # Play & download
    audio_placeholder.audio(temp_path)
    download_placeholder.download_button(
        "⬇ Download Recording (.wav)",
        data=open(temp_path, "rb").read(),
        file_name="recording.wav",
        mime="audio/wav"
    )

    # Transcribe
    with st.spinner("Transcribing..."):
        result = transcribe_audio(temp_path)

    # Display full transcript
    transcription_placeholder.subheader("Full Transcription")
    transcription_placeholder.write(result["transcript"])

    # Display overall confidence as mean of segment confidences
    segment_confidences = [np.exp(seg["avg_logprob"]) for seg in result["segments"]]
    avg_confidence = float(np.mean(segment_confidences)) if segment_confidences else 0.0
    confidence_placeholder.subheader("Pronunciation Confidence (Rough)")
    confidence_placeholder.write(f"{avg_confidence:.2f}")

    # Display audio duration
    duration_placeholder.subheader("Audio Duration")
    duration_placeholder.write(f"{elapsed_time:.2f} seconds")

    # Display segments as a table
    if result["segments"]:
        segments_placeholder.subheader("Segments")
        import pandas as pd
        segment_data = pd.DataFrame([{
            "Start (s)": seg["start"],
            "End (s)": seg["end"],
            "Text": seg["text"],
            "Confidence": np.exp(seg["avg_logprob"])
        } for seg in result["segments"]])
        segments_placeholder.dataframe(segment_data)


    import os
    os.remove(temp_path)