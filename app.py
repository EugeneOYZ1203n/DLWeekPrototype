import streamlit as st
import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
from transcribe import transcribe_audio
from question import get_question  # import question generator

SAMPLE_RATE = 16000


def record_audio(duration=5):
    st.info(f"Recording for {duration} seconds...")
    recording = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    return recording


# ----------------------------
# UI
# ----------------------------

st.title("🇯🇵 Japanese Speaking Practice")

# Pull a random question
question = get_question()
st.subheader("Practice Sentence:")
st.write(question)

duration = st.slider("Recording duration (seconds)", 3, 15, 5)

if st.button("🎤 Record & Transcribe"):

    audio = record_audio(duration)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        sf.write(tmpfile.name, audio, SAMPLE_RATE)
        temp_path = tmpfile.name

    st.audio(temp_path)

    transcript, audio_duration, confidence = transcribe_audio(temp_path)

    st.subheader("Transcription")
    st.write(transcript)

    st.subheader("Audio Duration")
    st.write(f"{audio_duration:.2f} seconds")

    st.subheader("Pronunciation Confidence (Rough)")
    st.write(f"{confidence:.2f}")

    os.remove(temp_path)