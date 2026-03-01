# transcribe.py
# Handles audio transcription using Faster-Whisper

import numpy as np
from faster_whisper import WhisperModel

MODEL_SIZE = "small"
DEVICE = "cpu"  # change to "cuda" if you have GPU


# Load model once globally
model = WhisperModel(MODEL_SIZE, device=DEVICE)


def transcribe_audio(audio_path: str, language: str = "ja"):
    """
    Transcribe an audio file and return:
    - transcript (str)
    - duration (float)
    - avg_confidence (float)
    """

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5
    )

    transcript = ""
    confidences = []

    for segment in segments:
        transcript += segment.text
        confidences.append(np.exp(segment.avg_logprob))

    avg_confidence = np.mean(confidences) if confidences else 0.0

    return transcript.strip(), info.duration, avg_confidence