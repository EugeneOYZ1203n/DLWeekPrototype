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
    - dict with keys:
        - 'transcript': full concatenated text
        - 'segments': list of segment dicts from Whisper
            Each segment dict contains start, end, text, avg_logprob, etc.
    """

    segments, info = model.transcribe(
        audio_path,
        language=language,
        beam_size=5
    )

    transcript = ""
    segment_list = []

    for seg in segments:
        transcript += seg.text
        # Convert segment object to dict
        segment_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "avg_logprob": seg.avg_logprob,
            "tokens": seg.tokens,
        })

    return {
        "transcript": transcript.strip(),
        "segments": segment_list
    }