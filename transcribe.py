# transcribe.py
# Handles audio transcription with Faster-Whisper
from faster_whisper import WhisperModel
from pydub import AudioSegment
from typing import Tuple

# Load model globally for efficiency
model = WhisperModel("small", device="cpu")  # change to "cuda" if GPU available

def transcribe_audio(audio_path: str, language: str = "ja") -> Tuple[str, float, float]:
    """
    Transcribes a user audio file to text and returns the duration in seconds.
    
    Parameters:
        audio_path: path to audio file
        language: language code (default: "ja")
    
    Returns:
        transcript: string containing recognized Japanese text
        duration: audio duration in seconds
    """
    # Transcribe using Faster-Whisper
    segments, info = model.transcribe(audio_path, language=language)
    
    # Combine segments into one transcript string
    transcript = " ".join(segment.text for segment in segments)
    
    return {
        "transcript": transcript,
        # Include number of pauses
        # Duration of silences
        # Duration of clip
    }