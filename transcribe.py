# transcribe.py
# Handles audio transcription with Faster-Whisper
from faster_whisper import WhisperModel
from pydub import AudioSegment
from typing import Dict, List

# Load model globally for efficiency
model = WhisperModel("small", device="cpu")  # change to "cuda" if GPU available

def transcribe_audio(audio_path: str, language: str = "ja") -> Dict:
    """
    Transcribes a user audio file to text and returns the duration in seconds.
    
    Parameters:
        audio_path: path to audio file
        language: language code (default: "ja")
    
    Returns:
        {
          "transcript": str,
          "pauses": [{"start": float, "end": float}, ...],
          "total_duration": float
        }
    """
    # Transcribe using Faster-Whisper
    segments, info = model.transcribe(audio_path, language=language)

    segment_list = list(segments)

    # Combine segments into one transcript string
    transcript = " ".join(segment.text.strip() for segment in segment_list).strip()

    # Compute audio duration with pydub (milliseconds -> seconds)
    audio = AudioSegment.from_file(audio_path)
    total_duration = len(audio) / 1000.0

    # Estimate pauses from gaps between adjacent transcription segments.
    pauses: List[Dict[str, float]] = []
    for i in range(1, len(segment_list)):
        prev_seg = segment_list[i - 1]
        curr_seg = segment_list[i]
        gap = float(curr_seg.start) - float(prev_seg.end)
        if gap > 0.15:
            pauses.append({
                "start": round(float(prev_seg.end), 3),
                "end": round(float(curr_seg.start), 3),
            })

    return {
        "transcript": transcript,
        "pauses": pauses,
        "total_duration": round(total_duration, 3),
    }
