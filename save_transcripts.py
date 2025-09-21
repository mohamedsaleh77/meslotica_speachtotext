#!/usr/bin/env python3
"""
💾 Add transcript saving functionality to the enhanced server
"""

import os
import json
from datetime import datetime
from pathlib import Path

class TranscriptSaver:
    """Save transcription results to files"""

    def __init__(self, save_dir: str = "transcripts"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_transcript(self, result: dict, audio_filename: str = None) -> str:
        """Save transcript result to JSON file"""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if audio_filename:
                base_name = Path(audio_filename).stem
                filename = f"{timestamp}_{base_name}_transcript.json"
            else:
                filename = f"{timestamp}_live_transcript.json"

            filepath = self.save_dir / filename

            # Add metadata
            result["saved_at"] = datetime.now().isoformat()
            result["audio_filename"] = audio_filename
            result["filepath"] = str(filepath)

            # Save to JSON
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"💾 Transcript saved: {filepath}")
            return str(filepath)

        except Exception as e:
            print(f"❌ Error saving transcript: {e}")
            return None

    def save_transcript_txt(self, transcript: str, audio_filename: str = None) -> str:
        """Save just the transcript text to a TXT file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if audio_filename:
                base_name = Path(audio_filename).stem
                filename = f"{timestamp}_{base_name}_transcript.txt"
            else:
                filename = f"{timestamp}_live_transcript.txt"

            filepath = self.save_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Transcript saved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Audio file: {audio_filename or 'Live recording'}\n")
                f.write("-" * 50 + "\n\n")
                f.write(transcript)

            print(f"📝 Text transcript saved: {filepath}")
            return str(filepath)

        except Exception as e:
            print(f"❌ Error saving text transcript: {e}")
            return None

# Example usage
if __name__ == "__main__":
    saver = TranscriptSaver()

    # Test saving
    test_result = {
        "transcript": "Hai, nama saya Tahira. Anda sedang menyaksikan Easy Malay.",
        "confidence": 0.88,
        "service": "Mesolitica Malaysian Whisper",
        "processing_time": 34.7,
        "word_count": 30
    }

    saver.save_transcript(test_result, "test_audio.mp3")
    saver.save_transcript_txt(test_result["transcript"], "test_audio.mp3")

    print("✅ Test transcripts saved in 'transcripts' folder")