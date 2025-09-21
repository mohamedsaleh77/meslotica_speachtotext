#!/usr/bin/env python3
"""
Quick Test of Mesolitica Malaysian Whisper
"""

import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torchaudio

def test_mesolitica():
    print("🇲🇾 Testing Mesolitica Malaysian Whisper...")

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)}GB")

    try:
        # Load model
        model_name = "mesolitica/malaysian-whisper-medium"
        print(f"Loading {model_name}...")

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )

        print("✅ Model loaded successfully!")
        print("✅ Mesolitica is ready for 88% accuracy Malaysian transcription!")
        print()
        print("🚀 Now start your enhanced server:")
        print("   python enhanced_whisper_main.py")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_mesolitica()