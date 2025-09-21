#!/usr/bin/env python3
"""
🧠 Smart Audio Chunker V2 - Advanced Chunking with Deduplication
================================================================
Processes long audio with minimal repetition and maximum accuracy
"""

import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tempfile
import os
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class SmartAudioChunkerV2:
    """Advanced audio chunker with intelligent boundary detection and deduplication"""

    def __init__(self,
                 max_chunk_duration: float = 25.0,  # Safer than 30
                 min_chunk_duration: float = 10.0,   # Don't create tiny chunks
                 boundary_overlap: float = 0.3,      # Minimal overlap only at boundaries
                 silence_threshold: float = 0.01,    # Stricter silence detection
                 min_silence_duration: float = 1.0,  # Require longer silence
                 sample_rate: int = 16000):
        """
        Initialize smart chunker V2

        Args:
            max_chunk_duration: Maximum chunk length in seconds (25s for safety)
            min_chunk_duration: Minimum chunk length to avoid tiny segments
            boundary_overlap: Small overlap only at boundaries (0.3s)
            silence_threshold: Stricter threshold for silence detection
            min_silence_duration: Minimum silence duration for valid boundary (1s)
            sample_rate: Audio sample rate (16kHz for Whisper)
        """
        self.max_chunk_duration = max_chunk_duration
        self.min_chunk_duration = min_chunk_duration
        self.boundary_overlap = boundary_overlap
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate

        # Calculated parameters
        self.max_chunk_samples = int(max_chunk_duration * sample_rate)
        self.min_chunk_samples = int(min_chunk_duration * sample_rate)
        self.overlap_samples = int(boundary_overlap * sample_rate)
        self.min_silence_samples = int(min_silence_duration * sample_rate)

    def detect_silence_boundaries(self, waveform: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Detect complete silence regions (not just low energy)

        Returns:
            List of (start, end) sample indices for silence regions
        """
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)

        # Use absolute values for energy detection
        waveform_abs = torch.abs(waveform)

        # Apply moving average for smoother detection
        kernel_size = int(0.05 * self.sample_rate)  # 50ms window
        if kernel_size > 1:
            kernel = torch.ones(kernel_size) / kernel_size
            waveform_smooth = torch.nn.functional.conv1d(
                waveform_abs.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=kernel_size//2
            ).squeeze()
        else:
            waveform_smooth = waveform_abs

        # Find silence regions
        is_silent = waveform_smooth < self.silence_threshold

        # Find continuous silence regions
        silence_regions = []
        in_silence = False
        start_idx = 0

        for i in range(len(is_silent)):
            if is_silent[i] and not in_silence:
                start_idx = i
                in_silence = True
            elif not is_silent[i] and in_silence:
                if i - start_idx >= self.min_silence_samples:
                    silence_regions.append((start_idx, i))
                in_silence = False

        # Handle if audio ends in silence
        if in_silence and len(is_silent) - start_idx >= self.min_silence_samples:
            silence_regions.append((start_idx, len(is_silent)))

        logger.info(f"🔍 Found {len(silence_regions)} complete silence regions")
        return silence_regions

    def find_optimal_chunk_points(self, waveform: torch.Tensor) -> List[int]:
        """
        Find optimal points to split audio based on silence boundaries
        """
        total_samples = waveform.shape[-1]
        total_duration = total_samples / self.sample_rate

        if total_duration <= self.max_chunk_duration:
            logger.info("📦 Audio fits in single chunk")
            return [0, total_samples]

        # Detect silence boundaries
        silence_regions = self.detect_silence_boundaries(waveform)

        # Create chunks at silence boundaries
        chunk_points = [0]
        current_pos = 0

        while current_pos < total_samples - self.min_chunk_samples:
            # Find next chunk end point
            target_end = min(current_pos + self.max_chunk_samples, total_samples)

            # Look for silence boundary near target
            best_boundary = None
            search_start = current_pos + self.min_chunk_samples
            search_end = target_end

            # Find the best silence region to split at
            for silence_start, silence_end in silence_regions:
                silence_middle = (silence_start + silence_end) // 2
                if search_start <= silence_middle <= search_end:
                    if best_boundary is None or abs(silence_middle - target_end) < abs(best_boundary - target_end):
                        best_boundary = silence_middle

            if best_boundary:
                chunk_points.append(best_boundary)
                current_pos = best_boundary
            else:
                # No good silence found, use max duration
                chunk_points.append(min(current_pos + self.max_chunk_samples, total_samples))
                current_pos = min(current_pos + self.max_chunk_samples, total_samples)

        # Add end point
        if chunk_points[-1] < total_samples:
            chunk_points.append(total_samples)

        # Remove duplicates and sort
        chunk_points = sorted(list(set(chunk_points)))

        logger.info(f"✂️ Created {len(chunk_points)-1} chunk boundaries")
        return chunk_points

    def create_chunks(self, audio_path: str) -> List[Dict]:
        """
        Create smart audio chunks from file
        """
        try:
            logger.info(f"🎬 Processing: {Path(audio_path).name}")

            # Load audio
            waveform, original_sr = torchaudio.load(audio_path)

            # Resample if needed
            if original_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(original_sr, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Find optimal chunk points
            chunk_points = self.find_optimal_chunk_points(waveform)

            # Create chunks with minimal boundary overlap
            chunks = []
            for i in range(len(chunk_points) - 1):
                start_sample = chunk_points[i]
                end_sample = chunk_points[i + 1]

                # Add tiny overlap at boundaries (not for first chunk)
                if i > 0:
                    start_sample = max(0, start_sample - self.overlap_samples)

                chunk_waveform = waveform[:, start_sample:end_sample]
                chunk_duration = (end_sample - start_sample) / self.sample_rate

                # Save chunk
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'_chunk_{i}.wav') as temp_file:
                    chunk_path = temp_file.name

                torchaudio.save(chunk_path, chunk_waveform, self.sample_rate)

                chunk_info = {
                    'index': i,
                    'file_path': chunk_path,
                    'start_time': start_sample / self.sample_rate,
                    'end_time': end_sample / self.sample_rate,
                    'duration': chunk_duration,
                    'start_sample': start_sample,
                    'end_sample': end_sample
                }

                chunks.append(chunk_info)
                logger.info(f"📦 Chunk {i}: {chunk_duration:.1f}s "
                          f"({chunk_info['start_time']:.1f}s - {chunk_info['end_time']:.1f}s)")

            return chunks

        except Exception as e:
            logger.error(f"❌ Error in chunking: {e}")
            return []

    def smart_merge_transcripts(self, transcript_chunks: List[Dict]) -> str:
        """
        Intelligently merge transcripts with advanced deduplication
        """
        if not transcript_chunks:
            return ""

        if len(transcript_chunks) == 1:
            return transcript_chunks[0].get('transcript', '')

        logger.info(f"🔗 Merging {len(transcript_chunks)} chunks with deduplication")

        merged_parts = []

        for i, chunk in enumerate(transcript_chunks):
            transcript = chunk.get('transcript', '').strip()

            if not transcript:
                continue

            # First chunk - add completely
            if i == 0:
                merged_parts.append(transcript)
            else:
                # Smart deduplication for subsequent chunks
                prev_text = merged_parts[-1] if merged_parts else ""
                cleaned_text = self._remove_overlap(prev_text, transcript)

                if cleaned_text:
                    merged_parts.append(cleaned_text)

        # Join all parts
        full_transcript = " ".join(merged_parts)

        # Post-process to remove repetitions
        full_transcript = self._remove_repetitions(full_transcript)

        logger.info(f"✅ Merged transcript: {len(full_transcript)} characters")
        return full_transcript

    def _remove_overlap(self, text1: str, text2: str, threshold: float = 0.8) -> str:
        """
        Remove overlapping content between two text segments
        """
        if not text1 or not text2:
            return text2

        words1 = text1.split()
        words2 = text2.split()

        # Check last N words of text1 against first N words of text2
        max_check = min(10, len(words1) // 2, len(words2) // 2)

        best_overlap = 0
        for check_size in range(max_check, 0, -1):
            # Check if end of text1 matches start of text2
            if words1[-check_size:] == words2[:check_size]:
                best_overlap = check_size
                break

            # Also check with fuzzy matching for slight variations
            seq1 = " ".join(words1[-check_size:])
            seq2 = " ".join(words2[:check_size])
            similarity = SequenceMatcher(None, seq1, seq2).ratio()

            if similarity > threshold:
                best_overlap = check_size
                break

        # Remove overlapping portion from text2
        if best_overlap > 0:
            return " ".join(words2[best_overlap:])

        return text2

    def _remove_repetitions(self, text: str) -> str:
        """
        Remove word/phrase repetitions from text
        """
        words = text.split()
        if len(words) < 2:
            return text

        cleaned_words = []
        i = 0

        while i < len(words):
            word = words[i]

            # Check for repeated words
            repeat_count = 1
            while i + repeat_count < len(words) and words[i + repeat_count].lower() == word.lower():
                repeat_count += 1

            # Keep only one instance if repeated more than 3 times
            if repeat_count > 3:
                cleaned_words.append(word)
                i += repeat_count
            else:
                # Check for repeated phrases (2-3 words)
                phrase_repeated = False
                for phrase_len in [3, 2]:
                    if i + phrase_len <= len(words):
                        phrase = words[i:i+phrase_len]
                        phrase_str = " ".join(phrase).lower()

                        # Count how many times this phrase repeats
                        phrase_count = 1
                        j = i + phrase_len
                        while j + phrase_len <= len(words):
                            next_phrase = " ".join(words[j:j+phrase_len]).lower()
                            if next_phrase == phrase_str:
                                phrase_count += 1
                                j += phrase_len
                            else:
                                break

                        if phrase_count > 2:  # Phrase repeated more than twice
                            cleaned_words.extend(phrase)
                            i = j
                            phrase_repeated = True
                            break

                if not phrase_repeated:
                    cleaned_words.append(word)
                    i += 1

        return " ".join(cleaned_words)

    def cleanup_chunks(self, chunks: List[Dict]):
        """Clean up temporary chunk files"""
        for chunk in chunks:
            try:
                file_path = chunk.get('file_path')
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not remove chunk file: {e}")

# Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python smart_chunker_v2.py <audio_file>")
        sys.exit(1)

    chunker = SmartAudioChunkerV2(
        max_chunk_duration=25,
        min_chunk_duration=10,
        boundary_overlap=0.3,
        silence_threshold=0.01,
        min_silence_duration=1.0
    )

    print(f"🧠 Testing smart chunker V2 with: {sys.argv[1]}")
    chunks = chunker.create_chunks(sys.argv[1])
    print(f"✅ Created {len(chunks)} optimized chunks")