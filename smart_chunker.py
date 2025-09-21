#!/usr/bin/env python3
"""
🧠 Smart Audio Chunker with Voice Activity Detection
==================================================
Split long audio files at natural speech pauses to prevent word corruption
"""

import torch
import torchaudio
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import tempfile
import os

logger = logging.getLogger(__name__)

class SmartAudioChunker:
    """Smart audio chunker that splits at natural speech pauses"""

    def __init__(self,
                 target_chunk_duration: float = 45.0,
                 overlap_duration: float = 8.0,
                 min_silence_duration: float = 0.5,
                 energy_threshold: float = 0.02,
                 sample_rate: int = 16000):
        """
        Initialize smart chunker with pause detection

        Args:
            target_chunk_duration: Target chunk length in seconds (45s for good context)
            overlap_duration: Overlap between chunks in seconds (8s for safety)
            min_silence_duration: Minimum silence duration to consider as pause (0.5s)
            energy_threshold: Energy threshold to detect silence (0.02 works well)
            sample_rate: Audio sample rate (16kHz for Whisper)
        """
        self.target_chunk_duration = target_chunk_duration
        self.overlap_duration = overlap_duration
        self.min_silence_duration = min_silence_duration
        self.energy_threshold = energy_threshold
        self.sample_rate = sample_rate

        # Calculated parameters
        self.target_chunk_samples = int(target_chunk_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.min_silence_samples = int(min_silence_duration * sample_rate)

    def detect_speech_pauses(self, waveform: torch.Tensor, window_size: int = 1024) -> List[int]:
        """
        Detect speech pauses using energy-based voice activity detection

        Args:
            waveform: Audio waveform tensor [1, samples]
            window_size: Window size for energy calculation

        Returns:
            List of sample indices where pauses occur
        """
        if waveform.dim() > 1:
            waveform = waveform.squeeze(0)

        # Calculate short-time energy
        waveform_abs = torch.abs(waveform)

        # Ensure waveform is 1D
        while waveform_abs.dim() > 1:
            waveform_abs = waveform_abs.squeeze()

        # Handle edge case where tensor becomes 0-dimensional
        if waveform_abs.dim() == 0:
            waveform_abs = waveform_abs.unsqueeze(0)

        padding = window_size // 2
        try:
            waveform_padded = torch.nn.functional.pad(waveform_abs, (padding, padding), mode='reflect')
        except Exception as e:
            logger.warning(f"⚠️ Padding failed with reflect mode, using constant: {e}")
            waveform_padded = torch.nn.functional.pad(waveform_abs, (padding, padding), mode='constant', value=0)

        # Compute energy using sliding window
        energy = []
        for i in range(0, len(waveform), window_size // 4):  # 75% overlap
            window_end = min(i + window_size, len(waveform_padded))
            window_energy = torch.mean(waveform_padded[i:window_end])
            energy.append(window_energy.item())

        energy = np.array(energy)

        # Normalize energy
        if len(energy) > 0:
            energy = energy / (np.max(energy) + 1e-8)

        # Find low-energy regions (potential pauses)
        is_silence = energy < self.energy_threshold

        # Find pause boundaries
        pause_boundaries = []
        in_silence = False
        silence_start = 0

        for i, silent in enumerate(is_silence):
            if silent and not in_silence:
                # Start of silence
                silence_start = i
                in_silence = True
            elif not silent and in_silence:
                # End of silence
                silence_duration_samples = (i - silence_start) * (window_size // 4)
                if silence_duration_samples >= self.min_silence_samples:
                    # This is a significant pause
                    pause_center = (silence_start + i) // 2
                    pause_sample = pause_center * (window_size // 4)
                    pause_boundaries.append(min(pause_sample, len(waveform) - 1))
                in_silence = False

        logger.info(f"🔍 Detected {len(pause_boundaries)} speech pauses")
        return pause_boundaries

    def find_optimal_split_points(self, waveform: torch.Tensor) -> List[int]:
        """
        Find optimal split points based on speech pauses and target duration

        Args:
            waveform: Audio waveform tensor

        Returns:
            List of sample indices for optimal splits
        """
        total_samples = waveform.shape[-1]
        total_duration = total_samples / self.sample_rate

        logger.info(f"📊 Audio duration: {total_duration:.1f}s, target chunks: {self.target_chunk_duration}s")

        if total_duration <= self.target_chunk_duration:
            logger.info("📦 Audio shorter than target chunk duration, no splitting needed")
            return [0, total_samples]

        # Detect speech pauses
        pause_boundaries = self.detect_speech_pauses(waveform)

        if not pause_boundaries:
            logger.warning("⚠️ No speech pauses detected, using time-based splitting")
            # Fallback to time-based splitting
            split_points = []
            current_sample = 0
            while current_sample < total_samples:
                split_points.append(current_sample)
                current_sample += self.target_chunk_samples - self.overlap_samples
            split_points.append(total_samples)
            return split_points

        # Find optimal split points near target durations
        split_points = [0]  # Always start at beginning
        current_split = 0

        while current_split < total_samples - self.target_chunk_samples:
            target_next_split = current_split + self.target_chunk_samples - self.overlap_samples

            # Find the closest pause boundary to target
            best_pause = None
            min_distance = float('inf')

            for pause in pause_boundaries:
                if pause > current_split + (self.target_chunk_samples * 0.7):  # At least 70% of target duration
                    distance = abs(pause - target_next_split)
                    if distance < min_distance:
                        min_distance = distance
                        best_pause = pause

            if best_pause is not None:
                split_points.append(best_pause)
                current_split = best_pause
            else:
                # No good pause found, use time-based split
                split_points.append(target_next_split)
                current_split = target_next_split

        split_points.append(total_samples)  # Always end at the end

        # Remove duplicates and sort
        split_points = sorted(list(set(split_points)))

        logger.info(f"✅ Found {len(split_points)-1} optimal split points")
        return split_points

    def create_chunks(self, audio_path: str) -> List[Dict]:
        """
        Create smart audio chunks from file

        Args:
            audio_path: Path to input audio file

        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            logger.info(f"🎬 Smart chunking: {Path(audio_path).name}")

            # Load and preprocess audio
            waveform, original_sample_rate = torchaudio.load(audio_path)

            # Resample to target sample rate
            if original_sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(original_sample_rate, self.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            total_duration = waveform.shape[-1] / self.sample_rate
            logger.info(f"📊 Preprocessed audio: {total_duration:.1f}s, {self.sample_rate}Hz")

            # Find optimal split points
            split_points = self.find_optimal_split_points(waveform)

            # Create chunks with overlap
            chunks = []
            for i in range(len(split_points) - 1):
                start_sample = split_points[i]
                end_sample = split_points[i + 1]

                # Add overlap to previous chunk (except first)
                if i > 0:
                    start_sample = max(0, start_sample - self.overlap_samples)

                # Add overlap to next chunk (except last)
                if i < len(split_points) - 2:
                    end_sample = min(waveform.shape[-1], end_sample + self.overlap_samples)

                chunk_waveform = waveform[:, start_sample:end_sample]
                chunk_duration = (end_sample - start_sample) / self.sample_rate

                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=f'_chunk_{i}.wav') as temp_file:
                    chunk_path = temp_file.name

                # Save chunk
                torchaudio.save(chunk_path, chunk_waveform, self.sample_rate)

                chunk_info = {
                    'index': i,
                    'file_path': chunk_path,
                    'start_time': start_sample / self.sample_rate,
                    'end_time': end_sample / self.sample_rate,
                    'duration': chunk_duration,
                    'start_sample': start_sample,
                    'end_sample': end_sample,
                    'has_overlap_start': i > 0,
                    'has_overlap_end': i < len(split_points) - 2
                }

                chunks.append(chunk_info)
                logger.info(f"📦 Chunk {i}: {chunk_duration:.1f}s ({chunk_info['start_time']:.1f}s - {chunk_info['end_time']:.1f}s)")

            logger.info(f"✅ Created {len(chunks)} smart chunks with pause detection")
            return chunks

        except Exception as e:
            logger.error(f"❌ Error in smart chunking: {e}")
            return []

    def merge_transcripts(self, transcript_chunks: List[Dict]) -> str:
        """
        Intelligently merge overlapping transcripts

        Args:
            transcript_chunks: List of dicts with 'transcript', 'start_time', 'end_time', etc.

        Returns:
            Merged transcript string
        """
        if not transcript_chunks:
            return ""

        if len(transcript_chunks) == 1:
            return transcript_chunks[0].get('transcript', '')

        logger.info(f"🔗 Merging {len(transcript_chunks)} transcript chunks")

        merged_transcript = ""

        for i, chunk in enumerate(transcript_chunks):
            transcript = chunk.get('transcript', '').strip()
            if not transcript:
                continue

            if i == 0:
                # First chunk - add completely
                merged_transcript = transcript
            else:
                # Subsequent chunks - smart overlap handling
                prev_chunk = transcript_chunks[i-1]
                overlap_start = chunk.get('start_time', 0)
                prev_end = prev_chunk.get('end_time', 0)

                if chunk.get('has_overlap_start', False) and overlap_start < prev_end:
                    # There's an overlap - try to merge intelligently
                    merged_transcript = self._merge_overlapping_text(merged_transcript, transcript)
                else:
                    # No overlap - just append
                    merged_transcript += " " + transcript

        final_transcript = merged_transcript.strip()
        logger.info(f"✅ Merged transcript: {len(final_transcript)} characters")
        return final_transcript

    def _merge_overlapping_text(self, text1: str, text2: str) -> str:
        """Merge two potentially overlapping text segments"""
        words1 = text1.split()
        words2 = text2.split()

        if not words1:
            return text2
        if not words2:
            return text1

        # Look for overlap in last few words of text1 with first few words of text2
        max_overlap_check = min(5, len(words1), len(words2))
        best_overlap = 0

        for overlap_size in range(max_overlap_check, 0, -1):
            if words1[-overlap_size:] == words2[:overlap_size]:
                best_overlap = overlap_size
                break

        if best_overlap > 0:
            # Remove overlapping words from text2
            remaining_words = words2[best_overlap:]
            if remaining_words:
                return text1 + " " + " ".join(remaining_words)
            else:
                return text1
        else:
            # No overlap found, just concatenate
            return text1 + " " + text2

    def cleanup_chunks(self, chunks: List[Dict]):
        """Clean up temporary chunk files"""
        for chunk in chunks:
            try:
                file_path = chunk.get('file_path')
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not remove chunk file: {e}")

# Test function
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python smart_chunker.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    chunker = SmartAudioChunker(
        target_chunk_duration=45,
        overlap_duration=8,
        min_silence_duration=0.5,
        energy_threshold=0.02
    )

    print(f"🧠 Testing smart chunker with: {audio_file}")
    chunks = chunker.create_chunks(audio_file)

    print(f"✅ Created {len(chunks)} smart chunks:")
    for chunk in chunks:
        print(f"  Chunk {chunk['index']}: {chunk['duration']:.1f}s "
              f"({chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s)")

    # Clean up
    chunker.cleanup_chunks(chunks)
    print("🧹 Cleaned up chunks")