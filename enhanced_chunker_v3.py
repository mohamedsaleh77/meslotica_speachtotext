#!/usr/bin/env python3
"""
🎯 Enhanced Audio Processor V3 - Speaker Diarization + Parallel GPU Processing
==============================================================================
Advanced transcription with speaker recognition and intelligent formatting
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import webrtcvad
from collections import defaultdict
import json
from datetime import datetime
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

class EnhancedAudioProcessorV3:
    """Advanced processor with speaker diarization and parallel processing"""

    def __init__(self,
                 max_chunk_duration: float = 20.0,  # Smaller for parallel processing
                 min_chunk_duration: float = 5.0,
                 silence_threshold: float = 0.01,
                 min_silence_duration: float = 0.8,
                 sample_rate: int = 16000,
                 enable_speaker_detection: bool = True,
                 max_parallel_chunks: int = 3):  # Process 3 chunks simultaneously
        """
        Initialize enhanced processor with speaker diarization
        """
        self.max_chunk_duration = max_chunk_duration
        self.min_chunk_duration = min_chunk_duration
        self.silence_threshold = silence_threshold
        self.min_silence_duration = min_silence_duration
        self.sample_rate = sample_rate
        self.enable_speaker_detection = enable_speaker_detection
        self.max_parallel_chunks = max_parallel_chunks

        # VAD for better silence detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2

        # Thread safety for GPU access
        self.gpu_lock = Lock()

        # Calculated parameters
        self.max_chunk_samples = int(max_chunk_duration * sample_rate)
        self.min_chunk_samples = int(min_chunk_duration * sample_rate)
        self.min_silence_samples = int(min_silence_duration * sample_rate)

    def extract_speaker_embeddings(self, waveform: torch.Tensor, chunk_info: Dict) -> np.ndarray:
        """
        Extract speaker embeddings from audio chunk using simple spectral features
        """
        try:
            # Simple approach: Use MFCC features as speaker embeddings
            transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=13,
                melkwargs={"n_mels": 40, "n_fft": 512}
            )

            # Extract MFCC features
            mfcc = transform(waveform)

            # Average pooling across time to get a fixed-size embedding
            embedding = torch.mean(mfcc, dim=2).squeeze().numpy()

            return embedding

        except Exception as e:
            logger.warning(f"⚠️ Could not extract speaker embedding: {e}")
            return np.zeros(13)  # Return zero embedding on failure

    def detect_speakers(self, chunks: List[Dict], waveforms: Dict) -> Dict[int, int]:
        """
        Detect and cluster speakers across chunks
        """
        if not self.enable_speaker_detection:
            return {i: 0 for i in range(len(chunks))}

        logger.info("🎤 Detecting speakers across chunks...")

        embeddings = []
        valid_indices = []

        # Extract embeddings for each chunk
        for i, chunk in enumerate(chunks):
            waveform = waveforms.get(chunk['index'])
            if waveform is not None:
                embedding = self.extract_speaker_embeddings(waveform, chunk)
                if embedding is not None and embedding.any():
                    embeddings.append(embedding)
                    valid_indices.append(i)

        if len(embeddings) < 2:
            # Not enough data for clustering
            return {i: 0 for i in range(len(chunks))}

        # Cluster embeddings to identify speakers
        try:
            embeddings_array = np.array(embeddings)

            # Use Agglomerative Clustering for speaker detection
            max_speakers = min(4, len(embeddings))  # Assume max 4 speakers
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                linkage='average',
                metric='cosine'
            )

            labels = clustering.fit_predict(embeddings_array)

            # Map back to chunk indices
            speaker_map = {i: 0 for i in range(len(chunks))}
            for idx, valid_idx in enumerate(valid_indices):
                speaker_map[valid_idx] = int(labels[idx])

            # Count speakers
            unique_speakers = len(set(labels))
            logger.info(f"✅ Detected {unique_speakers} unique speaker(s)")

            return speaker_map

        except Exception as e:
            logger.warning(f"⚠️ Speaker clustering failed: {e}")
            return {i: 0 for i in range(len(chunks))}

    def detect_silence_regions_vad(self, waveform: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Detect silence using WebRTC VAD for better accuracy
        """
        if waveform.dim() > 1:
            waveform = waveform.mean(dim=0)

        # Convert to int16 for VAD
        waveform_int16 = (waveform * 32767).to(torch.int16).numpy()

        # Process in 30ms frames (WebRTC VAD requirement)
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)

        silence_regions = []
        is_silent = []

        # Process each frame
        for i in range(0, len(waveform_int16) - frame_size, frame_size):
            frame = waveform_int16[i:i+frame_size].tobytes()
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                is_silent.append(not is_speech)
            except:
                is_silent.append(False)

        # Find continuous silence regions
        in_silence = False
        start_frame = 0

        for i, silent in enumerate(is_silent):
            if silent and not in_silence:
                start_frame = i
                in_silence = True
            elif not silent and in_silence:
                duration_samples = (i - start_frame) * frame_size
                if duration_samples >= self.min_silence_samples:
                    start_sample = start_frame * frame_size
                    end_sample = i * frame_size
                    silence_regions.append((start_sample, end_sample))
                in_silence = False

        logger.info(f"🔍 VAD detected {len(silence_regions)} silence regions")
        return silence_regions

    def create_smart_chunks(self, audio_path: str) -> Tuple[List[Dict], Dict[int, torch.Tensor]]:
        """
        Create intelligent chunks optimized for parallel processing
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

            total_samples = waveform.shape[-1]
            total_duration = total_samples / self.sample_rate

            # Use VAD for better silence detection
            silence_regions = self.detect_silence_regions_vad(waveform)

            # Create optimal chunks
            chunks = []
            waveforms = {}
            chunk_points = self._calculate_chunk_points(total_samples, silence_regions)

            for i in range(len(chunk_points) - 1):
                start_sample = chunk_points[i]
                end_sample = chunk_points[i + 1]

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
                waveforms[i] = chunk_waveform

                logger.info(f"📦 Chunk {i}: {chunk_duration:.1f}s "
                          f"({chunk_info['start_time']:.1f}s - {chunk_info['end_time']:.1f}s)")

            return chunks, waveforms

        except Exception as e:
            logger.error(f"❌ Error creating chunks: {e}")
            return [], {}

    def _calculate_chunk_points(self, total_samples: int, silence_regions: List[Tuple[int, int]]) -> List[int]:
        """
        Calculate optimal chunk points based on silence regions
        """
        if total_samples <= self.max_chunk_samples:
            return [0, total_samples]

        chunk_points = [0]
        current_pos = 0

        while current_pos < total_samples - self.min_chunk_samples:
            target_end = min(current_pos + self.max_chunk_samples, total_samples)

            # Find best silence region near target
            best_split = None
            for silence_start, silence_end in silence_regions:
                silence_middle = (silence_start + silence_end) // 2
                if current_pos + self.min_chunk_samples <= silence_middle <= target_end:
                    if best_split is None or abs(silence_middle - target_end) < abs(best_split - target_end):
                        best_split = silence_middle

            if best_split:
                chunk_points.append(best_split)
                current_pos = best_split
            else:
                chunk_points.append(min(target_end, total_samples))
                current_pos = min(target_end, total_samples)

        if chunk_points[-1] < total_samples:
            chunk_points.append(total_samples)

        return sorted(list(set(chunk_points)))

    def process_chunks_parallel(self, chunks: List[Dict], transcribe_func, model) -> List[Dict]:
        """
        Process chunks in parallel on GPU for faster transcription
        """
        logger.info(f"⚡ Processing {len(chunks)} chunks in parallel (max {self.max_parallel_chunks} concurrent)")

        results = [None] * len(chunks)

        with ThreadPoolExecutor(max_workers=self.max_parallel_chunks) as executor:
            # Submit all chunks for processing
            future_to_chunk = {}

            for chunk in chunks:
                future = executor.submit(self._process_single_chunk, chunk, transcribe_func, model)
                future_to_chunk[future] = chunk['index']

            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    results[chunk_idx] = result
                    logger.info(f"✅ Completed chunk {chunk_idx}/{len(chunks)-1}")
                except Exception as e:
                    logger.error(f"❌ Error processing chunk {chunk_idx}: {e}")
                    results[chunk_idx] = {'transcript': '', 'confidence': 0}

        return results

    def _process_single_chunk(self, chunk: Dict, transcribe_func, model) -> Dict:
        """
        Process a single chunk with GPU lock for thread safety
        """
        with self.gpu_lock:
            # Load chunk audio
            waveform, _ = torchaudio.load(chunk['file_path'])

            # Transcribe
            result = transcribe_func(waveform, model)

            # Add chunk metadata
            result['start_time'] = chunk['start_time']
            result['end_time'] = chunk['end_time']
            result['chunk_index'] = chunk['index']

            return result

    def format_transcript_with_speakers(self,
                                       chunks: List[Dict],
                                       transcripts: List[Dict],
                                       speaker_map: Dict[int, int]) -> str:
        """
        Format transcript with speaker labels, timestamps, and paragraphs
        """
        if not transcripts:
            return ""

        logger.info("📝 Formatting transcript with speakers and timestamps")

        # Group by speaker and time
        speaker_segments = defaultdict(list)
        current_speaker = None
        current_segment = []
        segment_start = 0

        for i, transcript in enumerate(transcripts):
            if not transcript or not transcript.get('transcript'):
                continue

            speaker_id = speaker_map.get(i, 0)
            text = transcript.get('transcript', '').strip()

            if not text:
                continue

            # Check if speaker changed
            if speaker_id != current_speaker:
                if current_segment and current_speaker is not None:
                    # Save previous segment
                    speaker_segments[segment_start] = {
                        'speaker': current_speaker,
                        'text': ' '.join(current_segment),
                        'start_time': segment_start,
                        'end_time': transcript.get('start_time', 0)
                    }

                # Start new segment
                current_speaker = speaker_id
                current_segment = [text]
                segment_start = transcript.get('start_time', 0)
            else:
                # Continue current segment
                current_segment.append(text)

        # Add final segment
        if current_segment and current_speaker is not None:
            speaker_segments[segment_start] = {
                'speaker': current_speaker,
                'text': ' '.join(current_segment),
                'start_time': segment_start,
                'end_time': transcripts[-1].get('end_time', 0) if transcripts else 0
            }

        # Format output
        formatted_lines = []
        formatted_lines.append("=" * 80)
        formatted_lines.append("TRANSCRIPTION WITH SPEAKER DIARIZATION")
        formatted_lines.append("=" * 80)
        formatted_lines.append("")

        # Sort by time and format
        for start_time in sorted(speaker_segments.keys()):
            segment = speaker_segments[start_time]

            # Format timestamp
            start_min = int(segment['start_time'] // 60)
            start_sec = int(segment['start_time'] % 60)
            end_min = int(segment['end_time'] // 60)
            end_sec = int(segment['end_time'] % 60)

            timestamp = f"[{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}]"
            speaker_label = f"Speaker {segment['speaker'] + 1}"

            # Add formatted segment
            formatted_lines.append(f"{timestamp} {speaker_label}:")

            # Break text into paragraphs (every 3-4 sentences)
            text = segment['text']
            sentences = re.split(r'(?<=[.!?])\s+', text)

            paragraph = []
            for i, sentence in enumerate(sentences):
                paragraph.append(sentence)
                if (i + 1) % 3 == 0 or i == len(sentences) - 1:
                    formatted_lines.append(' '.join(paragraph))
                    if i < len(sentences) - 1:
                        formatted_lines.append("")  # Paragraph break
                    paragraph = []

            formatted_lines.append("")
            formatted_lines.append("-" * 40)
            formatted_lines.append("")

        # Add summary
        total_speakers = len(set(speaker_map.values()))
        total_duration = transcripts[-1].get('end_time', 0) if transcripts else 0

        formatted_lines.append("")
        formatted_lines.append("=" * 80)
        formatted_lines.append("SUMMARY")
        formatted_lines.append("=" * 80)
        formatted_lines.append(f"Total Speakers: {total_speakers}")
        formatted_lines.append(f"Total Duration: {int(total_duration//60)}:{int(total_duration%60):02d}")
        formatted_lines.append(f"Total Chunks: {len(chunks)}")

        return '\n'.join(formatted_lines)

    def cleanup_chunks(self, chunks: List[Dict]):
        """Clean up temporary chunk files"""
        for chunk in chunks:
            try:
                file_path = chunk.get('file_path')
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"⚠️ Could not remove chunk file: {e}")

# Install required packages function
def install_requirements():
    """Install required packages for enhanced processing"""
    requirements = [
        "webrtcvad",
        "scikit-learn",
        "scipy"
    ]

    print("📦 Installing required packages for enhanced processing...")
    import subprocess
    for package in requirements:
        subprocess.check_call(["pip", "install", package])
    print("✅ Packages installed successfully")

if __name__ == "__main__":
    print("🎯 Enhanced Audio Processor V3 with Speaker Diarization")
    print("Features:")
    print("  - Speaker detection and labeling")
    print("  - Parallel GPU processing")
    print("  - Smart formatting with timestamps")
    print("  - VAD-based silence detection")

    # Check if packages are available
    try:
        import webrtcvad
        import sklearn
    except ImportError:
        print("\n⚠️ Missing required packages. Install with:")
        print("pip install webrtcvad scikit-learn scipy")