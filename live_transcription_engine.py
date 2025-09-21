#!/usr/bin/env python3
"""
🎙️ Live Transcription Engine - Real-time High-Accuracy Streaming
================================================================
Intelligent buffering with VAD for live transcription using Mesolitica
"""

import asyncio
import websockets
import json
import numpy as np
import torch
import torchaudio
import webrtcvad
import logging
from collections import deque
from datetime import datetime
import tempfile
import os
from threading import Lock
from typing import Dict, List, Optional, Tuple
import base64

logger = logging.getLogger(__name__)

class LiveTranscriptionEngine:
    """Real-time transcription engine with intelligent buffering and VAD"""

    def __init__(self,
                 transcription_service,  # Mesolitica service instance
                 buffer_duration: float = 15.0,     # Buffer 15 seconds of audio
                 overlap_duration: float = 1.5,     # 1.5 second overlap for continuity
                 vad_aggressiveness: int = 2,       # VAD sensitivity (0-3)
                 sample_rate: int = 16000,
                 chunk_duration_ms: int = 30):      # 30ms chunks for VAD
        """
        Initialize live transcription engine

        Args:
            transcription_service: The Mesolitica transcription service
            buffer_duration: How long to buffer audio before transcribing (15 seconds)
            overlap_duration: Overlap between buffers to prevent word loss (seconds)
            vad_aggressiveness: WebRTC VAD aggressiveness (0=least, 3=most aggressive)
            sample_rate: Audio sample rate (16kHz for Whisper)
            chunk_duration_ms: VAD frame duration (10, 20, or 30ms)
        """
        self.transcription_service = transcription_service
        self.buffer_duration = buffer_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms

        # Calculate buffer sizes
        self.buffer_size = int(buffer_duration * sample_rate)
        self.overlap_size = int(overlap_duration * sample_rate)
        self.chunk_size = int(chunk_duration_ms * sample_rate / 1000)

        # Audio buffer and VAD
        self.audio_buffer = deque(maxlen=self.buffer_size * 2)  # Circular buffer
        self.vad = webrtcvad.Vad(vad_aggressiveness)

        # State management
        self.is_recording = False
        self.speech_detected = False
        self.silence_counter = 0
        self.silence_threshold = 20  # Number of silent frames before triggering transcription

        # Buffer management
        self.current_buffer = []
        self.last_transcription_time = 0
        self.buffer_lock = Lock()

        # Results tracking
        self.partial_results = []
        self.full_transcript = ""

        logger.info(f"🎙️ Live transcription engine initialized:")
        logger.info(f"   Buffer duration: {buffer_duration}s")
        logger.info(f"   Overlap: {overlap_duration}s")
        logger.info(f"   VAD aggressiveness: {vad_aggressiveness}")

    async def start_live_session(self, websocket, path):
        """Handle WebSocket connection for live transcription"""
        logger.info(f"🔗 New live transcription session started")

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get('command')

                    if command == 'start_recording':
                        await self._handle_start_recording(websocket)
                    elif command == 'stop_recording':
                        await self._handle_stop_recording(websocket)
                    elif command == 'audio_chunk':
                        await self._handle_audio_chunk(websocket, data)
                    elif command == 'get_status':
                        await self._handle_get_status(websocket)

                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'message': str(e)
                    }))

        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 Live transcription session ended")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _handle_start_recording(self, websocket):
        """Start live recording session"""
        self.is_recording = True
        self.audio_buffer.clear()
        self.current_buffer = []
        self.partial_results = []
        self.full_transcript = ""
        self.last_transcription_time = datetime.now().timestamp()

        logger.info("🎤 Live recording started")

        await websocket.send(json.dumps({
            'type': 'recording_started',
            'message': 'Live transcription started',
            'timestamp': datetime.now().isoformat()
        }))

    async def _handle_stop_recording(self, websocket):
        """Stop recording and transcribe final buffer"""
        self.is_recording = False

        # Transcribe any remaining audio in buffer
        if len(self.current_buffer) > self.sample_rate:  # At least 1 second of audio
            await self._transcribe_current_buffer(websocket, final=True)

        logger.info("🛑 Live recording stopped")

        await websocket.send(json.dumps({
            'type': 'recording_stopped',
            'transcript': self.full_transcript,
            'message': 'Live transcription completed',
            'timestamp': datetime.now().isoformat()
        }))

    async def _handle_audio_chunk(self, websocket, data):
        """Process incoming audio chunk with VAD and intelligent buffering"""
        if not self.is_recording:
            return

        try:
            # Decode audio data (base64 -> numpy)
            audio_data_b64 = data.get('audio_data')
            if not audio_data_b64:
                return

            audio_bytes = base64.b64decode(audio_data_b64)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            with self.buffer_lock:
                # Add to circular buffer
                self.audio_buffer.extend(audio_np)
                self.current_buffer.extend(audio_np)

                # Process VAD on recent audio
                await self._process_voice_activity(websocket, audio_np)

                # Check if buffer is ready for transcription
                await self._check_buffer_ready(websocket)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    async def _process_voice_activity(self, websocket, audio_chunk: np.ndarray):
        """Process voice activity detection on audio chunk"""
        try:
            # Ensure chunk is correct size for VAD (30ms = 480 samples at 16kHz)
            if len(audio_chunk) < self.chunk_size:
                return

            # Process in VAD-sized frames
            for i in range(0, len(audio_chunk) - self.chunk_size, self.chunk_size):
                frame = audio_chunk[i:i + self.chunk_size]
                frame_bytes = frame.tobytes()

                # VAD detection
                is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)

                if is_speech:
                    self.speech_detected = True
                    self.silence_counter = 0
                else:
                    self.silence_counter += 1

        except Exception as e:
            logger.warning(f"VAD processing error: {e}")

    async def _check_buffer_ready(self, websocket):
        """Check if buffer is ready for transcription"""
        buffer_duration = len(self.current_buffer) / self.sample_rate
        current_time = datetime.now().timestamp()
        time_since_last = current_time - self.last_transcription_time

        # Trigger transcription if:
        # 1. Buffer is full (15 seconds)
        # 2. Silence detected after speech and buffer has content (>3 seconds)
        # 3. Maximum wait time exceeded (18 seconds)

        should_transcribe = False
        reason = ""

        if buffer_duration >= self.buffer_duration:
            should_transcribe = True
            reason = "buffer_full"
        elif (self.silence_counter >= self.silence_threshold and
              self.speech_detected and
              buffer_duration >= 3.0):
            should_transcribe = True
            reason = "speech_end_detected"
        elif time_since_last >= 18.0 and buffer_duration >= 1.0:
            should_transcribe = True
            reason = "max_wait_exceeded"

        if should_transcribe:
            await self._transcribe_current_buffer(websocket, reason=reason)

    async def _transcribe_current_buffer(self, websocket, final: bool = False, reason: str = ""):
        """Transcribe current audio buffer using Mesolitica"""
        if len(self.current_buffer) < self.sample_rate:  # Less than 1 second
            return

        try:
            # Convert buffer to tensor
            audio_tensor = torch.tensor(self.current_buffer, dtype=torch.float32).unsqueeze(0)
            audio_tensor = audio_tensor / 32768.0  # Normalize int16 to float32

            # Save to temporary file for Mesolitica
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_path = temp_file.name

            torchaudio.save(temp_path, audio_tensor, self.sample_rate)

            # Transcribe using Mesolitica service
            logger.info(f"🧠 Transcribing buffer: {len(self.current_buffer)/self.sample_rate:.1f}s ({reason})")

            result = self.transcription_service.transcribe_audio(temp_path, language="ms")

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if result.get('success'):
                transcript = result.get('transcript', '').strip()
                confidence = result.get('confidence', 0)

                if transcript:
                    # Add to results
                    timestamp = datetime.now().isoformat()
                    partial_result = {
                        'transcript': transcript,
                        'confidence': confidence,
                        'timestamp': timestamp,
                        'duration': len(self.current_buffer) / self.sample_rate,
                        'is_final': final
                    }

                    self.partial_results.append(partial_result)

                    # Update full transcript (with smart deduplication)
                    self.full_transcript = self._merge_with_deduplication(
                        self.full_transcript, transcript
                    )

                    # Send partial result to client
                    await websocket.send(json.dumps({
                        'type': 'partial_result',
                        'transcript': transcript,
                        'full_transcript': self.full_transcript,
                        'confidence': confidence,
                        'timestamp': timestamp,
                        'is_final': final,
                        'reason': reason
                    }))

                    logger.info(f"✅ Live transcription: {transcript[:50]}... (conf: {confidence:.2f})")

            # Manage buffer for next transcription
            if not final:
                # Keep overlap for continuity
                overlap_samples = min(self.overlap_size, len(self.current_buffer))
                self.current_buffer = self.current_buffer[-overlap_samples:]

                # Reset state
                self.speech_detected = False
                self.silence_counter = 0
                self.last_transcription_time = datetime.now().timestamp()

        except Exception as e:
            logger.error(f"❌ Live transcription error: {e}")
            await websocket.send(json.dumps({
                'type': 'error',
                'message': f'Transcription error: {str(e)}'
            }))

    def _merge_with_deduplication(self, existing_text: str, new_text: str) -> str:
        """Merge new transcript with existing, removing overlaps"""
        if not existing_text:
            return new_text

        if not new_text:
            return existing_text

        # Simple word-based deduplication
        existing_words = existing_text.split()
        new_words = new_text.split()

        # Look for overlap in last few words of existing with first few words of new
        max_overlap = min(8, len(existing_words), len(new_words))
        best_overlap = 0

        for overlap_size in range(max_overlap, 0, -1):
            if existing_words[-overlap_size:] == new_words[:overlap_size]:
                best_overlap = overlap_size
                break

        if best_overlap > 0:
            # Remove overlapping portion from new text
            remaining_words = new_words[best_overlap:]
            if remaining_words:
                return existing_text + " " + " ".join(remaining_words)
            else:
                return existing_text
        else:
            # No overlap found, add with space
            return existing_text + " " + new_text

    async def _handle_get_status(self, websocket):
        """Send current status to client"""
        status = {
            'type': 'status',
            'is_recording': self.is_recording,
            'buffer_duration': len(self.current_buffer) / self.sample_rate if self.current_buffer else 0,
            'speech_detected': self.speech_detected,
            'silence_counter': self.silence_counter,
            'partial_results_count': len(self.partial_results),
            'full_transcript_length': len(self.full_transcript)
        }

        await websocket.send(json.dumps(status))

# WebSocket server setup
async def start_live_transcription_server(transcription_service, host="localhost", port=8002):
    """Start WebSocket server for live transcription"""
    engine = LiveTranscriptionEngine(transcription_service)

    logger.info(f"🎙️ Starting live transcription server on ws://{host}:{port}")

    async with websockets.serve(engine.start_live_session, host, port):
        logger.info(f"✅ Live transcription server running on ws://{host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    # Test server
    print("🎙️ Live Transcription Engine")
    print("This module provides real-time transcription with intelligent buffering")
    print("Usage: Import and integrate with your Mesolitica service")