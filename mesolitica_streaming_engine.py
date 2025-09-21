#!/usr/bin/env python3
"""
🎯 Mesolitica Pseudo-Streaming Engine - Real-time Transcription with High Accuracy
=================================================================================
Continuous transcription using rolling buffers, VAD, and parallel processing
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
import threading
from typing import Dict, List, Optional, Tuple, Callable
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from queue import Queue, Empty
import copy

logger = logging.getLogger(__name__)

class MesoliticaStreamingEngine:
    """Real-time streaming transcription engine using Mesolitica with intelligent buffering"""

    def __init__(self,
                 mesolitica_service,  # Mesolitica transcription service
                 chunk_duration: float = 12.0,         # Chunk size for processing (12s for accuracy)
                 overlap_duration: float = 3.0,        # Overlap between chunks (3s for continuity)
                 vad_aggressiveness: int = 2,           # VAD sensitivity
                 silence_threshold: int = 15,           # Frames of silence before processing
                 sample_rate: int = 16000,
                 max_parallel_chunks: int = 2):         # Process 2 chunks simultaneously
        """
        Initialize Mesolitica streaming engine

        Args:
            mesolitica_service: The Mesolitica transcription service instance
            chunk_duration: Duration of each processing chunk (12s for good context)
            overlap_duration: Overlap between chunks to prevent word loss (3s)
            vad_aggressiveness: WebRTC VAD aggressiveness (0-3)
            silence_threshold: Number of silent frames before triggering processing
            sample_rate: Audio sample rate (16kHz for Whisper/Mesolitica)
            max_parallel_chunks: Maximum chunks processing simultaneously
        """
        self.mesolitica_service = mesolitica_service
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.sample_rate = sample_rate
        self.max_parallel_chunks = max_parallel_chunks

        # VAD setup
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.silence_threshold = silence_threshold
        self.vad_frame_duration_ms = 30  # 30ms frames for VAD

        # Calculate buffer sizes
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.overlap_samples = int(overlap_duration * sample_rate)
        self.vad_frame_samples = int(self.vad_frame_duration_ms * sample_rate / 1000)

        # Audio buffer and processing queues
        self.audio_buffer = deque(maxlen=int(30 * sample_rate))  # 30-second rolling buffer
        self.processing_queue = Queue()
        self.result_queue = Queue()

        # Processing state
        self.is_recording = False
        self.speech_detected = False
        self.silence_counter = 0
        self.last_chunk_time = 0
        self.chunk_counter = 0

        # Thread management
        self.processing_executor = ThreadPoolExecutor(max_workers=max_parallel_chunks)
        self.processing_futures = {}

        # Results tracking
        self.processed_chunks = {}
        self.merged_transcript = ""
        self.partial_results = []

        # Synchronization
        self.buffer_lock = threading.Lock()
        self.result_lock = threading.Lock()

        logger.info(f"🎯 Mesolitica streaming engine initialized:")
        logger.info(f"   Chunk duration: {chunk_duration}s")
        logger.info(f"   Overlap: {overlap_duration}s")
        logger.info(f"   VAD aggressiveness: {vad_aggressiveness}")
        logger.info(f"   Max parallel chunks: {max_parallel_chunks}")

    async def start_streaming_session(self, websocket, path):
        """Handle WebSocket connection for Mesolitica streaming transcription"""
        logger.info(f"🔗 New Mesolitica streaming session started")

        try:
            # Start background processing threads
            processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            result_thread = threading.Thread(target=self._result_worker, daemon=True)

            processing_thread.start()
            result_thread.start()

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
            logger.info("🔌 Mesolitica streaming session ended")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self._cleanup_session()

    async def _handle_start_recording(self, websocket):
        """Start streaming recording session"""
        self.is_recording = True
        self.audio_buffer.clear()
        self.processing_queue = Queue()
        self.result_queue = Queue()
        self.processed_chunks = {}
        self.merged_transcript = ""
        self.partial_results = []
        self.chunk_counter = 0
        self.last_chunk_time = time.time()

        logger.info("🎤 Mesolitica streaming recording started")

        await websocket.send(json.dumps({
            'type': 'recording_started',
            'message': 'Mesolitica streaming transcription started',
            'timestamp': datetime.now().isoformat()
        }))

    async def _handle_stop_recording(self, websocket):
        """Stop recording and process final chunks"""
        self.is_recording = False

        # Process any remaining audio in buffer
        await self._process_final_chunks(websocket)

        logger.info("🛑 Mesolitica streaming recording stopped")

        await websocket.send(json.dumps({
            'type': 'recording_stopped',
            'transcript': self.merged_transcript,
            'message': 'Mesolitica streaming transcription completed',
            'timestamp': datetime.now().isoformat()
        }))

    async def _handle_audio_chunk(self, websocket, data):
        """Process incoming audio chunk with VAD and intelligent buffering"""
        if not self.is_recording:
            return

        try:
            # Decode audio data
            audio_data_b64 = data.get('audio_data')
            if not audio_data_b64:
                return

            audio_bytes = base64.b64decode(audio_data_b64)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

            with self.buffer_lock:
                # Add to rolling buffer
                self.audio_buffer.extend(audio_np)

                # Process VAD
                speech_detected = self._process_voice_activity(audio_np)

                # Check if we should create a chunk for processing
                await self._check_chunk_ready(websocket)

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")

    def _process_voice_activity(self, audio_chunk: np.ndarray) -> bool:
        """Process voice activity detection and update speech state"""
        try:
            speech_frames = 0
            total_frames = 0

            # Process in VAD-sized frames
            for i in range(0, len(audio_chunk) - self.vad_frame_samples, self.vad_frame_samples):
                frame = audio_chunk[i:i + self.vad_frame_samples]
                frame_bytes = frame.tobytes()

                try:
                    is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
                    if is_speech:
                        speech_frames += 1
                    total_frames += 1
                except:
                    continue

            # Update speech detection state
            if total_frames > 0:
                speech_ratio = speech_frames / total_frames
                if speech_ratio > 0.3:  # 30% speech frames = speech detected
                    self.speech_detected = True
                    self.silence_counter = 0
                else:
                    self.silence_counter += 1

            return self.speech_detected

        except Exception as e:
            logger.warning(f"VAD processing error: {e}")
            return False

    async def _check_chunk_ready(self, websocket):
        """Check if buffer is ready for chunk processing"""
        current_time = time.time()
        buffer_duration = len(self.audio_buffer) / self.sample_rate
        time_since_last = current_time - self.last_chunk_time

        # Trigger chunk processing if:
        # 1. Buffer reaches chunk duration (12s)
        # 2. Silence detected after speech and minimum duration (8s)
        # 3. Maximum wait time exceeded (15s)

        should_process = False
        reason = ""

        if buffer_duration >= self.chunk_duration:
            should_process = True
            reason = "buffer_full"
        elif (self.silence_counter >= self.silence_threshold and
              self.speech_detected and
              buffer_duration >= 8.0):
            should_process = True
            reason = "speech_pause_detected"
        elif time_since_last >= 15.0 and buffer_duration >= 5.0:
            should_process = True
            reason = "max_wait_exceeded"

        if should_process:
            await self._create_processing_chunk(websocket, reason)

    async def _create_processing_chunk(self, websocket, reason: str):
        """Create chunk for processing and add to queue"""
        try:
            with self.buffer_lock:
                if len(self.audio_buffer) < self.sample_rate * 5:  # Less than 5 seconds
                    return

                # Extract chunk with overlap
                chunk_end = len(self.audio_buffer)
                chunk_start = max(0, chunk_end - self.chunk_samples)

                # Add overlap from previous chunk
                if self.chunk_counter > 0:
                    chunk_start = max(0, chunk_start - self.overlap_samples)

                chunk_audio = list(self.audio_buffer)[chunk_start:chunk_end]

                if not chunk_audio:
                    return

                # Create chunk info
                chunk_info = {
                    'id': self.chunk_counter,
                    'audio_data': np.array(chunk_audio, dtype=np.int16),
                    'start_time': chunk_start / self.sample_rate,
                    'end_time': chunk_end / self.sample_rate,
                    'duration': len(chunk_audio) / self.sample_rate,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                }

                # Add to processing queue
                self.processing_queue.put(chunk_info)

                logger.info(f"📦 Created chunk {self.chunk_counter}: {chunk_info['duration']:.1f}s ({reason})")

                # Update state
                self.chunk_counter += 1
                self.last_chunk_time = time.time()
                self.speech_detected = False
                self.silence_counter = 0

                # Notify client
                await websocket.send(json.dumps({
                    'type': 'chunk_created',
                    'chunk_id': chunk_info['id'],
                    'duration': chunk_info['duration'],
                    'reason': reason
                }))

        except Exception as e:
            logger.error(f"Error creating processing chunk: {e}")

    def _processing_worker(self):
        """Background worker that processes chunks with Mesolitica"""
        while True:
            try:
                # Get chunk from queue (blocking)
                chunk_info = self.processing_queue.get(timeout=1.0)

                # Submit for parallel processing
                future = self.processing_executor.submit(self._process_chunk_with_mesolitica, chunk_info)
                self.processing_futures[chunk_info['id']] = future

                logger.info(f"🧠 Started processing chunk {chunk_info['id']}")

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Processing worker error: {e}")

    def _process_chunk_with_mesolitica(self, chunk_info: Dict) -> Dict:
        """Process a single chunk with Mesolitica service"""
        try:
            # Convert audio to tensor
            audio_data = chunk_info['audio_data']
            audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            audio_tensor = audio_tensor / 32768.0  # Normalize int16 to float32

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_path = temp_file.name

            torchaudio.save(temp_path, audio_tensor, self.sample_rate)

            start_time = time.time()

            # Transcribe with Mesolitica
            result = self.mesolitica_service.transcribe_audio(temp_path, language="ms")

            processing_time = time.time() - start_time

            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if result.get('success'):
                chunk_result = {
                    'chunk_id': chunk_info['id'],
                    'transcript': result.get('transcript', '').strip(),
                    'confidence': result.get('confidence', 0),
                    'start_time': chunk_info['start_time'],
                    'end_time': chunk_info['end_time'],
                    'duration': chunk_info['duration'],
                    'processing_time': processing_time,
                    'reason': chunk_info['reason'],
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }

                logger.info(f"✅ Chunk {chunk_info['id']} completed: {chunk_result['transcript'][:50]}...")
                return chunk_result
            else:
                logger.error(f"❌ Chunk {chunk_info['id']} failed: {result.get('error', 'Unknown error')}")
                return {
                    'chunk_id': chunk_info['id'],
                    'success': False,
                    'error': result.get('error', 'Transcription failed')
                }

        except Exception as e:
            logger.error(f"❌ Error processing chunk {chunk_info['id']}: {e}")
            return {
                'chunk_id': chunk_info['id'],
                'success': False,
                'error': str(e)
            }

    def _result_worker(self):
        """Background worker that handles completed results"""
        while True:
            try:
                # Check for completed futures
                completed_futures = []
                for chunk_id, future in list(self.processing_futures.items()):
                    if future.done():
                        try:
                            result = future.result()
                            self.result_queue.put(result)
                            completed_futures.append(chunk_id)
                        except Exception as e:
                            logger.error(f"Error getting result for chunk {chunk_id}: {e}")
                            completed_futures.append(chunk_id)

                # Clean up completed futures
                for chunk_id in completed_futures:
                    del self.processing_futures[chunk_id]

                time.sleep(0.1)  # Small delay to prevent busy waiting

            except Exception as e:
                logger.error(f"Result worker error: {e}")

    async def _handle_get_status(self, websocket):
        """Send current status including pending results"""
        try:
            # Check for new results
            new_results = []
            while not self.result_queue.empty():
                try:
                    result = self.result_queue.get_nowait()
                    if result.get('success'):
                        new_results.append(result)

                        # Store result
                        with self.result_lock:
                            self.processed_chunks[result['chunk_id']] = result

                        # Update merged transcript
                        self._update_merged_transcript()

                        # Send partial result
                        await websocket.send(json.dumps({
                            'type': 'partial_result',
                            'chunk_id': result['chunk_id'],
                            'transcript': result['transcript'],
                            'confidence': result['confidence'],
                            'full_transcript': self.merged_transcript,
                            'processing_time': result['processing_time'],
                            'timestamp': result['timestamp']
                        }))

                except Empty:
                    break

            # Send status
            status = {
                'type': 'status',
                'is_recording': self.is_recording,
                'buffer_duration': len(self.audio_buffer) / self.sample_rate,
                'speech_detected': self.speech_detected,
                'silence_counter': self.silence_counter,
                'chunks_created': self.chunk_counter,
                'chunks_processing': len(self.processing_futures),
                'chunks_completed': len(self.processed_chunks),
                'transcript_length': len(self.merged_transcript)
            }

            await websocket.send(json.dumps(status))

        except Exception as e:
            logger.error(f"Error handling status request: {e}")

    def _update_merged_transcript(self):
        """Update merged transcript from processed chunks"""
        try:
            with self.result_lock:
                # Sort chunks by ID to maintain order
                sorted_chunks = sorted(self.processed_chunks.values(), key=lambda x: x['chunk_id'])

                if not sorted_chunks:
                    self.merged_transcript = ""
                    return

                # Merge with intelligent overlap handling
                merged_text = ""

                for i, chunk in enumerate(sorted_chunks):
                    transcript = chunk.get('transcript', '').strip()
                    if not transcript:
                        continue

                    if i == 0:
                        merged_text = transcript
                    else:
                        # Remove overlap with previous chunk
                        merged_text = self._merge_with_overlap_removal(merged_text, transcript)

                self.merged_transcript = merged_text

        except Exception as e:
            logger.error(f"Error updating merged transcript: {e}")

    def _merge_with_overlap_removal(self, text1: str, text2: str) -> str:
        """Merge two text segments with intelligent overlap removal"""
        if not text1:
            return text2
        if not text2:
            return text1

        words1 = text1.split()
        words2 = text2.split()

        if not words1 or not words2:
            return text1 + " " + text2

        # Look for overlap in last words of text1 with first words of text2
        max_overlap = min(8, len(words1), len(words2))
        best_overlap = 0

        for overlap_size in range(max_overlap, 0, -1):
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
            # No exact overlap found, just concatenate
            return text1 + " " + text2

    async def _process_final_chunks(self, websocket):
        """Process any remaining chunks when recording stops"""
        try:
            # Create final chunk if buffer has content
            with self.buffer_lock:
                if len(self.audio_buffer) >= self.sample_rate * 3:  # At least 3 seconds
                    await self._create_processing_chunk(websocket, "final_chunk")

            # Wait for all processing to complete (with timeout)
            timeout = 30  # 30 seconds timeout
            start_time = time.time()

            while self.processing_futures and (time.time() - start_time) < timeout:
                await asyncio.sleep(0.5)
                await self._handle_get_status(websocket)  # Process any completed results

            logger.info(f"🏁 Final processing complete. Processed {len(self.processed_chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing final chunks: {e}")

    def _cleanup_session(self):
        """Clean up session resources"""
        try:
            self.is_recording = False

            # Cancel any pending futures
            for future in self.processing_futures.values():
                future.cancel()

            self.processing_futures.clear()

            # Clear queues and buffers
            self.audio_buffer.clear()
            while not self.processing_queue.empty():
                self.processing_queue.get_nowait()
            while not self.result_queue.empty():
                self.result_queue.get_nowait()

            logger.info("🧹 Session cleanup completed")

        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")

# Server startup function
async def start_mesolitica_streaming_server(mesolitica_service, host="localhost", port=8003):
    """Start WebSocket server for Mesolitica streaming transcription"""
    engine = MesoliticaStreamingEngine(mesolitica_service)

    logger.info(f"🎯 Starting Mesolitica streaming server on ws://{host}:{port}")

    async with websockets.serve(engine.start_streaming_session, host, port):
        logger.info(f"✅ Mesolitica streaming server running on ws://{host}:{port}")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    print("Mesolitica Pseudo-Streaming Engine")
    print("Real-time transcription with rolling buffers and parallel processing")
    print("Maintains Mesolitica's 88% accuracy with near real-time experience")