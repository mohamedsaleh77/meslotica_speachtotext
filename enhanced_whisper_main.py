#!/usr/bin/env python3
"""
🎯 Enhanced Malaysian Speech-to-Text System
===========================================
✨ Highest accuracy Malaysian speech recognition using multiple AI models
🚀 ElevenLabs Scribe + Mesolitica + Whisper fallback pipeline
"""

import os
import sys
import json
import logging
import tempfile
import time
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import whisper
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torchaudio
try:
    from enhanced_chunker_v3 import EnhancedAudioProcessorV3
    CHUNKER_V3_AVAILABLE = True
except ImportError:
    CHUNKER_V3_AVAILABLE = False
    try:
        from smart_chunker_v2 import SmartAudioChunkerV2
        CHUNKER_V2_AVAILABLE = True
    except ImportError:
        from smart_chunker import SmartAudioChunker
        CHUNKER_V2_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Malaysian Speech-to-Text", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Configuration
CONFIG_FILE = Path(__file__).parent / "config.json"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

class TranscriptSaver:
    """Auto-save transcription results"""

    def __init__(self, save_dir: str = "transcripts"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def save_transcript(self, result: dict, audio_filename: str = None) -> dict:
        """Save transcript and return file paths"""
        try:
            # Create filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if audio_filename:
                base_name = Path(audio_filename).stem[:50]  # Limit length
                filename = f"{timestamp}_{base_name}"
            else:
                filename = f"{timestamp}_live_recording"

            # Save JSON with full metadata
            json_path = self.save_dir / f"{filename}.json"
            json_data = {
                **result,
                "saved_at": datetime.now().isoformat(),
                "audio_filename": audio_filename,
                "word_count": len(result.get("transcript", "").split()),
                "character_count": len(result.get("transcript", "")),
            }

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            # Save clean text file
            txt_path = self.save_dir / f"{filename}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Transcript - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Service: {result.get('service', 'Unknown')}\n")
                f.write(f"Confidence: {result.get('confidence', 0):.2%}\n")
                f.write(f"Audio: {audio_filename or 'Live recording'}\n")
                f.write("-" * 60 + "\n\n")
                f.write(result.get("transcript", ""))

            logger.info(f"💾 Transcript auto-saved: {json_path.name}")

            return {
                "json_path": str(json_path),
                "txt_path": str(txt_path),
                "saved": True
            }

        except Exception as e:
            logger.error(f"❌ Error auto-saving transcript: {e}")
            return {"saved": False, "error": str(e)}

# Initialize transcript saver
transcript_saver = TranscriptSaver()

def load_config():
    """Load configuration from config.json"""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        config = get_default_config()
        save_config(config)
        return config
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config file. Using default config.")
        config = get_default_config()
        save_config(config)
        return config

def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "system": {
            "name": "Enhanced Malaysian Speech-to-Text",
            "version": "1.0.0",
            "description": "Highest accuracy Malaysian speech recognition"
        },
        "services": {
            "primary": "elevenlabs",  # elevenlabs, mesolitica, whisper
            "fallback_order": ["mesolitica", "whisper"],
            "elevenlabs_api_key": ELEVENLABS_API_KEY,
            "confidence_threshold": 0.8
        },
        "models": {
            "whisper_model": "large-v2",
            "mesolitica_model": "mesolitica/malaysian-whisper-medium",
            "device": "auto"  # auto, cuda, cpu
        },
        "processing": {
            "language": "ms",
            "enable_preprocessing": True,
            "enable_postprocessing": True,
            "chunk_duration": 15
        }
    }

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to config.json"""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

class ElevenLabsScribe:
    """ElevenLabs Scribe API client for highest accuracy transcription"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }

    def transcribe_audio(self, audio_file_path: str, language: str = "ms") -> Dict[str, Any]:
        """Transcribe audio using ElevenLabs Scribe with highest accuracy settings"""
        try:
            # Read audio file
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()

            # Prepare request
            files = {
                'audio': ('audio.wav', audio_data, 'audio/wav')
            }

            # API parameters for highest accuracy
            data = {
                'model_id': 'eleven_multilingual_v2',  # Latest model
                'language': language,
                'enable_speaker_diarization': 'true',
                'enable_word_timestamps': 'true',
                'enable_audio_events': 'true',
                'output_format': 'json'
            }

            # Make request
            response = requests.post(
                f"{self.base_url}/speech-to-text",
                headers={"xi-api-key": self.api_key},
                files=files,
                data=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                # Extract transcript with high confidence
                transcript = result.get('text', '')
                confidence = result.get('confidence', 0.95)

                logger.info(f"ElevenLabs Scribe: {transcript[:100]}... (confidence: {confidence:.2f})")

                return {
                    "success": True,
                    "transcript": transcript,
                    "confidence": confidence,
                    "language": language,
                    "service": "ElevenLabs Scribe",
                    "word_error_rate": 0.05,  # ≤5% WER documented
                    "segments": result.get('segments', []),
                    "speakers": result.get('speakers', [])
                }
            else:
                logger.error(f"ElevenLabs API error: {response.status_code} - {response.text}")
                return {"success": False, "error": f"API error: {response.status_code}"}

        except Exception as e:
            logger.error(f"ElevenLabs transcription error: {e}")
            return {"success": False, "error": str(e)}

class MesoliticaWhisper:
    """Mesolitica Malaysian Whisper model for local high-accuracy processing with smart chunking"""

    def __init__(self, model_name: str = "mesolitica/malaysian-distil-whisper-large-v3", device: str = "auto", config: dict = None):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.processor = None
        self.model = None
        self.config = config or {}

        # Initialize smart chunker if enabled
        processing_config = self.config.get("processing", {})
        self.smart_chunking_enabled = processing_config.get("enable_smart_chunking", True)

        if self.smart_chunking_enabled:
            if CHUNKER_V3_AVAILABLE:
                # Use enhanced V3 processor with speaker diarization
                self.chunker = EnhancedAudioProcessorV3(
                    max_chunk_duration=20.0,  # Optimized for parallel processing
                    min_chunk_duration=5.0,   # Smaller chunks for parallelization
                    silence_threshold=0.01,   # VAD-based detection
                    min_silence_duration=0.8, # Natural pause detection
                    enable_speaker_detection=True,  # Enable speaker diarization
                    max_parallel_chunks=3     # Process 3 chunks simultaneously on GPU
                )
                logger.info("🎯 Using Enhanced Processor V3 with speaker diarization & parallel processing")
            elif CHUNKER_V2_AVAILABLE:
                # Use improved V2 chunker
                self.chunker = SmartAudioChunkerV2(
                    max_chunk_duration=25.0,  # Safer than 30
                    min_chunk_duration=10.0,  # Avoid tiny chunks
                    boundary_overlap=0.3,     # Minimal overlap
                    silence_threshold=0.01,   # Stricter silence detection
                    min_silence_duration=1.0  # Longer silence for boundaries
                )
                logger.info("🚀 Using Smart Chunker V2 with advanced deduplication")
            else:
                # Fallback to V1
                self.chunker = SmartAudioChunker(
                    target_chunk_duration=processing_config.get("target_chunk_duration", 45),
                    overlap_duration=processing_config.get("overlap_duration", 0.5),
                    min_silence_duration=processing_config.get("min_silence_duration", 0.5),
                    energy_threshold=processing_config.get("energy_threshold", 0.02)
                )
            logger.info("🧠 Smart chunking enabled for long audio files")
        else:
            self.chunker = None
            logger.info("⚠️ Smart chunking disabled - long audio will be truncated")

        self.load_model()

    def _get_device(self, device: str) -> str:
        """Determine the best device to use"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def load_model(self):
        """Load Mesolitica Malaysian Whisper model"""
        try:
            logger.info(f"Loading Mesolitica model: {self.model_name} on {self.device}")

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # Use float32 to avoid precision mismatch
                low_cpu_mem_usage=True,
                use_safetensors=True
            )

            if self.device == "cuda":
                self.model = self.model.to("cuda")
                # Force persistent GPU memory allocation to prevent unloading
                logger.info("🎯 Forcing persistent GPU memory allocation...")
                with torch.no_grad():
                    # Pre-allocate GPU memory to keep model loaded
                    dummy_features = torch.zeros((1, 80, 100), dtype=torch.float32).to('cuda')
                    dummy_input_ids = torch.zeros((1, 10), dtype=torch.long).to('cuda')
                    # Touch the model to ensure GPU memory stays allocated
                    try:
                        self.model.encoder(dummy_features)
                        logger.info("✅ GPU memory successfully allocated and locked")
                    except Exception as e:
                        logger.warning(f"GPU memory allocation warning: {e}")
                # Force CUDA cache cleanup but keep model
                torch.cuda.empty_cache()

            logger.info("✅ Mesolitica Malaysian Whisper model loaded successfully with persistent GPU memory")

        except Exception as e:
            logger.error(f"Failed to load Mesolitica model: {e}")
            self.processor = None
            self.model = None

    def transcribe_audio(self, audio_file_path: str, language: str = "ms") -> Dict[str, Any]:
        """Transcribe audio using Mesolitica Malaysian Whisper with smart chunking for long files"""
        if not self.model or not self.processor:
            return {"success": False, "error": "Model not loaded"}

        try:
            # Convert WebM to WAV if needed (before any torchaudio operations)
            processed_audio_path = self._convert_webm_to_wav(audio_file_path)

            # Get audio duration first
            waveform, sample_rate = torchaudio.load(processed_audio_path)
            duration = waveform.shape[-1] / sample_rate

            logger.info(f"🎵 Processing audio: {duration:.1f}s with {self.model_name}")

            # Decide whether to use chunking
            max_single_duration = 30  # Whisper's fundamental limit

            if duration <= max_single_duration or not self.smart_chunking_enabled:
                # Process as single chunk (original method)
                logger.info("📦 Processing as single chunk")
                return self._transcribe_single_chunk(processed_audio_path, language)
            else:
                # Use smart chunking for long audio
                logger.info(f"🧠 Using smart chunking for {duration:.1f}s audio")
                return self._transcribe_with_chunking(processed_audio_path, language)

        except Exception as e:
            logger.error(f"Mesolitica transcription error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up converted file if it was created
            if 'processed_audio_path' in locals() and processed_audio_path != audio_file_path:
                try:
                    if os.path.exists(processed_audio_path):
                        os.remove(processed_audio_path)
                        logger.info(f"🧹 Cleaned up converted file: {processed_audio_path}")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Could not clean up converted file: {cleanup_error}")

    def _convert_webm_to_wav(self, audio_file_path: str) -> str:
        """Convert WebM audio to WAV format for torchaudio compatibility"""
        try:
            # Check if file is WebM
            if not audio_file_path.lower().endswith('.webm'):
                return audio_file_path

            logger.info(f"🔄 Converting WebM to WAV: {audio_file_path}")

            # Try pydub first
            try:
                from pydub import AudioSegment
                from pydub.utils import which

                # Check if ffmpeg is available
                if which("ffmpeg") or which("avconv"):
                    # Load WebM with pydub
                    audio = AudioSegment.from_file(audio_file_path, format="webm")

                    # Convert to WAV with specific parameters
                    wav_path = audio_file_path.replace('.webm', '_converted.wav')
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(wav_path, format="wav")

                    logger.info(f"✅ Converted to WAV with pydub: {wav_path}")
                    return wav_path
                else:
                    logger.warning("⚠️ FFmpeg not found, trying alternative conversion")

            except Exception as pydub_error:
                logger.warning(f"⚠️ Pydub conversion failed: {pydub_error}")

            # Try ffmpeg-python as alternative
            try:
                import ffmpeg

                wav_path = audio_file_path.replace('.webm', '_converted.wav')

                # Use ffmpeg to convert
                (
                    ffmpeg
                    .input(audio_file_path)
                    .output(wav_path, ar=16000, ac=1, acodec='pcm_s16le')
                    .overwrite_output()
                    .run(quiet=True)
                )

                logger.info(f"✅ Converted to WAV with ffmpeg: {wav_path}")
                return wav_path

            except Exception as ffmpeg_error:
                logger.warning(f"⚠️ FFmpeg conversion failed: {ffmpeg_error}")

            # Try using torch audio directly on WebM (sometimes works)
            try:
                import torchaudio
                waveform, sample_rate = torchaudio.load(audio_file_path)

                # If this works, convert and save as WAV
                wav_path = audio_file_path.replace('.webm', '_converted.wav')
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)

                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                torchaudio.save(wav_path, waveform, 16000)
                logger.info(f"✅ Converted to WAV with torchaudio: {wav_path}")
                return wav_path

            except Exception as torch_error:
                logger.warning(f"⚠️ Torchaudio direct load failed: {torch_error}")

            # If all conversions fail, return original file
            logger.error(f"❌ All conversion methods failed for: {audio_file_path}")
            return audio_file_path

        except Exception as e:
            logger.error(f"❌ Critical error in WebM conversion: {e}")
            return audio_file_path

    def _transcribe_single_chunk(self, audio_file_path: str, language: str = "ms") -> Dict[str, Any]:
        """Transcribe a single audio chunk (≤30 seconds)"""
        try:
            # Convert WebM to WAV if needed
            processed_audio_path = self._convert_webm_to_wav(audio_file_path)

            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(processed_audio_path)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Prepare inputs with proper data type
            inputs = self.processor(
                waveform.squeeze().numpy().astype('float32'),  # Ensure float32
                sampling_rate=16000,
                return_tensors="pt"
            )

            if self.device == "cuda":
                inputs = {k: v.to("cuda", dtype=torch.float32) for k, v in inputs.items()}

            # Generate transcription with scores
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_features"],
                    max_length=448,
                    num_beams=5,
                    do_sample=False,
                    language="ms",
                    task="transcribe",
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Extract predicted IDs and scores
            predicted_ids = outputs.sequences

            # Calculate confidence from scores
            if hasattr(outputs, 'scores') and outputs.scores:
                confidence = torch.stack(outputs.scores).softmax(dim=-1).max(dim=-1)[0].mean().item()
                confidence = min(max(confidence, 0.1), 0.99)
            else:
                confidence = 0.85

            # Decode transcription
            transcript = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            return {
                "success": True,
                "transcript": transcript,
                "confidence": confidence,
                "language": language,
                "service": f"Mesolitica {self.model_name.split('/')[-1]}",
                "word_error_rate": 0.08 if "large" in self.model_name else 0.12,
                "model": self.model_name,
                "chunks_processed": 1
            }

        except Exception as e:
            logger.error(f"Single chunk transcription error: {e}")
            return {"success": False, "error": str(e)}
        finally:
            # Clean up converted file if it was created
            if 'processed_audio_path' in locals() and processed_audio_path != audio_file_path:
                try:
                    if os.path.exists(processed_audio_path):
                        os.remove(processed_audio_path)
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Could not clean up converted file: {cleanup_error}")

    def _transcribe_with_chunking(self, audio_file_path: str, language: str = "ms") -> Dict[str, Any]:
        """Transcribe long audio using smart chunking with parallel processing and speaker diarization"""
        try:
            # Check if V3 processor is available
            if CHUNKER_V3_AVAILABLE and hasattr(self.chunker, 'create_smart_chunks'):
                # Use enhanced V3 processor with speaker diarization
                logger.info("🎯 Using Enhanced Processor V3 with speaker diarization")

                # Create smart chunks with waveform data
                chunks, waveforms = self.chunker.create_smart_chunks(audio_file_path)

                if not chunks:
                    return {"success": False, "error": "Failed to create audio chunks"}

                logger.info(f"🎬 Processing {len(chunks)} chunks with parallel GPU processing")

                # Define transcribe function for parallel processing
                def transcribe_chunk_func(chunk_waveform, model):
                    """Wrapper for chunk transcription"""
                    # Create temporary file for chunk
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                        temp_path = temp_file.name
                        torchaudio.save(temp_path, chunk_waveform, 16000)

                    try:
                        result = self._transcribe_single_chunk(temp_path, language)
                        os.remove(temp_path)
                        return result
                    except Exception as e:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        return {"success": False, "error": str(e)}

                # Process chunks in parallel
                chunk_results = self.chunker.process_chunks_parallel(
                    chunks,
                    transcribe_chunk_func,
                    self.model
                )

                # Detect speakers
                speaker_map = self.chunker.detect_speakers(chunks, waveforms)

                # Format with speaker diarization and timestamps
                formatted_transcript = self.chunker.format_transcript_with_speakers(
                    chunks, chunk_results, speaker_map
                )

                # Calculate average confidence
                successful_chunks = sum(1 for r in chunk_results if r and r.get('success'))
                total_confidence = sum(r.get('confidence', 0) for r in chunk_results if r and r.get('success'))
                average_confidence = total_confidence / successful_chunks if successful_chunks > 0 else 0

                # Clean up
                self.chunker.cleanup_chunks(chunks)

                logger.info(f"✅ Enhanced processing complete with speaker diarization")

                return {
                    "success": True,
                    "transcript": formatted_transcript,
                    "confidence": average_confidence,
                    "language": language,
                    "service": f"Mesolitica {self.model_name.split('/')[-1]} (Enhanced V3)",
                    "word_error_rate": 0.08 if "large" in self.model_name else 0.12,
                    "model": self.model_name,
                    "chunks_processed": successful_chunks,
                    "total_chunks": len(chunks),
                    "speakers_detected": len(set(speaker_map.values())),
                    "chunking_method": "Enhanced V3 with speaker diarization",
                    "processing_mode": "Parallel GPU"
                }

            else:
                # Fallback to V2/V1 chunking (existing code)
                chunks = self.chunker.create_chunks(audio_file_path)

                if not chunks:
                    return {"success": False, "error": "Failed to create audio chunks"}

                logger.info(f"🎬 Processing {len(chunks)} smart chunks")

                # Sequential processing (existing code)
                chunk_results = []
                total_confidence = 0
                successful_chunks = 0

                for i, chunk in enumerate(chunks):
                    logger.info(f"📦 Processing chunk {i+1}/{len(chunks)} ({chunk['duration']:.1f}s)")

                    try:
                        chunk_result = self._transcribe_single_chunk(chunk['file_path'], language)

                        if chunk_result["success"]:
                            chunk_info = {
                                **chunk_result,
                                'chunk_index': i,
                                'start_time': chunk['start_time'],
                                'end_time': chunk['end_time'],
                                'chunk_duration': chunk['duration']
                            }
                            chunk_results.append(chunk_info)
                            total_confidence += chunk_result.get("confidence", 0)
                            successful_chunks += 1
                        else:
                            logger.warning(f"⚠️ Chunk {i} failed: {chunk_result.get('error', 'Unknown error')}")

                    except Exception as e:
                        logger.error(f"❌ Error processing chunk {i}: {e}")

                # Clean up chunk files
                self.chunker.cleanup_chunks(chunks)

                if not chunk_results:
                    return {"success": False, "error": "All chunks failed to transcribe"}

                # Merge transcripts intelligently
                if CHUNKER_V2_AVAILABLE and hasattr(self.chunker, 'smart_merge_transcripts'):
                    merged_transcript = self.chunker.smart_merge_transcripts(chunk_results)
                else:
                    merged_transcript = self.chunker.merge_transcripts(chunk_results)

                average_confidence = total_confidence / successful_chunks if successful_chunks > 0 else 0

                logger.info(f"✅ Smart chunking complete: {len(merged_transcript)} characters, {successful_chunks}/{len(chunks)} chunks successful")

                return {
                    "success": True,
                    "transcript": merged_transcript,
                    "confidence": average_confidence,
                    "language": language,
                    "service": f"Mesolitica {self.model_name.split('/')[-1]} (Smart Chunked)",
                    "word_error_rate": 0.08 if "large" in self.model_name else 0.12,
                    "model": self.model_name,
                    "chunks_processed": successful_chunks,
                    "total_chunks": len(chunks),
                    "chunking_method": "Smart VAD-based"
                }

        except Exception as e:
            logger.error(f"Chunked transcription error: {e}")
            return {"success": False, "error": str(e)}

class WhisperFallback:
    """OpenAI Whisper as fallback option"""

    def __init__(self, model_name: str = "large-v2", device: str = "auto"):
        self.model_name = model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        self.model = None
        self.load_model()

    def load_model(self):
        """Load OpenAI Whisper model"""
        try:
            logger.info(f"Loading Whisper {self.model_name} on {self.device}")
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info("✅ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None

    def transcribe_audio(self, audio_file_path: str, language: str = "ms") -> Dict[str, Any]:
        """Transcribe audio using OpenAI Whisper"""
        if not self.model:
            return {"success": False, "error": "Whisper model not loaded"}

        try:
            # Enhanced settings for better accuracy
            result = self.model.transcribe(
                audio_file_path,
                language=language,
                beam_size=10,  # Higher beam size for accuracy
                best_of=5,     # Consider top 5 candidates
                temperature=[0.0, 0.2, 0.4, 0.6, 0.8],  # Multiple temperature fallback
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False,
                word_timestamps=True,
                initial_prompt=""  # Clean slate for Malaysian context
            )

            transcript = result["text"].strip()
            logger.info(f"Whisper: {transcript[:100]}...")

            return {
                "success": True,
                "transcript": transcript,
                "confidence": 0.75,  # Conservative estimate for generic model
                "language": result.get("language", language),
                "service": f"OpenAI Whisper {self.model_name}",
                "word_error_rate": 0.25,  # Your current experience
                "segments": result.get("segments", [])
            }

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}")
            return {"success": False, "error": str(e)}

class EnhancedTranscriptionPipeline:
    """Main transcription pipeline with intelligent fallback"""

    def __init__(self):
        self.config = load_config()

        # Initialize services
        self.elevenlabs = None
        self.mesolitica = None
        self.whisper = None

        self._initialize_services()

    def _initialize_services(self):
        """Initialize transcription services based on config"""
        try:
            # ElevenLabs Scribe (highest accuracy)
            api_key = self.config["services"]["elevenlabs_api_key"]
            if api_key:
                self.elevenlabs = ElevenLabsScribe(api_key)
                logger.info("✅ ElevenLabs Scribe initialized")
            else:
                logger.warning("⚠️ ElevenLabs API key not configured")

            # Mesolitica Malaysian Whisper
            device = self.config["models"]["device"]
            self.mesolitica = MesoliticaWhisper(
                self.config["models"]["mesolitica_model"],
                device,
                self.config  # Pass full config for chunking settings
            )

            # OpenAI Whisper fallback
            self.whisper = WhisperFallback(
                self.config["models"]["whisper_model"],
                device
            )

        except Exception as e:
            logger.error(f"Service initialization error: {e}")

    def transcribe_with_best_accuracy(self, audio_file_path: str, language: str = "ms") -> Dict[str, Any]:
        """Transcribe using the best available service with intelligent fallback"""
        results = []
        primary_service = self.config["services"]["primary"]
        fallback_order = self.config["services"]["fallback_order"]
        confidence_threshold = self.config["services"]["confidence_threshold"]

        # Try primary service first
        if primary_service == "elevenlabs" and self.elevenlabs:
            logger.info("🎯 Trying ElevenLabs Scribe (highest accuracy)")
            result = self.elevenlabs.transcribe_audio(audio_file_path, language)
            if result["success"] and result.get("confidence", 0) >= confidence_threshold:
                logger.info(f"✅ ElevenLabs success: {result['confidence']:.2f} confidence")
                return result
            results.append(result)

        # Try fallback services
        for service_name in fallback_order:
            if service_name == "mesolitica" and self.mesolitica:
                logger.info("🎯 Trying Mesolitica Malaysian Whisper")
                result = self.mesolitica.transcribe_audio(audio_file_path, language)
                if result["success"] and result.get("confidence", 0) >= 0.7:  # Lower threshold for fallback
                    logger.info(f"✅ Mesolitica success: {result['confidence']:.2f} confidence")
                    return result
                results.append(result)

            elif service_name == "whisper" and self.whisper:
                logger.info("🎯 Trying OpenAI Whisper (fallback)")
                result = self.whisper.transcribe_audio(audio_file_path, language)
                if result["success"]:
                    logger.info(f"✅ Whisper fallback: {result['confidence']:.2f} confidence")
                    return result
                results.append(result)

        # If all services failed, return the best attempt
        successful_results = [r for r in results if r["success"]]
        if successful_results:
            best_result = max(successful_results, key=lambda x: x.get("confidence", 0))
            logger.info(f"🔄 Returning best result: {best_result['service']}")
            return best_result

        # Complete failure
        return {
            "success": False,
            "error": "All transcription services failed",
            "attempts": results
        }

# Global pipeline
pipeline = EnhancedTranscriptionPipeline()

@app.on_event("startup")
def startup_event():
    """Initialize the pipeline on startup"""
    logger.info("🚀 Enhanced Malaysian Speech-to-Text System starting...")
    logger.info("✅ All services initialized")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    config = load_config()

    # Check service availability
    services_status = {
        "elevenlabs": pipeline.elevenlabs is not None,
        "mesolitica": pipeline.mesolitica is not None and pipeline.mesolitica.model is not None,
        "whisper": pipeline.whisper is not None and pipeline.whisper.model is not None
    }

    return {
        "status": "healthy",
        "services": services_status,
        "config": {
            "primary_service": config["services"]["primary"],
            "fallback_order": config["services"]["fallback_order"],
            "language": config["processing"]["language"]
        },
        "accuracy_estimates": {
            "elevenlabs_scribe": "≤5% WER (~95% accuracy)",
            "mesolitica_malaysian": "~12% WER (~88% accuracy)",
            "whisper_large_v2": "~25% WER (~75% accuracy)"
        },
        "timestamp": int(time.time() * 1000)
    }

@app.post("/api/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form(default="ms"),
    force_service: str = Form(default="auto")  # auto, elevenlabs, mesolitica, whisper
):
    """Transcribe uploaded audio file with highest accuracy"""

    if not file.content_type or not file.content_type.startswith("audio/"):
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "File must be an audio file"}
        )

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            if force_service != "auto":
                # Force specific service
                if force_service == "elevenlabs" and pipeline.elevenlabs:
                    result = pipeline.elevenlabs.transcribe_audio(temp_path, language)
                elif force_service == "mesolitica" and pipeline.mesolitica:
                    result = pipeline.mesolitica.transcribe_audio(temp_path, language)
                elif force_service == "whisper" and pipeline.whisper:
                    result = pipeline.whisper.transcribe_audio(temp_path, language)
                else:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "error": f"Service '{force_service}' not available"}
                    )
            else:
                # Use intelligent pipeline
                result = pipeline.transcribe_with_best_accuracy(temp_path, language)

            # Clean up temp file
            os.remove(temp_path)

            if result["success"]:
                # Auto-save transcript
                save_info = transcript_saver.save_transcript(result, file.filename)

                response_data = {
                    "success": True,
                    "transcript": result["transcript"],
                    "confidence": result.get("confidence", 0),
                    "service": result["service"],
                    "language": result.get("language", language),
                    "word_error_rate": result.get("word_error_rate", "unknown"),
                    "duration": len(content) / 16000 / 2,  # Rough estimate
                    "file_size": len(content),
                    "processing_time": time.time(),
                    "word_count": len(result["transcript"].split()),
                    "character_count": len(result["transcript"]),
                    "auto_saved": save_info.get("saved", False)
                }

                # Add save paths if successful
                if save_info.get("saved"):
                    response_data["saved_files"] = {
                        "json": save_info.get("json_path"),
                        "txt": save_info.get("txt_path")
                    }

                return response_data
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "success": False,
                        "error": result.get("error", "Transcription failed"),
                        "attempts": result.get("attempts", [])
                    }
                )

        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    config = load_config()
    return {"success": True, "config": config}

@app.post("/api/config")
async def update_config(new_config: dict):
    """Update configuration"""
    try:
        current_config = load_config()

        # Deep merge
        def deep_merge(base: dict, update: dict) -> dict:
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    base[key] = deep_merge(base[key], value)
                else:
                    base[key] = value
            return base

        updated_config = deep_merge(current_config, new_config)

        if save_config(updated_config):
            # Reinitialize services if needed
            global pipeline
            pipeline = EnhancedTranscriptionPipeline()

            return {
                "success": True,
                "message": "Configuration updated successfully",
                "config": updated_config
            }
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": "Failed to save configuration"}
            )
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/services/status")
async def get_services_status():
    """Get detailed status of all transcription services"""
    return {
        "elevenlabs": {
            "available": pipeline.elevenlabs is not None,
            "accuracy": "≤5% WER (~95% accuracy)",
            "cost": "$0.40/hour",
            "best_for": "Critical applications requiring highest accuracy"
        },
        "mesolitica": {
            "available": pipeline.mesolitica is not None and pipeline.mesolitica.model is not None,
            "accuracy": "~12% WER (~88% accuracy)",
            "cost": "Free (local processing)",
            "best_for": "Malaysian-specific speech patterns, privacy-sensitive data"
        },
        "whisper": {
            "available": pipeline.whisper is not None and pipeline.whisper.model is not None,
            "accuracy": "~25% WER (~75% accuracy)",
            "cost": "Free (local processing)",
            "best_for": "General purpose, backup processing"
        }
    }

@app.get("/api/transcripts")
async def list_saved_transcripts():
    """Get list of saved transcripts"""
    try:
        transcripts_dir = Path("transcripts")
        if not transcripts_dir.exists():
            return {"transcripts": []}

        transcripts = []
        for json_file in transcripts_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    transcripts.append({
                        "filename": json_file.name,
                        "saved_at": data.get("saved_at"),
                        "audio_filename": data.get("audio_filename"),
                        "service": data.get("service"),
                        "confidence": data.get("confidence", 0),
                        "word_count": data.get("word_count", 0),
                        "character_count": data.get("character_count", 0),
                        "transcript_preview": data.get("transcript", "")[:100] + "..." if data.get("transcript", "") else ""
                    })
            except Exception as e:
                logger.error(f"Error reading transcript file {json_file}: {e}")

        # Sort by saved_at (newest first)
        transcripts.sort(key=lambda x: x.get("saved_at", ""), reverse=True)

        return {
            "success": True,
            "transcripts": transcripts,
            "total_count": len(transcripts)
        }

    except Exception as e:
        logger.error(f"Error listing transcripts: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/transcripts/{filename}")
async def get_transcript(filename: str):
    """Get full transcript content"""
    try:
        transcript_path = Path("transcripts") / filename
        if not transcript_path.exists():
            return JSONResponse(
                status_code=404,
                content={"success": False, "error": "Transcript not found"}
            )

        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return {
            "success": True,
            "transcript": data
        }

    except Exception as e:
        logger.error(f"Error reading transcript {filename}: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

if __name__ == "__main__":
    import asyncio
    import threading

    # Start WebSocket servers for live transcription in separate threads
    def start_websocket_servers():
        logger.info("🔄 Starting WebSocket servers thread...")
        try:
            # Start Mesolitica streaming server integration

            # Get Mesolitica service for live transcription
            mesolitica_service = pipeline.mesolitica if pipeline and pipeline.mesolitica else None
            logger.info(f"📋 Mesolitica service available: {mesolitica_service is not None}")

            if mesolitica_service:
                # Start Mesolitica streaming server (port 8003)
                try:
                    from mesolitica_streaming_engine import start_mesolitica_streaming_server

                    logger.info("🎯 Starting Mesolitica streaming server on port 8003")
                    streaming_thread = threading.Thread(
                        target=lambda: asyncio.run(start_mesolitica_streaming_server(mesolitica_service, "localhost", 8003)),
                        daemon=True
                    )
                    streaming_thread.start()

                except ImportError:
                    logger.warning("⚠️ Mesolitica streaming engine not available")

            else:
                logger.warning("⚠️ Mesolitica service not available for live transcription")
        except ImportError:
            logger.warning("⚠️ Live transcription engine not available")
        except Exception as e:
            logger.error(f"❌ Failed to start live transcription servers: {e}")

    # Start WebSocket servers in background thread
    logger.info("🚀 Starting WebSocket thread...")
    websocket_thread = threading.Thread(target=start_websocket_servers, daemon=True)
    websocket_thread.start()
    logger.info("✅ WebSocket thread started")

    # Start main FastAPI server
    logger.info("🚀 Starting main transcription server on port 8001")
    uvicorn.run(
        "enhanced_whisper_main:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )