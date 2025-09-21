# 🎯 Enhanced Malaysian Speech-to-Text System

**Highest accuracy Malaysian speech recognition using cutting-edge AI models**

## 🚀 Features

- **ElevenLabs Scribe**: ≤5% WER (~95% accuracy) - Industry leading
- **Mesolitica Malaysian Whisper**: ~12% WER (~88% accuracy) - Malaysian-specific training
- **OpenAI Whisper Large-v2**: ~25% WER (~75% accuracy) - Reliable fallback
- **Intelligent Pipeline**: Automatically uses best available service
- **Real-time Processing**: Fast transcription with quality optimization
- **Local Privacy**: Mesolitica and Whisper run locally

## 📊 Accuracy Comparison

| Service | Word Error Rate | Accuracy | Best For |
|---------|----------------|----------|----------|
| ElevenLabs Scribe | ≤5% | ~95% | Critical applications |
| Mesolitica Malaysian | ~12% | ~88% | Malaysian-specific speech |
| OpenAI Whisper v2 | ~25% | ~75% | General purpose/fallback |

## 🛠️ Installation

### 1. Clone and Setup
```bash
cd C:\Users\joshu\Desktop\Altureon\MalaySpeechToText
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Copy environment file
copy .env.example .env

# Edit .env file and add your ElevenLabs API key
# ELEVENLABS_API_KEY=your_key_here
```

### 3. Install Mesolitica Model
```bash
# Install Mesolitica dependencies
pip install git+https://github.com/mesolitica/malaya-speech.git
```

## 🚀 Quick Start

### Start the Server
```bash
python enhanced_whisper_main.py
```

Server runs on: `http://localhost:8000`

### Test Transcription
```bash
# Upload audio file for transcription
curl -X POST "http://localhost:8000/api/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav" \
  -F "language=ms"
```

## 📝 API Endpoints

### Health Check
```
GET /health
```
Returns system status and service availability.

### Transcribe Audio
```
POST /api/transcribe
```
Parameters:
- `file`: Audio file (WAV, MP3, M4A, etc.)
- `language`: Language code (default: "ms" for Malaysian)
- `force_service`: Force specific service ("auto", "elevenlabs", "mesolitica", "whisper")

### Configuration
```
GET /api/config           # Get current config
POST /api/config          # Update config
GET /api/services/status  # Get detailed service status
```

## 🎛️ Configuration

Edit `config.json` to customize:

```json
{
  "services": {
    "primary": "elevenlabs",           # Primary service
    "fallback_order": ["mesolitica", "whisper"],
    "confidence_threshold": 0.8
  },
  "models": {
    "whisper_model": "large-v2",
    "mesolitica_model": "mesolitica/malaysian-whisper-medium",
    "device": "auto"                   # auto, cuda, cpu
  }
}
```

## 💰 Cost Analysis

### ElevenLabs Scribe (Recommended for Production)
- **Cost**: $0.40/hour of audio
- **Accuracy**: ≤5% WER (~95% accuracy)
- **Best for**: Critical applications where accuracy matters

### Mesolitica Malaysian (Best for Privacy/Cost)
- **Cost**: Free (local processing)
- **Accuracy**: ~12% WER (~88% accuracy)
- **Best for**: Privacy-sensitive data, cost optimization

### OpenAI Whisper (Reliable Fallback)
- **Cost**: Free (local processing)
- **Accuracy**: ~25% WER (~75% accuracy)
- **Best for**: Backup processing, general use

## 🔧 Advanced Usage

### Force Specific Service
```python
import requests

# Force ElevenLabs for highest accuracy
response = requests.post(
    "http://localhost:8000/api/transcribe",
    files={"file": open("audio.wav", "rb")},
    data={"force_service": "elevenlabs", "language": "ms"}
)
```

### Batch Processing
```python
import os
import requests

audio_dir = "path/to/audio/files"
for filename in os.listdir(audio_dir):
    if filename.endswith(('.wav', '.mp3', '.m4a')):
        with open(os.path.join(audio_dir, filename), 'rb') as f:
            response = requests.post(
                "http://localhost:8000/api/transcribe",
                files={"file": f},
                data={"language": "ms"}
            )
            result = response.json()
            print(f"{filename}: {result['transcript']}")
```

## 🎯 Performance Tips

1. **For Highest Accuracy**: Use ElevenLabs Scribe
2. **For Cost Efficiency**: Use Mesolitica Malaysian Whisper
3. **For Privacy**: Use local models (Mesolitica + Whisper)
4. **GPU Acceleration**: Set `device: "cuda"` in config.json
5. **Audio Quality**: Use 16kHz WAV files for best results

## 🔍 Troubleshooting

### ElevenLabs API Issues
- Check API key in `.env` file
- Verify account credit balance
- Check network connectivity

### Mesolitica Model Issues
```bash
# Reinstall Mesolitica
pip uninstall malaya-speech
pip install git+https://github.com/mesolitica/malaya-speech.git
```

### GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Accuracy Testing

Test all services with your audio samples:

```bash
# Test with different services
curl -X POST "http://localhost:8000/api/transcribe" \
  -F "file=@test_audio.wav" \
  -F "force_service=elevenlabs"

curl -X POST "http://localhost:8000/api/transcribe" \
  -F "file=@test_audio.wav" \
  -F "force_service=mesolitica"

curl -X POST "http://localhost:8000/api/transcribe" \
  -F "file=@test_audio.wav" \
  -F "force_service=whisper"
```

## 🎯 Next Steps

1. **Get ElevenLabs API Key**: https://elevenlabs.io/pricing/api
2. **Test with your audio samples**
3. **Compare accuracy against your current ~60% system**
4. **Deploy the best performing service for production**

---

**Expected Improvement**: From ~60% accuracy to **88-95% accuracy** 🚀