# Translation Feature Documentation

## Overview
The Malaysian Speech-to-Text system now includes a powerful multi-language translation feature that allows you to view transcriptions in 4 languages:
- **Bahasa Malay** (Original)
- **English**
- **Tamil**
- **Chinese (Simplified)**

## Features

### 1. Transcription Viewer
- **Location**: Access via the "Transcription Viewer" tab in the main interface
- **Purpose**: Browse and view all saved transcriptions from the `transcripts` folder
- **Search**: Built-in search functionality to find specific transcriptions

### 2. Multi-Language Translation
- **Smart Caching**: Translations are cached locally to avoid repeated API calls
- **One-Click Translation**: Translate to all languages at once or select individual languages
- **Real-time Translation**: Fast translation using OpenAI GPT-3.5 Turbo

### 3. Translation Service Features
- **Cache Directory**: `translation_cache/`
- **Supported Languages**: Malay, English, Tamil, Chinese
- **API**: OpenAI GPT-3.5 Turbo
- **Caching**: MD5-based caching system

## How to Use

### Starting the System
```bash
# Install dependencies (if not already installed)
pip install -r requirements.txt
cd frontend && npm install

# Start the backend server
python enhanced_whisper_main.py

# In another terminal, start the frontend
cd frontend && npm start
```

### Using the Transcription Viewer

1. **Navigate to Transcription Viewer Tab**
   - Click on "Transcription Viewer" in the navigation menu
   - The tab shows with a Languages icon

2. **Browse Transcriptions**
   - All saved transcriptions appear in the left panel
   - Shows filename, date, confidence score, and preview
   - Use the search bar to filter transcriptions

3. **View and Translate**
   - Click on any transcription to view it
   - Original Malay text is displayed by default
   - Click language buttons to translate:
     - 🇲🇾 Bahasa Malay (Original)
     - 🇬🇧 English
     - 🇮🇳 Tamil
     - 🇨🇳 Chinese
   - Or click "Translate All" to translate to all languages at once

4. **Download Translations**
   - Click the download button to save all translations as JSON
   - Includes original text and all translations

## API Endpoints

### Translation Endpoints

#### Single Translation
- **Endpoint**: `POST /api/translate`
- **Parameters**:
  - `text`: Text to translate
  - `target_language`: Target language (english, tamil, chinese)
  - `source_language`: Source language (default: malay)

#### Multiple Translations
- **Endpoint**: `POST /api/translate/multiple`
- **Parameters**:
  - `text`: Text to translate
  - `source_language`: Source language (default: malay)
- **Returns**: Translations in English, Tamil, and Chinese

#### Cache Statistics
- **Endpoint**: `GET /api/translation/cache/stats`
- **Returns**: Cache statistics including total cached translations and size

### Transcription Endpoints

#### List Transcriptions
- **Endpoint**: `GET /api/transcripts`
- **Returns**: List of all saved transcriptions

#### Get Specific Transcription
- **Endpoint**: `GET /api/transcripts/{filename}`
- **Returns**: Full content of specific transcription

## Configuration

### OpenAI API Key
The system is pre-configured with an OpenAI API key in `enhanced_whisper_main.py`:

```python
translation_service = TranslationService(
    api_key="your-api-key-here",
    model="gpt-3.5-turbo"
)
```

### Cache Management
- Cache files are stored in `translation_cache/` directory
- Each translation is cached using MD5 hash of content + language
- Cache is persistent across sessions

## Testing

Run the test script to verify translation service:

```bash
python test_translation.py
```

This will:
1. Test single translation to English
2. Test multiple translations (English, Tamil, Chinese)
3. Display cache statistics
4. Verify caching functionality

## Troubleshooting

### Common Issues

1. **Translation fails with API error**
   - Check OpenAI API key is valid
   - Verify internet connection
   - Check API rate limits

2. **Translations not appearing**
   - Ensure backend server is running
   - Check browser console for errors
   - Verify CORS settings

3. **Cache not working**
   - Check `translation_cache/` directory exists
   - Verify write permissions
   - Clear cache if corrupted

### Clearing Cache

To clear the translation cache:
```bash
rm -rf translation_cache/*
```

## Technical Details

### File Structure
```
├── translation_service.py      # Translation service with OpenAI
├── translation_cache/          # Cache directory
├── enhanced_whisper_main.py    # Backend with translation endpoints
├── frontend/
│   └── src/
│       └── components/
│           └── TranscriptionViewer.js  # React component
```

### Dependencies
- `openai==0.28.0` - OpenAI API client
- `fastapi` - Backend framework
- `React` - Frontend framework

## Future Enhancements

Potential improvements:
1. Add more languages (Indonesian, Hindi, etc.)
2. Support for different translation models
3. Batch translation processing
4. Export translations to different formats (PDF, DOCX)
5. Translation quality metrics
6. User-editable translations