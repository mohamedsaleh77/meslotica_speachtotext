#!/usr/bin/env python3
"""
Translation Service with OpenAI Integration
Supports translation from Malay to English, Tamil, and Chinese
Includes caching to avoid repeated API calls
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
import logging
import openai
from datetime import datetime

logger = logging.getLogger(__name__)

class TranslationService:
    """Service for translating transcriptions using OpenAI API with caching"""

    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", cache_dir: str = "translation_cache"):
        """Initialize translation service with OpenAI API key and cache directory"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        if self.api_key:
            openai.api_key = self.api_key
        else:
            logger.warning("No OpenAI API key provided")

    def _get_cache_key(self, text: str, target_language: str) -> str:
        """Generate cache key for translation"""
        content = f"{text}:{target_language}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_translation(self, text: str, target_language: str) -> Optional[str]:
        """Check if translation exists in cache"""
        cache_key = self._get_cache_key(text, target_language)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    logger.info(f"Using cached translation for {target_language}")
                    return cache_data.get('translation')
            except Exception as e:
                logger.error(f"Error reading cache: {e}")

        return None

    def _save_to_cache(self, text: str, target_language: str, translation: str):
        """Save translation to cache"""
        cache_key = self._get_cache_key(text, target_language)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            cache_data = {
                'original_text': text,
                'target_language': target_language,
                'translation': translation,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved translation to cache for {target_language}")
        except Exception as e:
            logger.error(f"Error saving to cache: {e}")

    def translate(self, text: str, target_language: str, source_language: str = "Malay") -> Dict[str, any]:
        """
        Translate text to target language

        Args:
            text: Text to translate
            target_language: Target language (English, Tamil, Chinese)
            source_language: Source language (default: Malay)

        Returns:
            Dictionary with translation result
        """

        # Check cache first
        cached = self._get_cached_translation(text, target_language)
        if cached:
            return {
                'success': True,
                'translation': cached,
                'source_language': source_language,
                'target_language': target_language,
                'from_cache': True
            }

        # If no API key, return error
        if not self.api_key:
            return {
                'success': False,
                'error': 'No OpenAI API key configured',
                'source_language': source_language,
                'target_language': target_language
            }

        try:
            # Create translation prompt
            language_map = {
                'english': 'English',
                'tamil': 'Tamil',
                'chinese': 'Simplified Chinese',
                'malay': 'Bahasa Malay'
            }

            target_lang_name = language_map.get(target_language.lower(), target_language)
            source_lang_name = language_map.get(source_language.lower(), source_language)

            prompt = f"""Translate the following text from {source_lang_name} to {target_lang_name}.
Preserve the formatting including line breaks and structure.
Only provide the translation without any explanations or additional text.

Text to translate:
{text}"""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": f"You are a professional translator specializing in {source_lang_name} to {target_lang_name} translation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )

            translation = response.choices[0].message.content.strip()

            # Save to cache
            self._save_to_cache(text, target_language, translation)

            logger.info(f"Successfully translated to {target_language}")

            return {
                'success': True,
                'translation': translation,
                'source_language': source_language,
                'target_language': target_language,
                'from_cache': False,
                'tokens_used': response.usage.total_tokens
            }

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'source_language': source_language,
                'target_language': target_language
            }

    def translate_multiple(self, text: str, target_languages: List[str] = None, source_language: str = "Malay") -> Dict[str, any]:
        """
        Translate text to multiple languages

        Args:
            text: Text to translate
            target_languages: List of target languages (default: English, Tamil, Chinese)
            source_language: Source language (default: Malay)

        Returns:
            Dictionary with translations for all languages
        """
        if target_languages is None:
            target_languages = ['english', 'tamil', 'chinese']

        results = {
            'success': True,
            'source_language': source_language,
            'original_text': text,
            'translations': {}
        }

        for lang in target_languages:
            translation_result = self.translate(text, lang, source_language)
            if translation_result['success']:
                results['translations'][lang] = translation_result['translation']
            else:
                results['translations'][lang] = f"Translation error: {translation_result.get('error', 'Unknown error')}"
                results['success'] = False

        return results

    def get_cache_stats(self) -> Dict[str, any]:
        """Get statistics about the translation cache"""
        cache_files = list(self.cache_dir.glob("*.json"))

        stats = {
            'total_cached': len(cache_files),
            'cache_size_mb': sum(f.stat().st_size for f in cache_files) / (1024 * 1024),
            'languages': {}
        }

        for cache_file in cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    lang = data.get('target_language', 'unknown')
                    stats['languages'][lang] = stats['languages'].get(lang, 0) + 1
            except:
                pass

        return stats