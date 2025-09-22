#!/usr/bin/env python3
"""
Test script for the translation service
"""

from translation_service import TranslationService
import json

def test_translation():
    """Test the translation service"""

    # Initialize service
    # Set OPENAI_API_KEY environment variable before running
    import os
    service = TranslationService(
        api_key=os.getenv("OPENAI_API_KEY"),  # Load from environment variable
        model="gpt-3.5-turbo"
    )

    # Test text in Bahasa Malay
    test_text = "Terima kasih kerana menonton! Saya sedia menerima sebarang maklum balas."

    print("=" * 60)
    print("TRANSLATION SERVICE TEST")
    print("=" * 60)
    print(f"\nOriginal text (Malay):\n{test_text}\n")

    # Test single translation
    print("-" * 60)
    print("Testing single translation to English...")
    result_en = service.translate(test_text, "english", "malay")

    if result_en['success']:
        print(f"✓ English translation:\n{result_en['translation']}")
        print(f"  From cache: {result_en['from_cache']}")
        if 'tokens_used' in result_en:
            print(f"  Tokens used: {result_en['tokens_used']}")
    else:
        print(f"✗ Translation failed: {result_en['error']}")

    # Test multiple translations
    print("\n" + "-" * 60)
    print("Testing multiple translations...")
    result_multi = service.translate_multiple(test_text)

    if result_multi['success']:
        print("✓ Multiple translations completed:")
        for lang, translation in result_multi['translations'].items():
            print(f"\n{lang.upper()}:")
            print(f"{translation[:100]}..." if len(translation) > 100 else translation)
    else:
        print("✗ Multiple translation failed")

    # Test cache stats
    print("\n" + "-" * 60)
    print("Cache statistics:")
    stats = service.get_cache_stats()
    print(json.dumps(stats, indent=2))

    print("\n" + "=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_translation()