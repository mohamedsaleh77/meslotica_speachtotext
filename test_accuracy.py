#!/usr/bin/env python3
"""
🧪 Accuracy Testing Script for Malaysian Speech-to-Text
======================================================
Compare all services with your audio samples to find the best accuracy
"""

import os
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Any

class AccuracyTester:
    """Test and compare accuracy of different speech-to-text services"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []

    def test_audio_file(self, audio_path: str, expected_text: str = None) -> Dict[str, Any]:
        """Test a single audio file with all available services"""

        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}

        results = {
            "audio_file": os.path.basename(audio_path),
            "file_size": os.path.getsize(audio_path),
            "expected_text": expected_text,
            "services": {}
        }

        services = ["auto", "elevenlabs", "mesolitica", "whisper"]

        for service in services:
            print(f"🎯 Testing {service} with {results['audio_file']}...")

            try:
                start_time = time.time()

                with open(audio_path, 'rb') as f:
                    response = requests.post(
                        f"{self.base_url}/api/transcribe",
                        files={"file": f},
                        data={"force_service": service, "language": "ms"},
                        timeout=120
                    )

                processing_time = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()

                    if data["success"]:
                        transcript = data["transcript"]
                        confidence = data.get("confidence", 0)
                        actual_service = data.get("service", service)
                        wer = data.get("word_error_rate", "unknown")

                        # Calculate basic accuracy metrics if expected text provided
                        accuracy_score = None
                        if expected_text:
                            accuracy_score = self.calculate_simple_accuracy(expected_text, transcript)

                        results["services"][service] = {
                            "success": True,
                            "transcript": transcript,
                            "confidence": confidence,
                            "actual_service": actual_service,
                            "word_error_rate": wer,
                            "processing_time": processing_time,
                            "accuracy_score": accuracy_score,
                            "transcript_length": len(transcript.split())
                        }

                        print(f"✅ {service}: {confidence:.2f} confidence, {len(transcript.split())} words")
                        print(f"   Text: {transcript[:100]}...")

                    else:
                        results["services"][service] = {
                            "success": False,
                            "error": data.get("error", "Unknown error"),
                            "processing_time": processing_time
                        }
                        print(f"❌ {service}: {data.get('error', 'Unknown error')}")
                else:
                    results["services"][service] = {
                        "success": False,
                        "error": f"HTTP {response.status_code}: {response.text}",
                        "processing_time": processing_time
                    }
                    print(f"❌ {service}: HTTP {response.status_code}")

            except Exception as e:
                results["services"][service] = {
                    "success": False,
                    "error": str(e),
                    "processing_time": 0
                }
                print(f"❌ {service}: {str(e)}")

        return results

    def calculate_simple_accuracy(self, expected: str, actual: str) -> float:
        """Calculate simple word-level accuracy"""
        expected_words = expected.lower().split()
        actual_words = actual.lower().split()

        if not expected_words:
            return 0.0

        # Simple word matching
        matches = 0
        max_len = max(len(expected_words), len(actual_words))

        for i in range(min(len(expected_words), len(actual_words))):
            if expected_words[i] == actual_words[i]:
                matches += 1

        return matches / len(expected_words) if expected_words else 0.0

    def test_directory(self, audio_dir: str, expected_texts: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """Test all audio files in a directory"""
        audio_dir = Path(audio_dir)

        if not audio_dir.exists():
            print(f"❌ Directory not found: {audio_dir}")
            return []

        # Supported audio formats
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        audio_files = [f for f in audio_dir.iterdir() if f.suffix.lower() in audio_extensions]

        if not audio_files:
            print(f"❌ No audio files found in {audio_dir}")
            return []

        print(f"🎵 Found {len(audio_files)} audio files")

        results = []
        for audio_file in audio_files:
            expected_text = expected_texts.get(audio_file.name) if expected_texts else None
            result = self.test_audio_file(str(audio_file), expected_text)
            results.append(result)

            # Add some delay between requests
            time.sleep(1)

        return results

    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive accuracy report"""

        if not results:
            return {"error": "No results to analyze"}

        report = {
            "summary": {
                "total_files": len(results),
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "services_tested": ["elevenlabs", "mesolitica", "whisper"]
            },
            "service_performance": {},
            "detailed_results": results
        }

        # Analyze each service
        services = ["elevenlabs", "mesolitica", "whisper"]

        for service in services:
            successes = 0
            total_confidence = 0
            total_processing_time = 0
            total_accuracy = 0
            accuracy_count = 0

            for result in results:
                if service in result["services"]:
                    service_result = result["services"][service]

                    if service_result["success"]:
                        successes += 1
                        total_confidence += service_result.get("confidence", 0)
                        total_processing_time += service_result.get("processing_time", 0)

                        if service_result.get("accuracy_score") is not None:
                            total_accuracy += service_result["accuracy_score"]
                            accuracy_count += 1

            success_rate = successes / len(results) if results else 0
            avg_confidence = total_confidence / successes if successes > 0 else 0
            avg_processing_time = total_processing_time / len(results) if results else 0
            avg_accuracy = total_accuracy / accuracy_count if accuracy_count > 0 else None

            report["service_performance"][service] = {
                "success_rate": success_rate,
                "average_confidence": avg_confidence,
                "average_processing_time": avg_processing_time,
                "average_accuracy": avg_accuracy,
                "successful_transcriptions": successes,
                "total_attempts": len(results)
            }

        # Determine best service
        best_service = None
        best_score = 0

        for service, perf in report["service_performance"].items():
            # Score based on success rate and confidence
            score = perf["success_rate"] * perf["average_confidence"]
            if score > best_score:
                best_score = score
                best_service = service

        report["recommendation"] = {
            "best_service": best_service,
            "score": best_score,
            "reasoning": f"Based on {best_service} having the highest success rate and confidence"
        }

        return report

    def save_report(self, report: Dict[str, Any], filename: str = "accuracy_report.json"):
        """Save report to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"📊 Report saved to: {filename}")

    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the test results"""
        print("\n" + "="*60)
        print("🎯 ACCURACY TEST RESULTS SUMMARY")
        print("="*60)

        summary = report["summary"]
        print(f"📁 Files tested: {summary['total_files']}")
        print(f"🕒 Test time: {summary['test_timestamp']}")

        print(f"\n📊 SERVICE PERFORMANCE:")

        for service, perf in report["service_performance"].items():
            print(f"\n🔧 {service.upper()}:")
            print(f"   ✅ Success rate: {perf['success_rate']:.1%}")
            print(f"   🎯 Avg confidence: {perf['average_confidence']:.2f}")
            print(f"   ⏱️  Avg time: {perf['average_processing_time']:.1f}s")
            if perf['average_accuracy']:
                print(f"   📈 Avg accuracy: {perf['average_accuracy']:.1%}")

        print(f"\n🏆 RECOMMENDATION:")
        rec = report["recommendation"]
        print(f"   Best service: {rec['best_service'].upper()}")
        print(f"   Reason: {rec['reasoning']}")

        print("\n" + "="*60)

def main():
    """Main testing function"""
    tester = AccuracyTester()

    print("🎯 Enhanced Malaysian Speech-to-Text Accuracy Tester")
    print("=" * 60)

    # Check if server is running
    try:
        response = requests.get(f"{tester.base_url}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not running. Start with: python enhanced_whisper_main.py")
            return
    except:
        print("❌ Server not running. Start with: python enhanced_whisper_main.py")
        return

    # Test options
    print("\nTesting options:")
    print("1. Test a single audio file")
    print("2. Test all files in a directory")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        audio_path = input("Enter path to audio file: ").strip()
        expected_text = input("Enter expected text (optional): ").strip() or None

        print(f"\n🎵 Testing: {audio_path}")
        result = tester.test_audio_file(audio_path, expected_text)

        if "error" not in result:
            report = tester.generate_report([result])
            tester.print_summary(report)
            tester.save_report(report, f"single_file_report_{int(time.time())}.json")
        else:
            print(f"❌ Error: {result['error']}")

    elif choice == "2":
        audio_dir = input("Enter path to audio directory: ").strip()

        print(f"\n📁 Testing directory: {audio_dir}")
        results = tester.test_directory(audio_dir)

        if results:
            report = tester.generate_report(results)
            tester.print_summary(report)
            tester.save_report(report, f"directory_report_{int(time.time())}.json")
        else:
            print("❌ No results to analyze")

    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()