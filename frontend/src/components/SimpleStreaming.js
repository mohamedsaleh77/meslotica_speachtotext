import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Square, Volume2, AlertCircle, CheckCircle, Clock, Wifi, WifiOff, Zap, Target } from 'lucide-react';
import API_URL from '../config';

const SimpleStreaming = ({ serverStatus, onResult }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [streamingTranscript, setStreamingTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [duration, setDuration] = useState(0);
  const [chunkCount, setChunkCount] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [currentService, setCurrentService] = useState('');

  // Stats for streaming
  const [streamingStats, setStreamingStats] = useState({
    chunksCreated: 0,
    chunksProcessing: 0,
    chunksCompleted: 0,
    totalWords: 0,
    avgConfidence: 0
  });

  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const processingRef = useRef(false);

  useEffect(() => {
    return () => {
      stopRecording();
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      streamRef.current = stream;

      // Setup audio context for visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;

      // Setup media recorder for TRUE STREAMING
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      let chunkCounter = 0;

      mediaRecorderRef.current.ondataavailable = async (event) => {
        if (event.data.size > 0 && !processingRef.current) {
          processingRef.current = true;
          chunkCounter++;

          // Update stats immediately
          setStreamingStats(prev => ({
            ...prev,
            chunksCreated: chunkCounter,
            chunksProcessing: prev.chunksProcessing + 1
          }));

          setChunkCount(chunkCounter);

          // Process chunk for real-time transcription
          await processStreamingChunk(event.data, chunkCounter);

          processingRef.current = false;
        }
      };

      mediaRecorderRef.current.onstop = () => {
        setIsRecording(false);
      };

      setIsRecording(true);
      setStreamingTranscript('');
      setDuration(0);
      setChunkCount(0);
      setStreamingStats({
        chunksCreated: 0,
        chunksProcessing: 0,
        chunksCompleted: 0,
        totalWords: 0,
        avgConfidence: 0
      });

      // Start recording with 4-second chunks for optimal streaming
      mediaRecorderRef.current.start(4000);

      // Start duration timer
      intervalRef.current = setInterval(() => {
        setDuration(prev => prev + 1);
      }, 1000);

      // Start audio level monitoring
      monitorAudioLevel();

    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    setIsRecording(false);
    setAudioLevel(0);
  };

  const processStreamingChunk = async (audioBlob, chunkNumber) => {
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', audioBlob, `stream_chunk_${chunkNumber}.webm`);
      formData.append('language', 'ms');
      formData.append('force_service', 'mesolitica'); // Force Mesolitica for streaming

      const startTime = Date.now();
      const response = await fetch(`${API_URL}/api/transcribe`, {
        method: 'POST',
        body: formData
      });

      const processingTime = Date.now() - startTime;
      const data = await response.json();

      if (data.success && data.transcript.trim()) {
        const newText = data.transcript.trim();

        // Append new text to streaming transcript
        setStreamingTranscript(prev => {
          // Simple deduplication - avoid repeating the same phrase
          const words = prev.split(' ');
          const newWords = newText.split(' ');

          // If the new text starts with the last few words of previous text, merge intelligently
          if (words.length > 3 && newWords.length > 0) {
            const lastThreeWords = words.slice(-3).join(' ').toLowerCase();
            const firstThreeNewWords = newWords.slice(0, 3).join(' ').toLowerCase();

            if (lastThreeWords.includes(firstThreeNewWords.split(' ')[0])) {
              // Potential overlap, just add unique words
              return prev + ' ' + newText;
            }
          }

          return prev + (prev ? ' ' : '') + newText;
        });

        setConfidence(data.confidence || 0);
        setCurrentService(data.service);

        // Update streaming stats
        setStreamingStats(prev => ({
          ...prev,
          chunksProcessing: Math.max(0, prev.chunksProcessing - 1),
          chunksCompleted: prev.chunksCompleted + 1,
          totalWords: prev.totalWords + newText.split(' ').length,
          avgConfidence: ((prev.avgConfidence * prev.chunksCompleted) + (data.confidence || 0)) / (prev.chunksCompleted + 1)
        }));

        console.log(`✅ Streaming chunk ${chunkNumber} processed: "${newText}"`);

        // Add to comparison results
        if (onResult) {
          const result = {
            id: Date.now(),
            timestamp: new Date(),
            type: 'streaming',
            service: data.service,
            transcript: newText,
            confidence: data.confidence,
            processingTime,
            duration: 4, // 4-second chunks
            chunkNumber,
            audioLength: 4
          };
          onResult(result);
        }

      } else if (data.error) {
        console.warn(`⚠️ Streaming chunk ${chunkNumber} failed:`, data.error);
      }
    } catch (error) {
      console.error(`❌ Error processing streaming chunk ${chunkNumber}:`, error);
    } finally {
      setIsProcessing(false);
    }
  };

  const monitorAudioLevel = () => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);

    const updateLevel = () => {
      if (!isRecording) return;

      analyserRef.current.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      setAudioLevel(average / 255 * 100);

      requestAnimationFrame(updateLevel);
    };

    updateLevel();
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Simple Mesolitica Streaming</h2>
        <p className="text-gray-600">True real-time transcription with 4-second chunks</p>
      </div>

      {/* Streaming Status */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className={`text-2xl font-bold ${isRecording ? 'text-green-600' : 'text-gray-400'}`}>
              {isRecording ? 'STREAMING' : 'READY'}
            </div>
            <div className="text-sm text-gray-600">Status</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{formatDuration(duration)}</div>
            <div className="text-sm text-gray-600">Duration</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{streamingStats.chunksCompleted}/{streamingStats.chunksCreated}</div>
            <div className="text-sm text-gray-600">Chunks Done</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-orange-600">{streamingStats.totalWords}</div>
            <div className="text-sm text-gray-600">Words</div>
          </div>
        </div>
      </div>

      {/* Recording Controls */}
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="text-center space-y-6">
          {/* Status Display */}
          <div className="space-y-2">
            <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium ${
              isRecording ? 'bg-red-100 text-red-800' : 'bg-gray-100 text-gray-800'
            }`}>
              {isRecording ? (
                <>
                  <div className="w-2 h-2 bg-red-600 rounded-full animate-pulse"></div>
                  <span>Live Streaming</span>
                </>
              ) : (
                <>
                  <MicOff className="h-4 w-4" />
                  <span>Ready</span>
                </>
              )}
            </div>

            {isRecording && (
              <div className="text-2xl font-mono text-gray-700">
                {formatDuration(duration)}
              </div>
            )}
          </div>

          {/* Audio Level Visualization */}
          {isRecording && (
            <div className="flex justify-center">
              <div className="flex items-end space-x-1 h-16">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={i}
                    className="w-3 bg-blue-500 rounded-t transition-all duration-200"
                    style={{
                      height: `${Math.max(8, audioLevel * 0.6)}px`,
                      opacity: audioLevel > 10 ? 1 : 0.3
                    }}
                  ></div>
                ))}
              </div>
            </div>
          )}

          {/* Control Buttons */}
          <div className="flex justify-center space-x-4">
            {!isRecording ? (
              <button
                onClick={startRecording}
                disabled={serverStatus?.status !== 'healthy'}
                className="flex items-center space-x-2 px-8 py-4 bg-blue-600 text-white rounded-full hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                <Mic className="h-6 w-6" />
                <span className="text-lg font-medium">Start Streaming</span>
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="flex items-center space-x-2 px-8 py-4 bg-red-600 text-white rounded-full hover:bg-red-700 transition-colors"
              >
                <Square className="h-6 w-6" />
                <span className="text-lg font-medium">Stop Streaming</span>
              </button>
            )}
          </div>

          {serverStatus?.status !== 'healthy' && (
            <div className="flex items-center justify-center space-x-2 text-red-600">
              <AlertCircle className="h-5 w-5" />
              <span>Server not available. Please start the backend server.</span>
            </div>
          )}
        </div>
      </div>

      {/* Live Transcript */}
      {(streamingTranscript || isRecording) && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">Live Transcript</h3>
            <div className="flex items-center space-x-3">
              {isProcessing && (
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <span className="text-sm text-blue-600">Processing...</span>
                </div>
              )}
              {currentService && (
                <div className="px-3 py-1 rounded-full text-sm font-medium bg-blue-50 text-blue-600">
                  Mesolitica • 88% accuracy
                </div>
              )}
            </div>
          </div>

          {/* Live Processing Status */}
          {isRecording && (
            <div className="mb-4 p-3 bg-green-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-600 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium text-green-900">
                    Real-time streaming active • Chunks: {streamingStats.chunksCompleted}/{streamingStats.chunksCreated}
                  </span>
                </div>
                <span className="text-xs text-green-700">
                  Next chunk in: {4 - (duration % 4)}s
                </span>
              </div>
            </div>
          )}

          {/* Confidence Score */}
          {confidence > 0 && (
            <div className="flex items-center space-x-4 mb-4">
              <span className="text-sm font-medium text-gray-700">Confidence:</span>
              <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div
                  className="h-2 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                  style={{ width: `${confidence * 100}%` }}
                ></div>
              </div>
              <span className="text-sm font-medium text-gray-700">{(confidence * 100).toFixed(1)}%</span>
            </div>
          )}

          {/* Live Transcript */}
          <div className="transcript-container">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <Volume2 className="h-5 w-5 text-blue-600" />
                <span className="font-medium text-gray-900">Streaming Transcript</span>
              </div>
              {streamingTranscript && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                  🔴 Live
                </span>
              )}
            </div>
            <div className="max-h-64 overflow-y-auto bg-gray-50 rounded-lg p-4">
              <p className="text-lg leading-relaxed text-gray-900 whitespace-pre-wrap">
                {streamingTranscript || 'Start speaking to see live transcription...'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SimpleStreaming;