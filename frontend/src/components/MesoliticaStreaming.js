import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Square, Volume2, AlertCircle, Zap, Target } from 'lucide-react';

const MesoliticaStreaming = ({ serverStatus, onResult }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [partialTranscript, setPartialTranscript] = useState('');
  const [fullTranscript, setFullTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [duration, setDuration] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');

  // Streaming-specific state
  const [streamingStats, setStreamingStats] = useState({
    chunksCreated: 0,
    chunksProcessing: 0,
    chunksCompleted: 0,
    bufferDuration: 0,
    speechDetected: false,
    silenceCounter: 0
  });

  const [recentChunks, setRecentChunks] = useState([]);
  const [lastProcessingTime, setLastProcessingTime] = useState(0);

  // Refs
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const chunkCounterRef = useRef(0);
  const isProcessingRef = useRef(false);

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
      setConnectionStatus('connected');

      // Setup audio context for visualization
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      analyserRef.current.fftSize = 256;

      // Setup MediaRecorder for HTTP streaming
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      // Reset counters
      chunkCounterRef.current = 0;

      mediaRecorderRef.current.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          chunkCounterRef.current++;
          const chunkNumber = chunkCounterRef.current;

          // Update stats immediately
          setStreamingStats(prev => ({
            ...prev,
            chunksCreated: chunkNumber,
            chunksProcessing: prev.chunksProcessing + 1,
            speechDetected: true,
            bufferDuration: 15.0
          }));

          // Process chunk via HTTP
          await processChunkHTTP(event.data, chunkNumber);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        setIsRecording(false);
        setConnectionStatus('disconnected');
      };

      // Initialize state
      setIsRecording(true);
      setPartialTranscript('');
      setFullTranscript('');
      setStreamingStats({
        chunksCreated: 0,
        chunksProcessing: 0,
        chunksCompleted: 0,
        bufferDuration: 0,
        speechDetected: false,
        silenceCounter: 0
      });
      setRecentChunks([]);
      setDuration(0);

      // Start recording with 15-second chunks for optimal streaming
      mediaRecorderRef.current.start(15000);

      // Start duration timer
      intervalRef.current = setInterval(() => {
        setDuration(prev => prev + 1);
      }, 1000);

      // Start audio level monitoring
      monitorAudioLevel();

      console.log('✅ Streaming started with 15-second HTTP chunks');

    } catch (error) {
      console.error('Error starting recording:', error);
      alert('Error accessing microphone. Please check permissions.');
      setConnectionStatus('error');
    }
  };

  const processChunkHTTP = async (audioBlob, chunkNumber) => {
    if (isProcessingRef.current) {
      console.log(`⏳ Skipping chunk ${chunkNumber} - still processing previous`);
      return;
    }

    isProcessingRef.current = true;
    const startTime = Date.now();

    try {
      const formData = new FormData();
      formData.append('file', audioBlob, `streaming_chunk_${chunkNumber}.webm`);
      formData.append('language', 'ms');
      formData.append('force_service', 'mesolitica');

      console.log(`📤 Sending chunk ${chunkNumber} for transcription...`);

      const response = await fetch('/api/transcribe', {
        method: 'POST',
        body: formData
      });

      const processingTime = Date.now() - startTime;
      const data = await response.json();

      if (data.success && data.transcript && data.transcript.trim()) {
        const newText = data.transcript.trim();

        console.log(`✅ Chunk ${chunkNumber} result: "${newText}"`);

        // Add to partial transcript (latest chunk)
        setPartialTranscript(newText);

        // Add to full transcript
        setFullTranscript(prev => {
          // Simple deduplication check
          if (prev && prev.endsWith(newText)) {
            return prev; // Skip duplicate
          }
          return prev + (prev ? ' ' : '') + newText;
        });

        // Update confidence
        setConfidence(data.confidence || 0);

        // Update recent chunks display
        setRecentChunks(prev => [
          {
            id: chunkNumber,
            text: newText,
            confidence: data.confidence || 0,
            timestamp: new Date().toLocaleTimeString()
          },
          ...prev.slice(0, 4) // Keep last 5 chunks
        ]);

        // Update stats
        setStreamingStats(prev => ({
          ...prev,
          chunksProcessing: Math.max(0, prev.chunksProcessing - 1),
          chunksCompleted: prev.chunksCompleted + 1
        }));

        setLastProcessingTime(processingTime);

        // Add to comparison results
        if (onResult) {
          onResult({
            id: Date.now(),
            timestamp: new Date(),
            type: 'streaming',
            service: 'Mesolitica',
            transcript: newText,
            confidence: data.confidence,
            processingTime,
            duration: 15,
            chunkNumber,
            audioLength: 15
          });
        }

      } else {
        console.warn(`⚠️ Chunk ${chunkNumber}: No transcript or error`);

        // Still update stats even if no transcript
        setStreamingStats(prev => ({
          ...prev,
          chunksProcessing: Math.max(0, prev.chunksProcessing - 1),
          chunksCompleted: prev.chunksCompleted + 1
        }));
      }

    } catch (error) {
      console.error(`❌ Error processing chunk ${chunkNumber}:`, error);
      setConnectionStatus('error');

      // Update stats on error
      setStreamingStats(prev => ({
        ...prev,
        chunksProcessing: Math.max(0, prev.chunksProcessing - 1)
      }));
    } finally {
      isProcessingRef.current = false;
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
    setConnectionStatus('disconnected');
    console.log('⏹️ Streaming stopped');
  };

  const monitorAudioLevel = () => {
    if (!analyserRef.current) return;

    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);

    const updateLevel = () => {
      if (!isRecording) return;

      analyserRef.current.getByteFrequencyData(dataArray);
      const average = dataArray.reduce((a, b) => a + b) / dataArray.length;
      setAudioLevel(average / 255 * 100);

      // Simple VAD
      const isSpeech = average > 15;
      setStreamingStats(prev => ({
        ...prev,
        speechDetected: isSpeech,
        silenceCounter: isSpeech ? 0 : prev.silenceCounter + 1
      }));

      requestAnimationFrame(updateLevel);
    };

    updateLevel();
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConnectionColor = () => {
    if (connectionStatus === 'connected') return 'text-green-600 bg-green-50';
    if (connectionStatus === 'connecting') return 'text-yellow-600 bg-yellow-50';
    if (connectionStatus === 'error') return 'text-red-600 bg-red-50';
    return 'text-gray-600 bg-gray-50';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Mesolitica Streaming</h2>
        <p className="text-gray-600">Real-time Malaysian speech transcription with HTTP streaming</p>
      </div>

      {/* Streaming Dashboard */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className={`text-xl font-bold px-3 py-1 rounded-full inline-block ${getConnectionColor()}`}>
              {connectionStatus.toUpperCase()}
            </div>
            <div className="text-sm text-gray-600 mt-1">Streaming Status</div>
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{formatDuration(duration)}</div>
            <div className="text-sm text-gray-600">Duration</div>
          </div>

          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {streamingStats.bufferDuration.toFixed(1)}s
            </div>
            <div className="text-sm text-gray-600">Buffer</div>
          </div>

          <div className="text-center">
            <div className={`text-xl font-bold ${streamingStats.speechDetected ? 'text-green-600' : 'text-gray-400'}`}>
              {streamingStats.speechDetected ? 'SPEECH' : 'SILENCE'}
            </div>
            <div className="text-sm text-gray-600">VAD Status</div>
          </div>
        </div>

        {/* Chunks Progress */}
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="text-sm font-medium text-gray-700">
                Chunks: {streamingStats.chunksCompleted}/{streamingStats.chunksCreated}
              </div>
              {streamingStats.chunksProcessing > 0 && (
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <span className="text-sm text-blue-600">Processing {streamingStats.chunksProcessing} chunks...</span>
                </div>
              )}
            </div>
            {lastProcessingTime > 0 && (
              <div className="text-sm text-gray-600">
                Last chunk: {lastProcessingTime}ms
              </div>
            )}
          </div>

          {/* Progress Bar */}
          {streamingStats.chunksCreated > 0 && (
            <div className="mt-2 bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{
                  width: `${(streamingStats.chunksCompleted / streamingStats.chunksCreated) * 100}%`
                }}
              ></div>
            </div>
          )}
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
                  <span>Streaming Active</span>
                </>
              ) : (
                <>
                  <MicOff className="h-4 w-4" />
                  <span>Ready</span>
                </>
              )}
            </div>

            {isRecording && (
              <div className="text-xs text-gray-600">
                Next chunk in: {15 - (duration % 15)}s
              </div>
            )}
          </div>

          {/* Audio Level Visualization */}
          {isRecording && (
            <div className="flex justify-center">
              <div className="flex items-end space-x-1 h-20">
                {[1, 2, 3, 4, 5, 6, 7].map((i) => (
                  <div
                    key={i}
                    className="w-4 bg-gradient-to-t from-blue-600 to-blue-400 rounded-t transition-all duration-100"
                    style={{
                      height: `${Math.max(10, audioLevel * 0.8 + Math.random() * 10)}px`,
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
                <Target className="h-6 w-6" />
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

      {/* Live Transcription Results */}
      {(partialTranscript || fullTranscript || isRecording) && (
        <div className="space-y-4">
          {/* Recent Chunks */}
          {recentChunks.length > 0 && (
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Recent Chunks</h3>
              <div className="space-y-2">
                {recentChunks.map((chunk) => (
                  <div key={chunk.id} className="flex items-start space-x-3 text-sm">
                    <span className="text-xs text-gray-500">{chunk.timestamp}</span>
                    <span className="flex-1 text-gray-800">{chunk.text}</span>
                    <span className="text-xs text-blue-600">{(chunk.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Full Transcript */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Live Transcript</h3>
              <div className="flex items-center space-x-3">
                {confidence > 0 && (
                  <div className="text-sm text-gray-600">
                    Confidence: {(confidence * 100).toFixed(1)}%
                  </div>
                )}
                <div className="px-3 py-1 rounded-full text-sm font-medium bg-blue-50 text-blue-600">
                  Mesolitica • 88% accuracy
                </div>
              </div>
            </div>

            {/* Streaming indicator */}
            {isRecording && (
              <div className="mb-4 p-3 bg-green-50 rounded-lg">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Zap className="h-4 w-4 text-green-600 animate-pulse" />
                    <span className="text-sm font-medium text-green-900">
                      Live streaming active • Words appear every 15 seconds
                    </span>
                  </div>
                </div>
              </div>
            )}

            {/* Transcript */}
            <div className="max-h-64 overflow-y-auto bg-gray-50 rounded-lg p-4">
              <p className="text-lg leading-relaxed text-gray-900 whitespace-pre-wrap">
                {fullTranscript || 'Start speaking to see real-time transcription...'}
              </p>
              {partialTranscript && partialTranscript !== fullTranscript && (
                <span className="text-gray-400 animate-pulse"> {partialTranscript}</span>
              )}
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4 pt-4 border-t border-gray-200">
              <div className="text-center">
                <div className="text-xl font-bold text-blue-600">{duration}s</div>
                <div className="text-xs text-gray-600">Duration</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-green-600">
                  {fullTranscript.split(' ').filter(w => w).length}
                </div>
                <div className="text-xs text-gray-600">Words</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-purple-600">
                  {streamingStats.chunksCompleted}
                </div>
                <div className="text-xs text-gray-600">Chunks</div>
              </div>
              <div className="text-center">
                <div className="text-xl font-bold text-orange-600">88%</div>
                <div className="text-xs text-gray-600">Accuracy</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MesoliticaStreaming;