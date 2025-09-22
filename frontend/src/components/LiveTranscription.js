import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Square, Play, Pause, Volume2, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import API_URL from '../config';

const LiveTranscription = ({ serverStatus, onResult }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [currentService, setCurrentService] = useState('auto');
  const [selectedService, setSelectedService] = useState('mesolitica'); // Default to Mesolitica for live
  const [audioLevel, setAudioLevel] = useState(0);
  const [duration, setDuration] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [liveTranscript, setLiveTranscript] = useState('');
  const [chunkCount, setChunkCount] = useState(0);
  const [processingStatus, setProcessingStatus] = useState('ready');

  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const streamRef = useRef(null);
  const chunksRef = useRef([]);
  const intervalRef = useRef(null);
  const liveChunksRef = useRef([]);
  const chunkProcessingRef = useRef(false);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && isRecording) {
        mediaRecorderRef.current.stop();
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close();
      }
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isRecording]);

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

      // Setup media recorder for LIVE TRANSCRIPTION
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      liveChunksRef.current = [];
      chunksRef.current = [];

      // Handle live chunks for real-time transcription
      mediaRecorderRef.current.ondataavailable = async (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);

          // Process chunk immediately for live transcription
          if (!chunkProcessingRef.current) {
            chunkProcessingRef.current = true;
            setProcessingStatus('processing');

            const audioBlob = new Blob([event.data], { type: 'audio/webm' });
            await processLiveChunk(audioBlob);

            chunkProcessingRef.current = false;
            setProcessingStatus('ready');
          }
        }
      };

      mediaRecorderRef.current.onstop = () => {
        // Final cleanup when recording stops
        setProcessingStatus('stopped');
      };

      setIsRecording(true);
      setDuration(0);
      setLiveTranscript('');
      setTranscript('');
      setConfidence(0);
      setChunkCount(0);
      setProcessingStatus('recording');

      // Start recording with 15-second chunks for live processing
      mediaRecorderRef.current.start(15000); // 15 second chunks

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
    setIsPaused(false);
    setAudioLevel(0);
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      if (isPaused) {
        mediaRecorderRef.current.resume();
        intervalRef.current = setInterval(() => {
          setDuration(prev => prev + 1);
        }, 1000);
      } else {
        mediaRecorderRef.current.pause();
        if (intervalRef.current) {
          clearInterval(intervalRef.current);
        }
      }
      setIsPaused(!isPaused);
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

  const processLiveChunk = async (audioBlob) => {
    try {
      const formData = new FormData();
      formData.append('file', audioBlob, `live_chunk_${chunkCount}.webm`);
      formData.append('language', 'ms');
      formData.append('force_service', selectedService);

      const startTime = Date.now();
      const response = await fetch(`${API_URL}/api/transcribe`, {
        method: 'POST',
        body: formData
      });

      const processingTime = Date.now() - startTime;
      const data = await response.json();

      if (data.success && data.transcript.trim()) {
        // Update live transcript with new chunk result
        setLiveTranscript(prev => {
          const newText = prev + (prev ? ' ' : '') + data.transcript.trim();
          return newText;
        });

        setConfidence(data.confidence || 0);
        setCurrentService(data.service);
        setChunkCount(prev => prev + 1);

        console.log(`✅ Live chunk ${chunkCount} processed: "${data.transcript.trim()}"`);
      } else if (data.error) {
        console.warn(`⚠️ Live chunk ${chunkCount} failed:`, data.error);
      }
    } catch (error) {
      console.error(`❌ Error processing live chunk ${chunkCount}:`, error);
    }
  };

  const sendAudioForTranscription = async (audioBlob) => {
    setIsProcessing(true);

    try {
      const formData = new FormData();
      formData.append('file', audioBlob, 'recording.webm');
      formData.append('language', 'ms');
      formData.append('force_service', selectedService);

      const startTime = Date.now();
      const response = await fetch(`${API_URL}/api/transcribe`, {
        method: 'POST',
        body: formData
      });

      const processingTime = Date.now() - startTime;
      const data = await response.json();

      if (data.success) {
        setTranscript(data.transcript);
        setConfidence(data.confidence || 0);
        setCurrentService(data.service);

        // Add to comparison results
        const result = {
          id: Date.now(),
          timestamp: new Date(),
          type: 'live',
          service: data.service,
          transcript: data.transcript,
          confidence: data.confidence,
          processingTime,
          duration,
          wordErrorRate: data.word_error_rate,
          audioLength: duration
        };

        onResult(result);
      } else {
        console.error('Transcription error:', data.error);
        setTranscript(`Error: ${data.error}`);
      }
    } catch (error) {
      console.error('Error sending audio:', error);
      setTranscript(`Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getServiceColor = (service) => {
    if (service.includes('ElevenLabs')) return 'text-green-600 bg-green-50';
    if (service.includes('Mesolitica')) return 'text-blue-600 bg-blue-50';
    if (service.includes('Whisper')) return 'text-orange-600 bg-orange-50';
    return 'text-gray-600 bg-gray-50';
  };

  const getServiceAccuracy = (service) => {
    if (service.includes('ElevenLabs')) return '~95%';
    if (service.includes('Mesolitica')) return '~88%';
    if (service.includes('Whisper')) return '~75%';
    return 'Unknown';
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Live Speech Transcription</h2>
        <p className="text-gray-600">Test real-time Malaysian speech recognition with different AI services</p>
      </div>

      {/* Service Selection */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Transcription Service</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { value: 'auto', label: 'Smart Auto', description: 'Best available service', accuracy: 'Adaptive' },
            { value: 'elevenlabs', label: 'ElevenLabs Scribe', description: 'Highest accuracy', accuracy: '~95%' },
            { value: 'mesolitica', label: 'Mesolitica Malaysian', description: 'Malaysian-specific', accuracy: '~88%' },
            { value: 'whisper', label: 'OpenAI Whisper', description: 'Fallback option', accuracy: '~75%' }
          ].map((service) => (
            <div
              key={service.value}
              className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
                selectedService === service.value
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => setSelectedService(service.value)}
            >
              <div className="flex items-center space-x-2 mb-2">
                <input
                  type="radio"
                  checked={selectedService === service.value}
                  onChange={() => setSelectedService(service.value)}
                  className="text-blue-600"
                />
                <span className="font-medium text-gray-900">{service.label}</span>
              </div>
              <p className="text-sm text-gray-600 mb-1">{service.description}</p>
              <p className="text-xs font-medium text-blue-600">{service.accuracy} accuracy</p>
            </div>
          ))}
        </div>
      </div>

      {/* Recording Controls */}
      <div className="bg-white rounded-lg shadow-md p-8">
        <div className="text-center space-y-6">
          {/* Status Display */}
          <div className="space-y-2">
            <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full text-sm font-medium ${
              isRecording
                ? isPaused
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-red-100 text-red-800'
                : 'bg-gray-100 text-gray-800'
            }`}>
              {isRecording ? (
                isPaused ? (
                  <>
                    <Pause className="h-4 w-4" />
                    <span>Paused</span>
                  </>
                ) : (
                  <>
                    <div className="w-2 h-2 bg-red-600 rounded-full recording-pulse"></div>
                    <span>Recording</span>
                  </>
                )
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
          {isRecording && !isPaused && (
            <div className="flex justify-center">
              <div className="waveform">
                {[1, 2, 3, 4, 5].map((i) => (
                  <div
                    key={i}
                    className="waveform-bar"
                    style={{
                      height: `${Math.max(10, audioLevel * 0.8)}px`,
                      animationDelay: `${i * 0.1}s`
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
                <span className="text-lg font-medium">Start Recording</span>
              </button>
            ) : (
              <div className="flex space-x-4">
                <button
                  onClick={pauseRecording}
                  className="flex items-center space-x-2 px-6 py-3 bg-yellow-600 text-white rounded-full hover:bg-yellow-700 transition-colors"
                >
                  {isPaused ? <Play className="h-5 w-5" /> : <Pause className="h-5 w-5" />}
                  <span>{isPaused ? 'Resume' : 'Pause'}</span>
                </button>

                <button
                  onClick={stopRecording}
                  className="flex items-center space-x-2 px-6 py-3 bg-red-600 text-white rounded-full hover:bg-red-700 transition-colors"
                >
                  <Square className="h-5 w-5" />
                  <span>Stop & Transcribe</span>
                </button>
              </div>
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
      {(liveTranscript || isRecording || transcript || isProcessing) && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              {isRecording ? 'Live Transcription' : 'Transcription Result'}
            </h3>
            <div className="flex items-center space-x-3">
              {processingStatus === 'processing' && (
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                  <span className="text-sm text-blue-600">Processing...</span>
                </div>
              )}
              {currentService && (
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${getServiceColor(currentService)}`}>
                  {currentService} • {getServiceAccuracy(currentService)} accuracy
                </div>
              )}
            </div>
          </div>

          {/* Live Processing Status */}
          {isRecording && (
            <div className="mb-4 p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium text-blue-900">
                    Live transcription active • Chunks processed: {chunkCount}
                  </span>
                </div>
                <span className="text-xs text-blue-700">
                  Next chunk in: {15 - (duration % 15)}s
                </span>
              </div>
            </div>
          )}

          {isProcessing ? (
            <div className="flex items-center justify-center py-12">
              <div className="text-center">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-4"></div>
                <p className="text-gray-600">Processing final audio with Mesolitica...</p>
              </div>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Confidence Score */}
              {confidence > 0 && (
                <div className="flex items-center space-x-4">
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
                    <span className="font-medium text-gray-900">
                      {isRecording ? 'Live Transcript' : 'Final Transcript'}
                    </span>
                  </div>
                  {(liveTranscript || transcript) && (
                    <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                      {isRecording ? '🔴 Live' : '💾 Saved'}
                    </span>
                  )}
                </div>
                <div className="max-h-64 overflow-y-auto">
                  <p className="text-lg leading-relaxed text-gray-900 whitespace-pre-wrap">
                    {isRecording
                      ? (liveTranscript || 'Listening... speak to see live transcription')
                      : (transcript || 'No speech detected in the recording.')
                    }
                  </p>
                </div>
              </div>

              {/* Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-gray-200">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{duration}s</div>
                  <div className="text-sm text-gray-600">Duration</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {(liveTranscript || transcript || '').split(' ').length}
                  </div>
                  <div className="text-sm text-gray-600">Words</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">{chunkCount}</div>
                  <div className="text-sm text-gray-600">Chunks</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">88%</div>
                  <div className="text-sm text-gray-600">Mesolitica Accuracy</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

    </div>
  );
};

export default LiveTranscription;