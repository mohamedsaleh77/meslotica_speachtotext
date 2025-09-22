import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, X, File, Play, CheckCircle, AlertCircle, Clock, BarChart } from 'lucide-react';
import API_URL from '../config';

const FileUpload = ({ serverStatus, onResult }) => {
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedService, setSelectedService] = useState('auto');
  const [results, setResults] = useState([]);

  const onDrop = useCallback((acceptedFiles) => {
    const newFiles = acceptedFiles.map(file => ({
      id: Date.now() + Math.random(),
      file,
      name: file.name,
      size: file.size,
      status: 'ready',
      progress: 0,
      result: null,
      error: null
    }));

    setUploadedFiles(prev => [...prev, ...newFiles]);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.webm']
    },
    multiple: true
  });

  const removeFile = (fileId) => {
    setUploadedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  const transcribeFile = async (fileData) => {
    const formData = new FormData();
    formData.append('file', fileData.file);
    formData.append('language', 'ms');
    formData.append('force_service', selectedService);

    const startTime = Date.now();

    try {
      const response = await fetch(`${API_URL}/api/transcribe`, {
        method: 'POST',
        body: formData
      });

      const processingTime = Date.now() - startTime;
      const data = await response.json();

      if (data.success) {
        const result = {
          id: Date.now(),
          timestamp: new Date(),
          type: 'file',
          fileName: fileData.name,
          service: data.service,
          transcript: data.transcript,
          confidence: data.confidence,
          processingTime,
          wordErrorRate: data.word_error_rate,
          fileSize: fileData.size,
          duration: data.duration
        };

        // Add to comparison results
        onResult(result);

        return {
          success: true,
          transcript: data.transcript,
          confidence: data.confidence,
          service: data.service,
          processingTime,
          wordErrorRate: data.word_error_rate
        };
      } else {
        throw new Error(data.error || 'Transcription failed');
      }
    } catch (error) {
      throw new Error(error.message);
    }
  };

  const transcribeAllFiles = async () => {
    if (uploadedFiles.length === 0) return;

    setIsProcessing(true);
    const newResults = [];

    for (let i = 0; i < uploadedFiles.length; i++) {
      const fileData = uploadedFiles[i];

      // Update file status to processing
      setUploadedFiles(prev => prev.map(f =>
        f.id === fileData.id ? { ...f, status: 'processing', progress: 0 } : f
      ));

      try {
        const result = await transcribeFile(fileData);

        // Update file with success result
        setUploadedFiles(prev => prev.map(f =>
          f.id === fileData.id
            ? { ...f, status: 'completed', progress: 100, result, error: null }
            : f
        ));

        newResults.push({ ...result, fileName: fileData.name });

      } catch (error) {
        // Update file with error
        setUploadedFiles(prev => prev.map(f =>
          f.id === fileData.id
            ? { ...f, status: 'error', progress: 0, result: null, error: error.message }
            : f
        ));
      }
    }

    setResults(newResults);
    setIsProcessing(false);
  };

  const transcribeSingleFile = async (fileId) => {
    const fileData = uploadedFiles.find(f => f.id === fileId);
    if (!fileData) return;

    setUploadedFiles(prev => prev.map(f =>
      f.id === fileId ? { ...f, status: 'processing', progress: 0 } : f
    ));

    try {
      const result = await transcribeFile(fileData);

      setUploadedFiles(prev => prev.map(f =>
        f.id === fileId
          ? { ...f, status: 'completed', progress: 100, result, error: null }
          : f
      ));

    } catch (error) {
      setUploadedFiles(prev => prev.map(f =>
        f.id === fileId
          ? { ...f, status: 'error', progress: 0, result: null, error: error.message }
          : f
      ));
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getServiceColor = (service) => {
    if (service?.includes('ElevenLabs')) return 'text-green-600 bg-green-50';
    if (service?.includes('Mesolitica')) return 'text-blue-600 bg-blue-50';
    if (service?.includes('Whisper')) return 'text-orange-600 bg-orange-50';
    return 'text-gray-600 bg-gray-50';
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'ready':
        return <Clock className="h-5 w-5 text-gray-500" />;
      case 'processing':
        return <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>;
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case 'error':
        return <AlertCircle className="h-5 w-5 text-red-600" />;
      default:
        return <File className="h-5 w-5 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">File Upload Transcription</h2>
        <p className="text-gray-600">Upload audio files to test and compare different transcription services</p>
      </div>

      {/* Service Selection */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Transcription Service</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {[
            { value: 'auto', label: 'Smart Auto', description: 'Best available service', accuracy: 'Adaptive', cost: 'Variable' },
            { value: 'elevenlabs', label: 'ElevenLabs Scribe', description: 'Highest accuracy', accuracy: '~95%', cost: '$0.40/hour' },
            { value: 'mesolitica', label: 'Mesolitica Malaysian', description: 'Malaysian-specific', accuracy: '~88%', cost: 'FREE' },
            { value: 'whisper', label: 'OpenAI Whisper', description: 'Fallback option', accuracy: '~75%', cost: 'FREE' }
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
              <div className="space-y-1">
                <p className="text-xs font-medium text-blue-600">{service.accuracy} accuracy</p>
                <p className="text-xs text-gray-500">{service.cost}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* File Upload Area */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            isDragActive
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <input {...getInputProps()} />
          <Upload className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          {isDragActive ? (
            <p className="text-lg text-blue-600">Drop the audio files here...</p>
          ) : (
            <div>
              <p className="text-lg text-gray-600 mb-2">
                Drag & drop audio files here, or click to select
              </p>
              <p className="text-sm text-gray-500">
                Supports: WAV, MP3, M4A, FLAC, OGG, AAC, WebM
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Uploaded Files List */}
      {uploadedFiles.length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-900">
              Uploaded Files ({uploadedFiles.length})
            </h3>
            <div className="space-x-2">
              <button
                onClick={transcribeAllFiles}
                disabled={isProcessing || serverStatus?.status !== 'healthy'}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
              >
                {isProcessing ? 'Processing...' : 'Transcribe All'}
              </button>
              <button
                onClick={() => setUploadedFiles([])}
                disabled={isProcessing}
                className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 disabled:bg-gray-400 transition-colors"
              >
                Clear All
              </button>
            </div>
          </div>

          <div className="space-y-4">
            {uploadedFiles.map((fileData) => (
              <div key={fileData.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    {getStatusIcon(fileData.status)}
                    <div>
                      <p className="font-medium text-gray-900">{fileData.name}</p>
                      <p className="text-sm text-gray-500">{formatFileSize(fileData.size)}</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    {fileData.status === 'ready' && (
                      <button
                        onClick={() => transcribeSingleFile(fileData.id)}
                        disabled={serverStatus?.status !== 'healthy'}
                        className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
                      >
                        <Play className="h-4 w-4" />
                      </button>
                    )}
                    <button
                      onClick={() => removeFile(fileData.id)}
                      disabled={fileData.status === 'processing'}
                      className="p-1 text-gray-500 hover:text-red-600 disabled:text-gray-300 transition-colors"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {/* Progress Bar */}
                {fileData.status === 'processing' && (
                  <div className="mb-3">
                    <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
                      <span>Processing with {selectedService === 'auto' ? 'best available service' : selectedService}...</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full animate-pulse" style={{ width: '100%' }}></div>
                    </div>
                  </div>
                )}

                {/* Results */}
                {fileData.result && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${getServiceColor(fileData.result.service)}`}>
                        {fileData.result.service}
                      </div>
                      <div className="text-sm text-gray-600">
                        {(fileData.result.processingTime / 1000).toFixed(1)}s processing time
                      </div>
                    </div>

                    {/* Confidence */}
                    <div className="flex items-center space-x-4">
                      <span className="text-sm font-medium text-gray-700">Confidence:</span>
                      <div className="flex-1 bg-gray-200 rounded-full h-2">
                        <div
                          className="h-2 rounded-full bg-gradient-to-r from-red-500 via-yellow-500 to-green-500"
                          style={{ width: `${(fileData.result.confidence || 0) * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-gray-700">
                        {((fileData.result.confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </div>

                    {/* Transcript */}
                    <div className="bg-gray-50 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-sm font-medium text-gray-700">Full Transcript</span>
                        {fileData.result.auto_saved && (
                          <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                            💾 Auto-saved
                          </span>
                        )}
                      </div>
                      <div className="max-h-64 overflow-y-auto">
                        <p className="text-gray-900 leading-relaxed whitespace-pre-wrap">
                          {fileData.result.transcript || 'No speech detected in the audio file.'}
                        </p>
                      </div>
                      {fileData.result.saved_files && (
                        <div className="mt-3 pt-3 border-t border-gray-200">
                          <p className="text-xs text-gray-600">
                            📁 Saved: {fileData.result.saved_files.txt?.split('/').pop() || 'transcript.txt'}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Metrics */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4 pt-3 border-t border-gray-200">
                      <div className="text-center">
                        <div className="text-lg font-bold text-blue-600">
                          {((fileData.result.confidence || 0) * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-gray-600">Confidence</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-green-600">
                          {(fileData.result.transcript || '').split(' ').length}
                        </div>
                        <div className="text-xs text-gray-600">Words</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-purple-600">
                          {(fileData.result.processingTime / 1000).toFixed(1)}s
                        </div>
                        <div className="text-xs text-gray-600">Processing</div>
                      </div>
                      {fileData.result.chunks_processed && (
                        <div className="text-center">
                          <div className="text-lg font-bold text-indigo-600">
                            {fileData.result.chunks_processed}
                            {fileData.result.total_chunks && `/${fileData.result.total_chunks}`}
                          </div>
                          <div className="text-xs text-gray-600">Chunks</div>
                        </div>
                      )}
                      <div className="text-center">
                        <div className="text-lg font-bold text-orange-600">
                          {fileData.result.service?.includes('ElevenLabs') ? '95%' :
                           fileData.result.service?.includes('large') ? '92%' :
                           fileData.result.service?.includes('Mesolitica') ? '88%' : '75%'}
                        </div>
                        <div className="text-xs text-gray-600">Expected Accuracy</div>
                      </div>
                    </div>

                    {/* Chunking Info */}
                    {fileData.result.chunking_method && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-gray-600">
                            🧠 Smart chunking: {fileData.result.chunking_method}
                          </span>
                          <span className="text-blue-600 font-medium">
                            Complete audio processed
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* Error */}
                {fileData.error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="flex items-center space-x-2 text-red-800">
                      <AlertCircle className="h-5 w-5" />
                      <span className="font-medium">Error:</span>
                    </div>
                    <p className="text-red-700 mt-1">{fileData.error}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}


      {serverStatus?.status !== 'healthy' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-red-800">
            <AlertCircle className="h-5 w-5" />
            <span className="font-medium">Server not available</span>
          </div>
          <p className="text-red-700 mt-1">
            Please start the backend server with: <code className="bg-red-100 px-2 py-1 rounded">python enhanced_whisper_main.py</code>
          </p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;