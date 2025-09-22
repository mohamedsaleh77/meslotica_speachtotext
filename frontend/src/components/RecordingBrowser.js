import React, { useState, useEffect } from 'react';
import { Play, Download, Clock, FileAudio, Mic, Loader2, CheckCircle, AlertCircle, Volume2 } from 'lucide-react';
import API_URL from '../config';

const RecordingBrowser = ({ serverStatus, onResult }) => {
  const [recordings, setRecordings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [transcribing, setTranscribing] = useState(null);
  const [transcripts, setTranscripts] = useState({});
  const [selectedService, setSelectedService] = useState('mesolitica');
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchRecordings();
  }, []);

  const fetchRecordings = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_URL}/api/recordings`);
      const data = await response.json();

      if (data.success) {
        setRecordings(data.recordings);
      } else {
        setError(data.error || 'Failed to fetch recordings');
      }
    } catch (err) {
      setError('Failed to connect to server');
      console.error('Error fetching recordings:', err);
    } finally {
      setLoading(false);
    }
  };

  const transcribeRecording = async (filename) => {
    try {
      setTranscribing(filename);
      setError(null);

      const response = await fetch(`${API_URL}/api/recordings/transcribe/${filename}?service=${selectedService}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      const result = await response.json();

      if (result.success) {
        setTranscripts(prev => ({
          ...prev,
          [filename]: result
        }));

        // Call onResult callback if provided
        if (onResult) {
          onResult({
            ...result,
            filename: filename,
            type: 'recording'
          });
        }
      } else {
        setError(`Transcription failed: ${result.error}`);
      }
    } catch (err) {
      setError('Failed to transcribe recording');
      console.error('Error transcribing recording:', err);
    } finally {
      setTranscribing(null);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (timestamp) => {
    return new Date(timestamp * 1000).toLocaleString();
  };

  const formatDuration = (filename) => {
    // Extract timestamp from filename pattern: chunk_YYYYMMDD_HHMMSS_sss_id.wav
    const match = filename.match(/chunk_(\d{8})_(\d{6})_(\d{3})/);
    if (match) {
      const [, date, time, ms] = match;
      const hour = time.slice(0, 2);
      const minute = time.slice(2, 4);
      const second = time.slice(4, 6);
      return `${hour}:${minute}:${second}.${ms}`;
    }
    return 'Unknown';
  };

  const getServiceStatus = () => {
    if (!serverStatus) return { available: false, status: 'unknown' };
    return {
      available: serverStatus.services?.mesolitica || false,
      status: serverStatus.status || 'unknown'
    };
  };

  const serviceStatus = getServiceStatus();

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="animate-spin h-8 w-8 text-blue-500" />
        <span className="ml-2 text-gray-600">Loading recordings...</span>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-2 flex items-center">
          <FileAudio className="h-6 w-6 mr-2 text-blue-500" />
          Khutbah Recordings
        </h2>
        <p className="text-gray-600">Browse and transcribe audio recordings using Mesolitica</p>
      </div>

      {/* Service Selection and Status */}
      <div className="bg-white rounded-lg shadow-sm border p-4 mb-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <label className="text-sm font-medium text-gray-700">Transcription Service:</label>
            <select
              value={selectedService}
              onChange={(e) => setSelectedService(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-1 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="mesolitica">Mesolitica (Recommended for Malay)</option>
              <option value="whisper">Whisper Large V2</option>
              <option value="auto">Auto (Best Available)</option>
            </select>
          </div>
          <div className="flex items-center space-x-2">
            {serviceStatus.available ? (
              <><CheckCircle className="h-4 w-4 text-green-500" />
              <span className="text-sm text-green-600">Service Ready</span></>
            ) : (
              <><AlertCircle className="h-4 w-4 text-red-500" />
              <span className="text-sm text-red-600">Service Unavailable</span></>
            )}
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-700">{error}</span>
          </div>
        </div>
      )}

      {/* Recordings List */}
      {recordings.length === 0 ? (
        <div className="text-center py-12">
          <FileAudio className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No recordings found</h3>
          <p className="text-gray-600">No audio files found in the recordings directory.</p>
          <button
            onClick={fetchRecordings}
            className="mt-4 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors"
          >
            Refresh
          </button>
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-gray-900">
              {recordings.length} recording{recordings.length !== 1 ? 's' : ''} found
            </h3>
            <button
              onClick={fetchRecordings}
              className="text-blue-500 hover:text-blue-600 text-sm"
            >
              Refresh
            </button>
          </div>

          {recordings.map((recording) => {
            const isTranscribing = transcribing === recording.filename;
            const hasTranscript = transcripts[recording.filename];

            return (
              <div key={recording.filename} className="bg-white rounded-lg shadow-sm border p-4">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3">
                      <Volume2 className="h-5 w-5 text-gray-400" />
                      <div>
                        <h4 className="font-medium text-gray-900">{recording.filename}</h4>
                        <div className="flex items-center space-x-4 text-sm text-gray-600 mt-1">
                          <span className="flex items-center">
                            <Clock className="h-4 w-4 mr-1" />
                            {formatDate(recording.modified)}
                          </span>
                          <span>{formatFileSize(recording.size)}</span>
                          <span>Duration: {formatDuration(recording.filename)}</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    {hasTranscript && (
                      <div className="text-sm text-green-600 flex items-center">
                        <CheckCircle className="h-4 w-4 mr-1" />
                        Transcribed
                      </div>
                    )}
                    <button
                      onClick={() => transcribeRecording(recording.filename)}
                      disabled={isTranscribing || !serviceStatus.available}
                      className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors flex items-center"
                    >
                      {isTranscribing ? (
                        <>
                          <Loader2 className="animate-spin h-4 w-4 mr-2" />
                          Transcribing...
                        </>
                      ) : (
                        <>
                          <Mic className="h-4 w-4 mr-2" />
                          {hasTranscript ? 'Re-transcribe' : 'Transcribe'}
                        </>
                      )}
                    </button>
                  </div>
                </div>

                {/* Transcript Display */}
                {hasTranscript && (
                  <div className="mt-4 p-4 bg-gray-50 rounded-lg border">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-4 text-sm text-gray-600">
                        <span>Service: {hasTranscript.service}</span>
                        <span>Confidence: {(hasTranscript.confidence * 100).toFixed(1)}%</span>
                        <span>Time: {hasTranscript.processing_time?.toFixed(2)}s</span>
                      </div>
                    </div>
                    <div className="bg-white p-3 rounded border">
                      <p className="text-gray-900 whitespace-pre-wrap">{hasTranscript.transcript}</p>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default RecordingBrowser;