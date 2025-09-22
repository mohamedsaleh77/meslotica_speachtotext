import React, { useState, useEffect } from 'react';
import { FileText, Globe, Search, Calendar, ChevronRight, Loader2, AlertCircle, Languages, Download, RotateCcw } from 'lucide-react';
import API_URL from '../config';

const TranscriptionViewer = () => {
  const [transcriptions, setTranscriptions] = useState([]);
  const [selectedTranscription, setSelectedTranscription] = useState(null);
  const [translations, setTranslations] = useState({});
  const [loading, setLoading] = useState(true);
  const [translating, setTranslating] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('malay');

  // Language configuration
  const languages = {
    malay: { name: 'Bahasa Malay', flag: '🇲🇾', code: 'ms' },
    english: { name: 'English', flag: '🇬🇧', code: 'en' },
    tamil: { name: 'Tamil', flag: '🇮🇳', code: 'ta' },
    chinese: { name: 'Chinese', flag: '🇨🇳', code: 'zh' }
  };

  useEffect(() => {
    fetchTranscriptions();
  }, []);

  const fetchTranscriptions = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_URL}/api/transcripts`);
      const data = await response.json();

      if (data.success) {
        // Sort by date, newest first
        const sorted = data.transcripts.sort((a, b) => {
          return new Date(b.saved_at || b.timestamp || 0) - new Date(a.saved_at || a.timestamp || 0);
        });
        setTranscriptions(sorted);
      } else {
        setError(data.error || 'Failed to fetch transcriptions');
      }
    } catch (err) {
      setError('Failed to connect to server');
      console.error('Error fetching transcriptions:', err);
    } finally {
      setLoading(false);
    }
  };

  const selectTranscription = async (transcript) => {
    try {
      setSelectedTranscription(transcript);
      setSelectedLanguage('malay');
      setTranslations({});

      // Fetch full content if needed
      if (transcript.filename) {
        const response = await fetch(`${API_URL}/api/transcripts/${transcript.filename}`);
        const data = await response.json();

        if (data.success && data.content) {
          setSelectedTranscription({ ...transcript, ...data.content });
        }
      }
    } catch (err) {
      console.error('Error fetching transcript details:', err);
    }
  };

  const translateContent = async (targetLanguage) => {
    if (!selectedTranscription || !selectedTranscription.transcript) return;

    // Check if already translated
    if (translations[targetLanguage]) {
      setSelectedLanguage(targetLanguage);
      return;
    }

    try {
      setTranslating(true);
      setError(null);

      const formData = new FormData();
      formData.append('text', selectedTranscription.transcript);
      formData.append('target_language', targetLanguage);
      formData.append('source_language', 'malay');

      const response = await fetch(`${API_URL}/api/translate`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.success) {
        setTranslations(prev => ({
          ...prev,
          [targetLanguage]: result.translation
        }));
        setSelectedLanguage(targetLanguage);
      } else {
        setError(result.error || 'Translation failed');
      }
    } catch (err) {
      setError('Failed to translate content');
      console.error('Translation error:', err);
    } finally {
      setTranslating(false);
    }
  };

  const translateAll = async () => {
    if (!selectedTranscription || !selectedTranscription.transcript) return;

    try {
      setTranslating(true);
      setError(null);

      const formData = new FormData();
      formData.append('text', selectedTranscription.transcript);
      formData.append('source_language', 'malay');

      const response = await fetch(`${API_URL}/api/translate/multiple`, {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (result.success && result.translations) {
        setTranslations(result.translations);
      } else {
        setError(result.error || 'Translation failed');
      }
    } catch (err) {
      setError('Failed to translate content');
      console.error('Translation error:', err);
    } finally {
      setTranslating(false);
    }
  };

  const downloadTranscription = () => {
    if (!selectedTranscription) return;

    const content = {
      original: selectedTranscription.transcript,
      translations: translations,
      metadata: {
        filename: selectedTranscription.filename || selectedTranscription.audio_filename,
        saved_at: selectedTranscription.saved_at,
        service: selectedTranscription.service,
        confidence: selectedTranscription.confidence
      }
    };

    const blob = new Blob([JSON.stringify(content, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedTranscription.filename || 'transcription'}_translations.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const getCurrentContent = () => {
    if (selectedLanguage === 'malay') {
      return selectedTranscription?.transcript || '';
    }
    return translations[selectedLanguage] || '';
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown date';
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatDuration = (duration) => {
    if (!duration) return 'N/A';
    const minutes = Math.floor(duration / 60);
    const seconds = Math.floor(duration % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const filteredTranscriptions = transcriptions.filter(t => {
    const searchLower = searchTerm.toLowerCase();
    const transcript = t.transcript || '';
    const filename = t.filename || t.audio_filename || '';

    return transcript.toLowerCase().includes(searchLower) ||
           filename.toLowerCase().includes(searchLower);
  });

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-8 text-center">
        <Loader2 className="w-8 h-8 animate-spin text-blue-500 mx-auto mb-4" />
        <p className="text-gray-300">Loading transcriptions...</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Languages className="w-8 h-8 text-purple-500" />
          <h2 className="text-2xl font-bold text-white">Transcription Viewer</h2>
        </div>
        <button
          onClick={fetchTranscriptions}
          className="p-2 rounded-lg bg-gray-700 hover:bg-gray-600 transition-colors"
        >
          <RotateCcw className="w-5 h-5 text-gray-300" />
        </button>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4 mb-4">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <p className="text-red-400">{error}</p>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Transcription List */}
        <div className="space-y-4">
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search transcriptions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-700 text-white rounded-lg focus:ring-2 focus:ring-purple-500 outline-none"
            />
          </div>

          {/* Transcription Items */}
          <div className="space-y-2 max-h-[600px] overflow-y-auto custom-scrollbar">
            {filteredTranscriptions.length === 0 ? (
              <div className="text-center py-8 text-gray-400">
                <FileText className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No transcriptions found</p>
              </div>
            ) : (
              filteredTranscriptions.map((transcript, index) => (
                <button
                  key={index}
                  onClick={() => selectTranscription(transcript)}
                  className={`w-full text-left p-4 rounded-lg transition-all ${
                    selectedTranscription?.filename === transcript.filename
                      ? 'bg-purple-900/30 border border-purple-500/50'
                      : 'bg-gray-700 hover:bg-gray-600'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <FileText className="w-4 h-4 text-purple-400" />
                        <p className="text-sm font-medium text-white truncate">
                          {transcript.filename || transcript.audio_filename || 'Untitled'}
                        </p>
                      </div>
                      <p className="text-xs text-gray-400 line-clamp-2">
                        {transcript.transcript?.substring(0, 100)}...
                      </p>
                      <div className="flex items-center gap-4 mt-2">
                        <div className="flex items-center gap-1">
                          <Calendar className="w-3 h-3 text-gray-500" />
                          <span className="text-xs text-gray-400">
                            {formatDate(transcript.saved_at || transcript.timestamp)}
                          </span>
                        </div>
                        {transcript.confidence && (
                          <span className="text-xs text-green-400">
                            {(transcript.confidence * 100).toFixed(1)}%
                          </span>
                        )}
                      </div>
                    </div>
                    <ChevronRight className="w-5 h-5 text-gray-400 ml-2" />
                  </div>
                </button>
              ))
            )}
          </div>
        </div>

        {/* Content Viewer */}
        <div className="space-y-4">
          {selectedTranscription ? (
            <>
              {/* Language Selector */}
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium text-gray-300">Select Language</h3>
                  <button
                    onClick={translateAll}
                    disabled={translating}
                    className="px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  >
                    {translating ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Translating...
                      </>
                    ) : (
                      <>
                        <Globe className="w-4 h-4" />
                        Translate All
                      </>
                    )}
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  {Object.entries(languages).map(([key, lang]) => (
                    <button
                      key={key}
                      onClick={() => key === 'malay' ? setSelectedLanguage('malay') : translateContent(key)}
                      disabled={translating}
                      className={`p-3 rounded-lg transition-all flex items-center gap-2 ${
                        selectedLanguage === key
                          ? 'bg-purple-600 text-white'
                          : 'bg-gray-600 hover:bg-gray-500 text-gray-200'
                      } disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                      <span className="text-xl">{lang.flag}</span>
                      <span className="text-sm font-medium">{lang.name}</span>
                      {translations[key] && key !== 'malay' && (
                        <span className="ml-auto text-xs bg-green-600 px-2 py-0.5 rounded">Cached</span>
                      )}
                    </button>
                  ))}
                </div>
              </div>

              {/* Content Display */}
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-lg font-medium text-white">
                    {languages[selectedLanguage].name} Content
                  </h3>
                  <button
                    onClick={downloadTranscription}
                    className="p-2 bg-gray-600 hover:bg-gray-500 rounded-lg transition-colors"
                  >
                    <Download className="w-4 h-4 text-gray-300" />
                  </button>
                </div>
                <div className="bg-gray-800 rounded-lg p-4 max-h-[400px] overflow-y-auto custom-scrollbar">
                  {translating && selectedLanguage !== 'malay' ? (
                    <div className="text-center py-8">
                      <Loader2 className="w-8 h-8 animate-spin text-purple-500 mx-auto mb-2" />
                      <p className="text-gray-400">Translating to {languages[selectedLanguage].name}...</p>
                    </div>
                  ) : (
                    <pre className="text-gray-200 whitespace-pre-wrap text-sm font-mono leading-relaxed">
                      {getCurrentContent() || 'No content available'}
                    </pre>
                  )}
                </div>
              </div>

              {/* Metadata */}
              {selectedTranscription && (
                <div className="bg-gray-700 rounded-lg p-4">
                  <h3 className="text-sm font-medium text-gray-300 mb-2">Metadata</h3>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Service:</span>
                      <span className="text-gray-200">{selectedTranscription.service || 'Unknown'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Confidence:</span>
                      <span className="text-green-400">
                        {selectedTranscription.confidence ? `${(selectedTranscription.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Word Count:</span>
                      <span className="text-gray-200">{selectedTranscription.word_count || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Characters:</span>
                      <span className="text-gray-200">{selectedTranscription.character_count || 'N/A'}</span>
                    </div>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="bg-gray-700 rounded-lg p-8 text-center">
              <FileText className="w-16 h-16 text-gray-500 mx-auto mb-4" />
              <p className="text-gray-400">Select a transcription to view its content</p>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(0, 0, 0, 0.2);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(139, 92, 246, 0.5);
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(139, 92, 246, 0.7);
        }
        .line-clamp-2 {
          display: -webkit-box;
          -webkit-line-clamp: 2;
          -webkit-box-orient: vertical;
          overflow: hidden;
        }
      `}</style>
    </div>
  );
};

export default TranscriptionViewer;