import React, { useState, useEffect } from 'react';
import LiveTranscription from './components/LiveTranscription';
import MesoliticaStreaming from './components/MesoliticaStreaming';
import SimpleStreaming from './components/SimpleStreaming';
import FileUpload from './components/FileUpload';
import ServiceStatus from './components/ServiceStatus';
import ComparisonDashboard from './components/ComparisonDashboard';
import RecordingBrowser from './components/RecordingBrowser';
import TranscriptionViewer from './components/TranscriptionViewer';
import { Mic, Upload, BarChart3, Settings, Zap, Target, Radio, FileAudio, Languages } from 'lucide-react';
import API_URL from './config';

function App() {
  const [activeTab, setActiveTab] = useState('live');
  const [serverStatus, setServerStatus] = useState(null);
  const [comparisonResults, setComparisonResults] = useState([]);

  useEffect(() => {
    checkServerStatus();
    const interval = setInterval(checkServerStatus, 30000); // Check every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const checkServerStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/health`);
      const data = await response.json();
      setServerStatus(data);
    } catch (error) {
      console.error('Error checking server status:', error);
      setServerStatus({ status: 'error', error: 'Server not reachable' });
    }
  };

  const addComparisonResult = (result) => {
    setComparisonResults(prev => [result, ...prev.slice(0, 9)]); // Keep last 10 results
  };

  const tabs = [
    { id: 'live', label: 'Live Transcription', icon: Mic },
    { id: 'streaming', label: 'Mesolitica Streaming', icon: Target },
    { id: 'simple', label: 'Simple Streaming', icon: Radio },
    { id: 'upload', label: 'File Upload', icon: Upload },
    { id: 'recordings', label: 'Khutbah Recordings', icon: FileAudio },
    { id: 'transcriptions', label: 'Transcription Viewer', icon: Languages },
    { id: 'comparison', label: 'Accuracy Comparison', icon: BarChart3 },
    { id: 'status', label: 'Service Status', icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Target className="h-8 w-8 text-blue-600" />
                <div>
                  <h1 className="text-2xl font-bold text-gray-900">
                    Enhanced Malaysian Speech-to-Text
                  </h1>
                  <p className="text-sm text-gray-600">
                    From 60% to 95% accuracy • Real-time testing platform
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {serverStatus && (
                <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
                  serverStatus.status === 'healthy'
                    ? 'bg-green-100 text-green-800'
                    : 'bg-red-100 text-red-800'
                }`}>
                  <div className={`w-2 h-2 rounded-full ${
                    serverStatus.status === 'healthy' ? 'bg-green-600' : 'bg-red-600'
                  }`}></div>
                  {serverStatus.status === 'healthy' ? 'Server Online' : 'Server Offline'}
                </div>
              )}

              <div className="flex items-center space-x-1 text-sm text-gray-600">
                <Zap className="h-4 w-4 text-yellow-500" />
                <span className="font-medium">Live Testing</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-8">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-5 w-5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Accuracy Banner */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-3">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex items-center justify-center space-x-8 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-400 rounded-full"></div>
              <span>Current System: ~60% accuracy</span>
            </div>
            <div className="text-yellow-300">→</div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <span>Enhanced System: Up to 95% accuracy</span>
            </div>
            <div className="flex items-center space-x-1">
              <Zap className="h-4 w-4" />
              <span className="font-semibold">+35% Improvement</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'live' && (
          <LiveTranscription
            serverStatus={serverStatus}
            onResult={addComparisonResult}
          />
        )}

        {activeTab === 'streaming' && (
          <MesoliticaStreaming
            serverStatus={serverStatus}
            onResult={addComparisonResult}
          />
        )}

        {activeTab === 'simple' && (
          <SimpleStreaming
            serverStatus={serverStatus}
            onResult={addComparisonResult}
          />
        )}

        {activeTab === 'upload' && (
          <FileUpload
            serverStatus={serverStatus}
            onResult={addComparisonResult}
          />
        )}

        {activeTab === 'recordings' && (
          <RecordingBrowser
            serverStatus={serverStatus}
            onResult={addComparisonResult}
          />
        )}

        {activeTab === 'transcriptions' && (
          <TranscriptionViewer />
        )}

        {activeTab === 'comparison' && (
          <ComparisonDashboard
            results={comparisonResults}
            serverStatus={serverStatus}
          />
        )}

        {activeTab === 'status' && (
          <ServiceStatus
            serverStatus={serverStatus}
            onRefresh={checkServerStatus}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-50 border-t border-gray-200 py-8 mt-16">
        <div className="max-w-7xl mx-auto px-4 text-center">
          <div className="flex items-center justify-center space-x-6 text-sm text-gray-600">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-blue-600" />
              <span>Enhanced Malaysian STT</span>
            </div>
            <span>•</span>
            <span>Real-time accuracy testing</span>
            <span>•</span>
            <span>ElevenLabs • Mesolitica • Whisper</span>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Test all services to find the best accuracy for your Malaysian speech recognition needs
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;