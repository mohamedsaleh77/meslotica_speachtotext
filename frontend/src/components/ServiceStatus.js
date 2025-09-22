import React, { useState, useEffect } from 'react';
import { RefreshCw, CheckCircle, XCircle, AlertCircle, Settings, Server, Zap, DollarSign, Clock, Target } from 'lucide-react';
import API_URL from '../config';

const ServiceStatus = ({ serverStatus, onRefresh }) => {
  const [detailedStatus, setDetailedStatus] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  useEffect(() => {
    fetchDetailedStatus();
  }, []);

  const fetchDetailedStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/services/status`);
      if (response.ok) {
        const data = await response.json();
        setDetailedStatus(data);
      }
    } catch (error) {
      console.error('Error fetching detailed status:', error);
    }
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await onRefresh();
    await fetchDetailedStatus();
    setIsRefreshing(false);
  };

  const getStatusIcon = (available) => {
    if (available) {
      return <CheckCircle className="h-6 w-6 text-green-600" />;
    } else {
      return <XCircle className="h-6 w-6 text-red-600" />;
    }
  };

  const getStatusBadge = (available) => {
    if (available) {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
          Available
        </span>
      );
    } else {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
          Unavailable
        </span>
      );
    }
  };

  const services = [
    {
      id: 'elevenlabs',
      name: 'ElevenLabs Scribe',
      description: 'Highest accuracy commercial API',
      accuracy: '≤5% WER (~95% accuracy)',
      cost: '$0.40/hour ($0.22/hour enterprise)',
      speed: 'Fast (cloud processing)',
      features: ['Speaker diarization', 'Word-level timestamps', 'Audio events', 'Highest accuracy'],
      bestFor: 'Critical applications requiring maximum accuracy',
      setup: 'Requires API key from elevenlabs.io'
    },
    {
      id: 'mesolitica',
      name: 'Mesolitica Malaysian Whisper',
      description: 'Malaysian-specific fine-tuned model',
      accuracy: '~12% WER (~88% accuracy)',
      cost: 'FREE (local processing)',
      speed: 'Medium (local GPU/CPU)',
      features: ['Malaysian speech patterns', 'Local processing', 'Privacy-friendly', 'No API costs'],
      bestFor: 'Cost-effective Malaysian-specific transcription',
      setup: 'Automatically downloads model on first use'
    },
    {
      id: 'whisper',
      name: 'OpenAI Whisper Large-v2',
      description: 'General-purpose multilingual model',
      accuracy: '~25% WER (~75% accuracy)',
      cost: 'FREE (local processing)',
      speed: 'Medium (local GPU/CPU)',
      features: ['Multilingual support', 'Robust performance', 'Well-tested', 'Reliable fallback'],
      bestFor: 'General purpose and backup processing',
      setup: 'Automatically downloads model on first use'
    }
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-gray-900">Service Status</h2>
          <p className="text-gray-600 mt-1">Monitor and configure your transcription services</p>
        </div>
        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
        >
          <RefreshCw className={`h-5 w-5 ${isRefreshing ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* Overall System Status */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
          <Server className="h-5 w-5" />
          <span>System Status</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Server Health */}
          <div className="text-center">
            <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-3 ${
              serverStatus?.status === 'healthy' ? 'bg-green-100' : 'bg-red-100'
            }`}>
              {serverStatus?.status === 'healthy' ? (
                <CheckCircle className="h-8 w-8 text-green-600" />
              ) : (
                <XCircle className="h-8 w-8 text-red-600" />
              )}
            </div>
            <h4 className="font-medium text-gray-900">Backend Server</h4>
            <p className={`text-sm ${
              serverStatus?.status === 'healthy' ? 'text-green-600' : 'text-red-600'
            }`}>
              {serverStatus?.status === 'healthy' ? 'Running' : 'Offline'}
            </p>
          </div>

          {/* Available Services */}
          <div className="text-center">
            <div className="mx-auto w-16 h-16 rounded-full bg-blue-100 flex items-center justify-center mb-3">
              <Zap className="h-8 w-8 text-blue-600" />
            </div>
            <h4 className="font-medium text-gray-900">Active Services</h4>
            <p className="text-sm text-blue-600">
              {serverStatus?.services ?
                Object.values(serverStatus.services).filter(Boolean).length : 0} / 3
            </p>
          </div>

          {/* System Ready */}
          <div className="text-center">
            <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-3 ${
              serverStatus?.status === 'healthy' &&
              serverStatus?.services &&
              Object.values(serverStatus.services).some(Boolean)
                ? 'bg-green-100' : 'bg-yellow-100'
            }`}>
              <Target className={`h-8 w-8 ${
                serverStatus?.status === 'healthy' &&
                serverStatus?.services &&
                Object.values(serverStatus.services).some(Boolean)
                  ? 'text-green-600' : 'text-yellow-600'
              }`} />
            </div>
            <h4 className="font-medium text-gray-900">System Ready</h4>
            <p className={`text-sm ${
              serverStatus?.status === 'healthy' &&
              serverStatus?.services &&
              Object.values(serverStatus.services).some(Boolean)
                ? 'text-green-600' : 'text-yellow-600'
            }`}>
              {serverStatus?.status === 'healthy' &&
               serverStatus?.services &&
               Object.values(serverStatus.services).some(Boolean)
                ? 'Ready for testing' : 'Setup required'}
            </p>
          </div>
        </div>
      </div>

      {/* Detailed Service Status */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-gray-900">Service Details</h3>

        {services.map((service) => {
          const isAvailable = serverStatus?.services?.[service.id] || false;
          const serviceDetails = detailedStatus?.[service.id];

          return (
            <div key={service.id} className="bg-white rounded-lg shadow-md p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-start space-x-4">
                  {getStatusIcon(isAvailable)}
                  <div>
                    <h4 className="text-lg font-semibold text-gray-900">{service.name}</h4>
                    <p className="text-gray-600">{service.description}</p>
                  </div>
                </div>
                {getStatusBadge(isAvailable)}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                {/* Accuracy */}
                <div className="bg-green-50 rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <Target className="h-4 w-4 text-green-600" />
                    <span className="text-sm font-medium text-green-800">Accuracy</span>
                  </div>
                  <p className="text-sm text-green-700">{service.accuracy}</p>
                </div>

                {/* Cost */}
                <div className="bg-blue-50 rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <DollarSign className="h-4 w-4 text-blue-600" />
                    <span className="text-sm font-medium text-blue-800">Cost</span>
                  </div>
                  <p className="text-sm text-blue-700">{service.cost}</p>
                </div>

                {/* Speed */}
                <div className="bg-purple-50 rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <Clock className="h-4 w-4 text-purple-600" />
                    <span className="text-sm font-medium text-purple-800">Speed</span>
                  </div>
                  <p className="text-sm text-purple-700">{service.speed}</p>
                </div>

                {/* Best For */}
                <div className="bg-orange-50 rounded-lg p-3">
                  <div className="flex items-center space-x-2 mb-1">
                    <Zap className="h-4 w-4 text-orange-600" />
                    <span className="text-sm font-medium text-orange-800">Best For</span>
                  </div>
                  <p className="text-sm text-orange-700 line-clamp-2">{service.bestFor}</p>
                </div>
              </div>

              {/* Features */}
              <div className="mb-4">
                <h5 className="text-sm font-medium text-gray-900 mb-2">Features</h5>
                <div className="flex flex-wrap gap-2">
                  {service.features.map((feature, index) => (
                    <span
                      key={index}
                      className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800"
                    >
                      {feature}
                    </span>
                  ))}
                </div>
              </div>

              {/* Setup Info */}
              <div className="bg-gray-50 rounded-lg p-3">
                <div className="flex items-center space-x-2 mb-1">
                  <Settings className="h-4 w-4 text-gray-600" />
                  <span className="text-sm font-medium text-gray-800">Setup</span>
                </div>
                <p className="text-sm text-gray-700">{service.setup}</p>

                {service.id === 'elevenlabs' && !isAvailable && (
                  <div className="mt-2 text-sm text-blue-600">
                    <a
                      href="https://elevenlabs.io/pricing/api"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline"
                    >
                      → Get your ElevenLabs API key here
                    </a>
                  </div>
                )}
              </div>

              {/* Additional Details from API */}
              {serviceDetails && (
                <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                  <h5 className="text-sm font-medium text-blue-900 mb-2">Additional Info</h5>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                    <div>
                      <span className="font-medium text-blue-800">Accuracy:</span>
                      <span className="text-blue-700 ml-1">{serviceDetails.accuracy}</span>
                    </div>
                    <div>
                      <span className="font-medium text-blue-800">Cost:</span>
                      <span className="text-blue-700 ml-1">{serviceDetails.cost}</span>
                    </div>
                    <div>
                      <span className="font-medium text-blue-800">Best for:</span>
                      <span className="text-blue-700 ml-1">{serviceDetails.best_for}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Configuration Help */}
      <div className="bg-blue-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-blue-900 mb-3">Setup Instructions</h3>
        <div className="space-y-3 text-sm text-blue-800">
          <div>
            <strong>1. ElevenLabs Scribe (Highest Accuracy):</strong>
            <ul className="list-disc list-inside ml-4 mt-1 space-y-1">
              <li>Go to <a href="https://elevenlabs.io/pricing/api" target="_blank" rel="noopener noreferrer" className="underline">elevenlabs.io/pricing/api</a></li>
              <li>Sign up and get your API key</li>
              <li>Add <code className="bg-blue-100 px-1 rounded">ELEVENLABS_API_KEY=your_key_here</code> to your .env file</li>
              <li>Restart the server</li>
            </ul>
          </div>

          <div>
            <strong>2. Mesolitica Malaysian (Free):</strong>
            <ul className="list-disc list-inside ml-4 mt-1 space-y-1">
              <li>Model downloads automatically on first use</li>
              <li>Requires internet connection for initial download</li>
              <li>Uses local GPU/CPU for processing</li>
            </ul>
          </div>

          <div>
            <strong>3. OpenAI Whisper (Fallback):</strong>
            <ul className="list-disc list-inside ml-4 mt-1 space-y-1">
              <li>Model downloads automatically on first use</li>
              <li>Always available as backup option</li>
              <li>Reliable but lower accuracy for Malaysian speech</li>
            </ul>
          </div>
        </div>
      </div>

      {/* System Info */}
      {serverStatus && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">System Information</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-gray-700">Server Status:</span>
              <span className={`ml-2 ${
                serverStatus.status === 'healthy' ? 'text-green-600' : 'text-red-600'
              }`}>
                {serverStatus.status}
              </span>
            </div>

            {serverStatus.accuracy_estimates && (
              <>
                <div>
                  <span className="font-medium text-gray-700">ElevenLabs Accuracy:</span>
                  <span className="ml-2 text-green-600">{serverStatus.accuracy_estimates.elevenlabs_scribe}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Mesolitica Accuracy:</span>
                  <span className="ml-2 text-blue-600">{serverStatus.accuracy_estimates.mesolitica_malaysian}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Whisper Accuracy:</span>
                  <span className="ml-2 text-orange-600">{serverStatus.accuracy_estimates.whisper_large_v2}</span>
                </div>
              </>
            )}

            {serverStatus.config && (
              <>
                <div>
                  <span className="font-medium text-gray-700">Primary Service:</span>
                  <span className="ml-2 text-gray-600">{serverStatus.config.primary_service}</span>
                </div>
                <div>
                  <span className="font-medium text-gray-700">Language:</span>
                  <span className="ml-2 text-gray-600">{serverStatus.config.language}</span>
                </div>
              </>
            )}

            <div>
              <span className="font-medium text-gray-700">Last Updated:</span>
              <span className="ml-2 text-gray-600">
                {serverStatus.timestamp ? new Date(serverStatus.timestamp).toLocaleString() : 'Unknown'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Error State */}
      {serverStatus?.status !== 'healthy' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-red-800 mb-2">
            <AlertCircle className="h-5 w-5" />
            <span className="font-medium">Server Connection Issue</span>
          </div>
          <p className="text-red-700 mb-3">
            Unable to connect to the backend server. Please ensure the server is running.
          </p>
          <div className="space-y-1 text-sm text-red-600">
            <p>• Check that you've started the server with: <code className="bg-red-100 px-1 rounded">python enhanced_whisper_main.py</code></p>
            <p>• Verify the server is running on port 8000</p>
            <p>• Check the console for any error messages</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ServiceStatus;