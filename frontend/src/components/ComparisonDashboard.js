import React, { useState, useMemo } from 'react';
import { BarChart, TrendingUp, Clock, Target, Award, Zap, Filter, Download, RefreshCw, AlertCircle } from 'lucide-react';

const ComparisonDashboard = ({ results, serverStatus }) => {
  const [selectedService, setSelectedService] = useState('all');
  const [timeFilter, setTimeFilter] = useState('all');
  const [sortBy, setSortBy] = useState('timestamp');

  // Filter and sort results
  const filteredResults = useMemo(() => {
    let filtered = [...results];

    // Filter by service
    if (selectedService !== 'all') {
      filtered = filtered.filter(r =>
        r.service.toLowerCase().includes(selectedService.toLowerCase())
      );
    }

    // Filter by time
    const now = new Date();
    if (timeFilter !== 'all') {
      const timeThresholds = {
        '1h': 60 * 60 * 1000,
        '24h': 24 * 60 * 60 * 1000,
        '7d': 7 * 24 * 60 * 60 * 1000
      };

      const threshold = timeThresholds[timeFilter];
      if (threshold) {
        filtered = filtered.filter(r =>
          now - new Date(r.timestamp) <= threshold
        );
      }
    }

    // Sort results
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'confidence':
          return (b.confidence || 0) - (a.confidence || 0);
        case 'processingTime':
          return (a.processingTime || 0) - (b.processingTime || 0);
        case 'service':
          return (a.service || '').localeCompare(b.service || '');
        case 'timestamp':
        default:
          return new Date(b.timestamp) - new Date(a.timestamp);
      }
    });

    return filtered;
  }, [results, selectedService, timeFilter, sortBy]);

  // Calculate statistics
  const stats = useMemo(() => {
    if (filteredResults.length === 0) {
      return {
        totalTests: 0,
        avgConfidence: 0,
        avgProcessingTime: 0,
        serviceBreakdown: {},
        bestService: null,
        accuracyImprovement: 0
      };
    }

    const totalTests = filteredResults.length;
    const avgConfidence = filteredResults.reduce((sum, r) => sum + (r.confidence || 0), 0) / totalTests;
    const avgProcessingTime = filteredResults.reduce((sum, r) => sum + (r.processingTime || 0), 0) / totalTests;

    // Service breakdown
    const serviceBreakdown = {};
    filteredResults.forEach(r => {
      const serviceName = getServiceName(r.service);
      if (!serviceBreakdown[serviceName]) {
        serviceBreakdown[serviceName] = {
          count: 0,
          avgConfidence: 0,
          avgProcessingTime: 0,
          totalConfidence: 0,
          totalProcessingTime: 0
        };
      }
      serviceBreakdown[serviceName].count++;
      serviceBreakdown[serviceName].totalConfidence += (r.confidence || 0);
      serviceBreakdown[serviceName].totalProcessingTime += (r.processingTime || 0);
    });

    // Calculate averages for each service
    Object.keys(serviceBreakdown).forEach(service => {
      const data = serviceBreakdown[service];
      data.avgConfidence = data.totalConfidence / data.count;
      data.avgProcessingTime = data.totalProcessingTime / data.count;
    });

    // Find best service by confidence
    const bestService = Object.keys(serviceBreakdown).reduce((best, current) => {
      if (!best || serviceBreakdown[current].avgConfidence > serviceBreakdown[best].avgConfidence) {
        return current;
      }
      return best;
    }, null);

    // Calculate accuracy improvement from baseline (75% - Whisper)
    const baselineAccuracy = 0.75;
    const currentAvgAccuracy = avgConfidence;
    const accuracyImprovement = ((currentAvgAccuracy - baselineAccuracy) / baselineAccuracy) * 100;

    return {
      totalTests,
      avgConfidence,
      avgProcessingTime,
      serviceBreakdown,
      bestService,
      accuracyImprovement
    };
  }, [filteredResults]);

  const getServiceName = (service) => {
    if (service.includes('ElevenLabs')) return 'ElevenLabs';
    if (service.includes('Mesolitica')) return 'Mesolitica';
    if (service.includes('Whisper')) return 'Whisper';
    return 'Other';
  };

  const getServiceColor = (serviceName) => {
    const colors = {
      'ElevenLabs': 'bg-green-500',
      'Mesolitica': 'bg-blue-500',
      'Whisper': 'bg-orange-500',
      'Other': 'bg-gray-500'
    };
    return colors[serviceName] || colors['Other'];
  };

  const exportResults = () => {
    const dataStr = JSON.stringify(filteredResults, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `transcription_results_${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const clearResults = () => {
    if (window.confirm('Are you sure you want to clear all results?')) {
      // This would need to be implemented in the parent component
      console.log('Clear results requested');
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">Accuracy Comparison Dashboard</h2>
        <p className="text-gray-600">Analyze and compare performance across different transcription services</p>
      </div>

      {/* Controls */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Service Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Filter by Service</label>
            <select
              value={selectedService}
              onChange={(e) => setSelectedService(e.target.value)}
              className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="all">All Services</option>
              <option value="elevenlabs">ElevenLabs Scribe</option>
              <option value="mesolitica">Mesolitica Malaysian</option>
              <option value="whisper">OpenAI Whisper</option>
            </select>
          </div>

          {/* Time Filter */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Time Period</label>
            <select
              value={timeFilter}
              onChange={(e) => setTimeFilter(e.target.value)}
              className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="all">All Time</option>
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
            </select>
          </div>

          {/* Sort */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="w-full rounded-lg border-gray-300 focus:border-blue-500 focus:ring-blue-500"
            >
              <option value="timestamp">Latest First</option>
              <option value="confidence">Highest Confidence</option>
              <option value="processingTime">Fastest Processing</option>
              <option value="service">Service Name</option>
            </select>
          </div>

          {/* Actions */}
          <div className="flex items-end space-x-2">
            <button
              onClick={exportResults}
              disabled={filteredResults.length === 0}
              className="flex items-center space-x-1 px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-sm"
            >
              <Download className="h-4 w-4" />
              <span>Export</span>
            </button>
            <button
              onClick={clearResults}
              className="flex items-center space-x-1 px-3 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 text-sm"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Clear</span>
            </button>
          </div>
        </div>
      </div>

      {/* Statistics Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Total Tests */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Tests</p>
              <p className="text-3xl font-bold text-gray-900">{stats.totalTests}</p>
            </div>
            <BarChart className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        {/* Average Confidence */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Confidence</p>
              <p className="text-3xl font-bold text-green-600">
                {(stats.avgConfidence * 100).toFixed(1)}%
              </p>
            </div>
            <Target className="h-8 w-8 text-green-600" />
          </div>
        </div>

        {/* Average Processing Time */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Processing</p>
              <p className="text-3xl font-bold text-purple-600">
                {(stats.avgProcessingTime / 1000).toFixed(1)}s
              </p>
            </div>
            <Clock className="h-8 w-8 text-purple-600" />
          </div>
        </div>

        {/* Accuracy Improvement */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Improvement</p>
              <p className={`text-3xl font-bold ${stats.accuracyImprovement >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {stats.accuracyImprovement >= 0 ? '+' : ''}{stats.accuracyImprovement.toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="h-8 w-8 text-green-600" />
          </div>
        </div>
      </div>

      {/* Service Performance Comparison */}
      {Object.keys(stats.serviceBreakdown).length > 0 && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Service Performance Comparison</h3>
          <div className="space-y-4">
            {Object.entries(stats.serviceBreakdown).map(([serviceName, data]) => (
              <div key={serviceName} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`w-4 h-4 rounded-full ${getServiceColor(serviceName)}`}></div>
                    <h4 className="font-medium text-gray-900">{serviceName}</h4>
                    {stats.bestService === serviceName && (
                      <Award className="h-5 w-5 text-yellow-500" title="Best Performance" />
                    )}
                  </div>
                  <span className="text-sm text-gray-600">{data.count} tests</span>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <p className="text-sm text-gray-600">Avg Confidence</p>
                    <p className="text-xl font-bold text-green-600">
                      {(data.avgConfidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Avg Processing</p>
                    <p className="text-xl font-bold text-purple-600">
                      {(data.avgProcessingTime / 1000).toFixed(1)}s
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Expected Accuracy</p>
                    <p className="text-xl font-bold text-blue-600">
                      {serviceName === 'ElevenLabs' ? '~95%' :
                       serviceName === 'Mesolitica' ? '~88%' :
                       serviceName === 'Whisper' ? '~75%' : 'Unknown'}
                    </p>
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="mt-3">
                  <div className="flex items-center justify-between text-sm text-gray-600 mb-1">
                    <span>Confidence Level</span>
                    <span>{(data.avgConfidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${getServiceColor(serviceName)}`}
                      style={{ width: `${data.avgConfidence * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Results */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Recent Results ({filteredResults.length})
          </h3>
          {filteredResults.length > 0 && (
            <div className="text-sm text-gray-600">
              Showing {filteredResults.length} of {results.length} total
            </div>
          )}
        </div>

        {filteredResults.length === 0 ? (
          <div className="text-center py-12">
            <BarChart className="h-12 w-12 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Yet</h3>
            <p className="text-gray-600 mb-4">
              Start testing with live transcription or file upload to see comparison data here.
            </p>
            <div className="space-y-2 text-sm text-gray-500">
              <p>• Use the Live Transcription tab to record and test different services</p>
              <p>• Use the File Upload tab to test with your audio files</p>
              <p>• Results will automatically appear here for comparison</p>
            </div>
          </div>
        ) : (
          <div className="space-y-4 max-h-96 overflow-y-auto">
            {filteredResults.map((result) => (
              <div key={result.id} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${getServiceColor(getServiceName(result.service))}`}></div>
                    <span className="font-medium text-gray-900">{result.service}</span>
                    <span className="text-sm text-gray-500">
                      {result.type === 'live' ? 'Live Recording' : 'File Upload'}
                    </span>
                    {result.fileName && (
                      <span className="text-sm text-gray-500">• {result.fileName}</span>
                    )}
                  </div>
                  <span className="text-sm text-gray-500">
                    {new Date(result.timestamp).toLocaleString()}
                  </span>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                  <div>
                    <p className="text-xs text-gray-600">Confidence</p>
                    <p className="font-medium text-green-600">
                      {((result.confidence || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Processing</p>
                    <p className="font-medium text-purple-600">
                      {(result.processingTime / 1000).toFixed(1)}s
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Words</p>
                    <p className="font-medium text-blue-600">
                      {(result.transcript || '').split(' ').length}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Duration</p>
                    <p className="font-medium text-gray-600">
                      {result.duration ? `${result.duration}s` :
                       result.audioLength ? `${result.audioLength}s` : 'N/A'}
                    </p>
                  </div>
                </div>

                {/* Transcript Preview */}
                <div className="bg-gray-50 rounded p-3">
                  <p className="text-sm text-gray-900 line-clamp-2">
                    {result.transcript || 'No speech detected'}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recommendations */}
      {stats.totalTests > 0 && (
        <div className="bg-blue-50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-blue-900 mb-3 flex items-center space-x-2">
            <Zap className="h-5 w-5" />
            <span>Smart Recommendations</span>
          </h3>
          <div className="space-y-2 text-sm text-blue-800">
            {stats.bestService && (
              <p>• <strong>{stats.bestService}</strong> is performing best with {(stats.serviceBreakdown[stats.bestService].avgConfidence * 100).toFixed(1)}% average confidence</p>
            )}
            {stats.avgConfidence < 0.8 && (
              <p>• Consider using ElevenLabs Scribe for higher accuracy (up to 95%)</p>
            )}
            {stats.avgProcessingTime > 5000 && (
              <p>• Processing time is high - consider optimizing audio quality or using Mesolitica for faster local processing</p>
            )}
            {stats.accuracyImprovement > 20 && (
              <p>• Excellent improvement! You're getting {stats.accuracyImprovement.toFixed(1)}% better accuracy than baseline</p>
            )}
          </div>
        </div>
      )}

      {serverStatus?.status !== 'healthy' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-red-800">
            <AlertCircle className="h-5 w-5" />
            <span className="font-medium">Server Status Issue</span>
          </div>
          <p className="text-red-700 mt-1">
            Some features may not work properly. Check the Service Status tab for details.
          </p>
        </div>
      )}
    </div>
  );
};

export default ComparisonDashboard;