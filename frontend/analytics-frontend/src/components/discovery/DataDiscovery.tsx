import React, { useState, useEffect } from 'react';
import { analyticsAPI } from '../../services/api';

interface DataSource {
  id: string;
  source_id: string;
  type: string;
  confidence: number;
  reasoning: string;
  context: any;
  status?: 'discovered' | 'connecting' | 'connected' | 'error';
}

const DataDiscovery: React.FC = () => {
  const [discovering, setDiscovering] = useState(false);
  const [discoveredSources, setDiscoveredSources] = useState<DataSource[]>([]);
  const [selectedMode, setSelectedMode] = useState('balanced');
  const [scanProgress, setScanProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [lastScanTime, setLastScanTime] = useState<string | null>(null);

  const discoveryModes = [
    {
      id: 'fast',
      name: 'Fast Scan',
      description: 'Quick discovery of obvious data sources',
      icon: '⚡',
      speed: 'Very Fast',
      accuracy: 'Good'
    },
    {
      id: 'balanced',
      name: 'Balanced',
      description: 'Good balance of speed and thoroughness',
      icon: '⚖️',
      speed: 'Moderate',
      accuracy: 'Very Good'
    },
    {
      id: 'thorough',
      name: 'Deep Scan',
      description: 'Comprehensive analysis of all possible sources',
      icon: '🔬',
      speed: 'Slower',
      accuracy: 'Excellent'
    }
  ];

  useEffect(() => {
    loadPreviousDiscoveries();
  }, []);

  const loadPreviousDiscoveries = () => {
    try {
      const saved = localStorage.getItem('discovered_sources');
      if (saved) {
        const sources = JSON.parse(saved);
        setDiscoveredSources(sources);
      }
    } catch (error) {
      console.error('Failed to load previous discoveries:', error);
    }
  };

  const startDiscovery = async () => {
    setDiscovering(true);
    setError(null);
    setScanProgress(0);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setScanProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + Math.random() * 15;
        });
      }, 500);

      // Mock discovered sources for demo
      setTimeout(() => {
        clearInterval(progressInterval);
        setScanProgress(100);

        const mockSources: DataSource[] = [
          {
            id: '1',
            source_id: 'production_db',
            type: 'PostgreSQL',
            confidence: 0.95,
            reasoning: 'Active database connection detected on standard port 5432',
            context: { host: 'localhost', port: 5432, database: 'analytics_prod' },
            status: 'discovered'
          },
          {
            id: '2',
            source_id: 'sales_data.csv',
            type: 'CSV File',
            confidence: 0.87,
            reasoning: 'CSV file with structured sales data found in data directory',
            context: { rows: 10000, columns: 15, size: '2.3MB' },
            status: 'discovered'
          },
          {
            id: '3',
            source_id: 'redis_cache',
            type: 'Redis',
            confidence: 0.72,
            reasoning: 'Redis instance detected with active connections',
            context: { host: 'localhost', port: 6379, keys: 1247 },
            status: 'discovered'
          }
        ];

        setDiscoveredSources(mockSources);
        setLastScanTime(new Date().toLocaleString());
        localStorage.setItem('discovered_sources', JSON.stringify(mockSources));

        setTimeout(() => setScanProgress(0), 2000);
      }, 3000);

    } catch (error) {
      setError('Discovery failed. Please check your connection and try again.');
      setScanProgress(0);
    } finally {
      setDiscovering(false);
    }
  };

  const connectToSource = async (source: DataSource) => {
    setDiscoveredSources(sources =>
      sources.map(s => s.id === source.id ? { ...s, status: 'connecting' } : s)
    );

    try {
      // Simulate connection process
      await new Promise(resolve => setTimeout(resolve, 2000));

      setDiscoveredSources(sources =>
        sources.map(s => s.id === source.id ? { ...s, status: 'connected' } : s)
      );
    } catch (error) {
      setDiscoveredSources(sources =>
        sources.map(s => s.id === source.id ? { ...s, status: 'error' } : s)
      );
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-green-600 bg-green-100';
      case 'connecting': return 'text-blue-600 bg-blue-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return '✅';
      case 'connecting': return '🔄';
      case 'error': return '❌';
      default: return '🔍';
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Smart Data Discovery</h1>
        <p className="text-gray-600">
          Automatically discover and connect to available data sources in your environment
        </p>
      </div>

      {/* Discovery Controls */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        <div className="lg:col-span-2">
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Discovery Mode</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {discoveryModes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setSelectedMode(mode.id)}
                  className={`p-4 rounded-lg border-2 transition-all duration-200 text-left ${
                    selectedMode === mode.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <span className="text-2xl">{mode.icon}</span>
                    <h3 className="font-semibold text-gray-900">{mode.name}</h3>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{mode.description}</p>
                  <div className="flex justify-between text-xs">
                    <span className="text-gray-500">Speed: {mode.speed}</span>
                    <span className="text-gray-500">Accuracy: {mode.accuracy}</span>
                  </div>
                </button>
              ))}
            </div>

            <button
              onClick={startDiscovery}
              disabled={discovering}
              className={`btn-primary flex items-center space-x-2 ${
                discovering ? 'opacity-50 cursor-not-allowed' : ''
              }`}
            >
              {discovering ? (
                <>
                  <svg className="animate-spin w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span>Discovering...</span>
                </>
              ) : (
                <>
                  <span>🔍</span>
                  <span>Start Discovery</span>
                </>
              )}
            </button>

            {/* Progress Bar */}
            {scanProgress > 0 && (
              <div className="mt-4">
                <div className="flex justify-between text-sm text-gray-600 mb-1">
                  <span>Discovery Progress</span>
                  <span>{Math.round(scanProgress)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${scanProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Discovery Stats</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Sources Found</span>
                <span className="font-semibold text-gray-900">{discoveredSources.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Connected</span>
                <span className="font-semibold text-green-600">
                  {discoveredSources.filter(s => s.status === 'connected').length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">High Confidence</span>
                <span className="font-semibold text-blue-600">
                  {discoveredSources.filter(s => s.confidence >= 0.8).length}
                </span>
              </div>
              {lastScanTime && (
                <div className="pt-2 border-t border-gray-200">
                  <div className="text-xs text-gray-500">
                    Last scan: {lastScanTime}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                📁 Scan Local Files
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                🌐 Check Network Databases
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                ☁️ Cloud Service Scan
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Discovered Sources */}
      {discoveredSources.length > 0 && (
        <div className="card">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-semibold text-gray-900">Discovered Data Sources</h2>
            <button
              onClick={() => {
                setDiscoveredSources([]);
                localStorage.removeItem('discovered_sources');
              }}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear All
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {discoveredSources.map((source) => (
              <div
                key={source.id}
                className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                      <span className="text-blue-600 font-semibold">
                        {source.type.slice(0, 2).toUpperCase()}
                      </span>
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">{source.source_id}</h3>
                      <p className="text-sm text-gray-500">{source.type}</p>
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getConfidenceColor(source.confidence)}`}>
                      {Math.round(source.confidence * 100)}%
                    </span>
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(source.status || 'discovered')}`}>
                      {getStatusIcon(source.status || 'discovered')} {source.status || 'discovered'}
                    </span>
                  </div>
                </div>

                <p className="text-sm text-gray-600 mb-4">{source.reasoning}</p>

                {source.context && (
                  <div className="mb-4 p-3 bg-gray-50 rounded-lg">
                    <h4 className="text-sm font-medium text-gray-700 mb-1">Context</h4>
                    <div className="text-xs text-gray-600">
                      {Object.entries(source.context).map(([key, value]) => (
                        <div key={key} className="flex justify-between">
                          <span>{key}:</span>
                          <span>{String(value)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex space-x-2">
                  {source.status !== 'connected' && (
                    <button
                      onClick={() => connectToSource(source)}
                      disabled={source.status === 'connecting'}
                      className={`btn-primary text-sm px-4 py-2 ${
                        source.status === 'connecting' ? 'opacity-50 cursor-not-allowed' : ''
                      }`}
                    >
                      {source.status === 'connecting' ? 'Connecting...' : 'Connect'}
                    </button>
                  )}

                  <button className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium text-sm px-4 py-2 rounded-lg transition-colors">
                    Details
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center space-x-2">
            <span className="text-red-600">⚠️</span>
            <span className="text-red-800 font-medium">Discovery Error</span>
          </div>
          <p className="text-red-700 mt-1">{error}</p>
          <button
            onClick={() => setError(null)}
            className="mt-2 text-sm text-red-600 hover:text-red-800 font-medium"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Empty State */}
      {discoveredSources.length === 0 && !discovering && !error && (
        <div className="card text-center py-12">
          <div className="text-6xl mb-4">🔍</div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Ready to Discover</h3>
          <p className="text-gray-600 mb-6">
            Click "Start Discovery" to automatically find data sources in your environment
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-2xl mx-auto">
            <div className="text-center">
              <div className="text-2xl mb-2">📊</div>
              <p className="text-sm text-gray-600">Databases & Warehouses</p>
            </div>
            <div className="text-center">
              <div className="text-2xl mb-2">📁</div>
              <p className="text-sm text-gray-600">Files & Documents</p>
            </div>
            <div className="text-center">
              <div className="text-2xl mb-2">☁️</div>
              <p className="text-sm text-gray-600">Cloud Services & APIs</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataDiscovery;