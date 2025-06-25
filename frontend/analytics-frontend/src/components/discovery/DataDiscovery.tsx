import React, { useState, useEffect } from 'react';
import { Search, Database, Check, X, Play, Settings, RefreshCw, Plus, AlertCircle, Zap, Brain, Eye } from 'lucide-react';
import { analyticsAPI, extractErrorMessage, DataSource, DiscoveryResponse } from '../../services/api';

interface DiscoverySession {
  id: string;
  timestamp: string;
  sources_found: number;
  scan_mode: string;
  duration_ms: number;
  success_rate: number;
}

const DataDiscovery: React.FC = () => {
  const [discoveredSources, setDiscoveredSources] = useState<DataSource[]>([]);
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [selectedSources, setSelectedSources] = useState<Set<string>>(new Set());
  const [testingConnections, setTestingConnections] = useState<Set<string>>(new Set());
  const [connectedSources, setConnectedSources] = useState<Set<string>>(new Set());
  const [failedSources, setFailedSources] = useState<Set<string>>(new Set());
  const [scanMode, setScanMode] = useState<'fast' | 'balanced' | 'thorough'>('balanced');
  const [includeEnvironmentScan, setIncludeEnvironmentScan] = useState(true);
  const [maxRecommendations, setMaxRecommendations] = useState(10);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.6);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [systemCapabilities, setSystemCapabilities] = useState<any>(null);
  const [discoveryHistory, setDiscoveryHistory] = useState<DiscoverySession[]>([]);
  const [lastScanMetadata, setLastScanMetadata] = useState<any>(null);
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);

  useEffect(() => {
    initializeDiscovery();
    loadDiscoveryHistory();
  }, []);

  const initializeDiscovery = async () => {
    try {
      setBackendStatus('checking');

      // Check backend status and capabilities
      const [healthCheck, capabilities] = await Promise.allSettled([
        analyticsAPI.healthCheck(),
        analyticsAPI.getCapabilities()
      ]);

      if (healthCheck.status === 'fulfilled' && healthCheck.value.status === 'healthy') {
        setBackendStatus('connected');
      } else {
        setBackendStatus('disconnected');
      }

      if (capabilities.status === 'fulfilled') {
        setSystemCapabilities(capabilities.value);
      }

      // Auto-discover sources on load
      await performDiscovery();

    } catch (error) {
      console.error('Failed to initialize discovery:', error);
      setBackendStatus('disconnected');
      setError('Failed to initialize data discovery. Please check backend connection.');
    }
  };

  const loadDiscoveryHistory = () => {
    try {
      const history = JSON.parse(localStorage.getItem('discovery_history') || '[]');
      setDiscoveryHistory(history.slice(0, 5)); // Keep last 5 sessions
    } catch (error) {
      console.error('Failed to load discovery history:', error);
    }
  };

  const saveDiscoverySession = (session: DiscoverySession) => {
    try {
      const history = JSON.parse(localStorage.getItem('discovery_history') || '[]');
      history.unshift(session);
      localStorage.setItem('discovery_history', JSON.stringify(history.slice(0, 10)));
      setDiscoveryHistory(history.slice(0, 5));
    } catch (error) {
      console.error('Failed to save discovery session:', error);
    }
  };

  const performDiscovery = async () => {
    if (backendStatus !== 'connected') {
      setError('Backend is not connected. Please check the server status.');
      return;
    }

    setIsScanning(true);
    setScanProgress(0);
    setError(null);

    const startTime = Date.now();

    try {
      console.log('🔍 Starting data source discovery...');

      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setScanProgress(prev => Math.min(prev + Math.random() * 20, 95));
      }, 500);

      const response: DiscoveryResponse = await analyticsAPI.discoverDataSources({
        mode: scanMode,
        include_environment_scan: includeEnvironmentScan,
        max_recommendations: maxRecommendations,
        confidence_threshold: confidenceThreshold
      });

      clearInterval(progressInterval);
      setScanProgress(100);

      if (response.status === 'success') {
        setDiscoveredSources(response.recommendations);
        setLastScanMetadata(response.metadata);

        console.log(`✅ Discovery completed: ${response.recommendations.length} sources found`);

        // Save discovery session
        const session: DiscoverySession = {
          id: Date.now().toString(),
          timestamp: new Date().toISOString(),
          sources_found: response.recommendations.length,
          scan_mode: scanMode,
          duration_ms: Date.now() - startTime,
          success_rate: response.recommendations.length > 0 ? 100 : 0
        };
        saveDiscoverySession(session);

        // Auto-select high-confidence sources
        const highConfidenceSources = response.recommendations.filter(source => source.confidence >= 0.8);
        const highConfidenceIds = new Set<string>();
        highConfidenceSources.forEach(source => highConfidenceIds.add(source.id));
        setSelectedSources(highConfidenceIds);

      } else {
        setError(`Discovery failed: ${response.message || 'Unknown error'}`);
      }

    } catch (error) {
      console.error('Discovery failed:', error);
      setError(extractErrorMessage(error));
    } finally {
      setIsScanning(false);
      setScanProgress(0);
    }
  };

  const testConnection = async (source: DataSource) => {
    setTestingConnections(prev => {
      const newSet = new Set(prev);
      newSet.add(source.id);
      return newSet;
    });
    setError(null);

    try {
      console.log(`🔌 Testing connection to ${source.source_id}...`);

      // Use the analyticsAPI to test connection
      const result = await analyticsAPI.testConnection(source.id, source.context);

      if (result.status === 'success') {
        setConnectedSources(prev => {
          const newSet = new Set(prev);
          newSet.add(source.id);
          return newSet;
        });
        setFailedSources(prev => {
          const newSet = new Set(prev);
          newSet.delete(source.id);
          return newSet;
        });
        console.log(`✅ Connection successful: ${source.source_id}`);
      } else {
        setFailedSources(prev => {
          const newSet = new Set(prev);
          newSet.add(source.id);
          return newSet;
        });
        setConnectedSources(prev => {
          const newSet = new Set(prev);
          newSet.delete(source.id);
          return newSet;
        });
        console.log(`❌ Connection failed: ${source.source_id}`);
      }

    } catch (error) {
      console.error(`Connection test failed for ${source.source_id}:`, error);
      setFailedSources(prev => {
        const newSet = new Set(prev);
        newSet.add(source.id);
        return newSet;
      });
      setError(`Connection test failed for ${source.source_id}: ${extractErrorMessage(error)}`);
    } finally {
      setTestingConnections(prev => {
        const newSet = new Set(prev);
        newSet.delete(source.id);
        return newSet;
      });
    }
  };

  const connectToSource = async (source: DataSource) => {
    try {
      console.log(`🔗 Connecting to ${source.source_id}...`);

      const result = await analyticsAPI.connectDataSource(source.id, source.context);

      if (result.status === 'success') {
        setConnectedSources(prev => {
          const newSet = new Set(prev);
          newSet.add(source.id);
          return newSet;
        });
        console.log(`✅ Connected to ${source.source_id}`);

        // Record feedback for successful connection
        await analyticsAPI.recordFeedback(
          `connect_${source.id}`,
          'connected',
          { source_id: source.id, timestamp: new Date().toISOString() }
        );
      } else {
        setError(`Failed to connect to ${source.source_id}: ${result.error || 'Unknown error'}`);
      }

    } catch (error) {
      console.error(`Connection failed for ${source.source_id}:`, error);
      setError(`Connection failed: ${extractErrorMessage(error)}`);
    }
  };

  const connectSelectedSources = async () => {
    const selectedSourcesList = discoveredSources.filter(source => selectedSources.has(source.id));

    for (const source of selectedSourcesList) {
      await connectToSource(source);
    }
  };

  const getSourceIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'database':
      case 'postgresql':
        return '🐘';
      case 'mysql':
        return '🐬';
      case 'mongodb':
        return '🍃';
      case 'redis':
        return '🔴';
      case 'file':
      case 'csv':
        return '📁';
      case 'excel':
        return '📊';
      case 'api':
      case 'rest_api':
        return '🔗';
      case 'bi_tool':
      case 'tableau':
        return '📈';
      case 'power_bi':
        return '📊';
      default:
        return '🗄️';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600 bg-green-100';
    if (confidence >= 0.6) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const getStatusIndicator = () => {
    switch (backendStatus) {
      case 'checking':
        return { color: 'bg-yellow-500', text: 'Checking...', pulse: true };
      case 'connected':
        return { color: 'bg-green-500', text: 'Discovery Ready', pulse: false };
      case 'disconnected':
        return { color: 'bg-red-500', text: 'Backend Offline', pulse: false };
    }
  };

  const statusInfo = getStatusIndicator();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900 flex items-center space-x-2">
                <Search className="w-6 h-6 text-blue-500" />
                <span>Smart Data Discovery</span>
              </h1>
              <p className="text-gray-600 text-sm mt-1">
                Automatically discover and connect to data sources in your environment
              </p>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 text-sm">
                <div className={`w-2 h-2 rounded-full ${statusInfo.color} ${statusInfo.pulse ? 'animate-pulse' : ''}`} />
                <span className="text-gray-600">{statusInfo.text}</span>
              </div>

              {systemCapabilities && (
                <div className="text-xs text-gray-500 flex items-center space-x-2">
                  {systemCapabilities.smart_features?.auto_data_discovery && (
                    <span className="text-green-600">🔍 Auto-Discovery</span>
                  )}
                  {systemCapabilities.smart_features?.intelligent_recommendations && (
                    <span className="text-blue-600">🧠 Smart Recommendations</span>
                  )}
                  {systemCapabilities.smart_features?.environment_scanning && (
                    <span className="text-purple-600">🌍 Environment Scan</span>
                  )}
                </div>
              )}

              <button
                onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
                className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
                title="Advanced Settings"
              >
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-6">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
              <div className="flex-1">
                <h3 className="text-sm font-medium text-red-800">Discovery Error</h3>
                <p className="mt-1 text-sm text-red-700">{error}</p>
              </div>
              <button
                onClick={() => setError(null)}
                className="flex-shrink-0 text-red-400 hover:text-red-600"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* Backend Status Warning */}
        {backendStatus === 'disconnected' && (
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <AlertCircle className="w-5 h-5 text-yellow-600" />
                <div>
                  <h3 className="font-semibold text-yellow-900">Backend Connection Issue</h3>
                  <p className="text-yellow-700 text-sm mt-1">
                    Smart Defaults Engine is offline. Discovery features are limited.
                  </p>
                </div>
              </div>
              <button
                onClick={initializeDiscovery}
                className="text-sm bg-yellow-600 text-white px-3 py-1 rounded hover:bg-yellow-700 transition-colors"
              >
                Retry Connection
              </button>
            </div>
          </div>
        )}

        {/* Discovery Controls */}
        <div className="bg-white rounded-xl shadow-lg border border-white/20 p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <Brain className="w-5 h-5 text-blue-500 mr-2" />
                Smart Discovery Controls
              </h2>
              <p className="text-gray-600 text-sm mt-1">
                Configure and run intelligent data source discovery
              </p>
            </div>

            <div className="flex items-center space-x-3">
              {discoveredSources.length > 0 && (
                <div className="text-sm text-gray-600">
                  Found: <span className="font-semibold">{discoveredSources.length}</span> sources
                </div>
              )}

              <button
                onClick={performDiscovery}
                disabled={isScanning || backendStatus !== 'connected'}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white px-6 py-2 rounded-lg transition-colors inline-flex items-center space-x-2 font-medium"
              >
                {isScanning ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                    <span>Discovering...</span>
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    <span>Start Discovery</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Quick Settings */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Scan Mode</label>
              <select
                value={scanMode}
                onChange={(e) => setScanMode(e.target.value as 'fast' | 'balanced' | 'thorough')}
                disabled={isScanning}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="fast">Fast (30s)</option>
                <option value="balanced">Balanced (60s)</option>
                <option value="thorough">Thorough (2-3min)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Max Sources</label>
              <input
                type="number"
                value={maxRecommendations}
                onChange={(e) => setMaxRecommendations(parseInt(e.target.value))}
                disabled={isScanning}
                min="1"
                max="50"
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Confidence</label>
              <input
                type="range"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                disabled={isScanning}
                min="0.1"
                max="1.0"
                step="0.1"
                className="w-full mt-3"
              />
              <div className="text-xs text-gray-500 mt-1">
                Min: {Math.round(confidenceThreshold * 100)}%
              </div>
            </div>
          </div>

          {/* Advanced Settings */}
          {showAdvancedSettings && (
            <div className="border-t border-gray-200 pt-4 mt-4">
              <h3 className="text-sm font-medium text-gray-700 mb-3">Advanced Options</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={includeEnvironmentScan}
                    onChange={(e) => setIncludeEnvironmentScan(e.target.checked)}
                    disabled={isScanning}
                    className="rounded"
                  />
                  <span className="text-sm text-gray-700">Include environment scanning</span>
                </label>
              </div>
            </div>
          )}

          {/* Progress Bar */}
          {isScanning && (
            <div className="mt-4">
              <div className="flex items-center justify-between text-sm text-gray-600 mb-2">
                <span>Scanning for data sources...</span>
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

        {/* Discovered Sources */}
        {discoveredSources.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg border border-white/20 p-6 mb-6">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <Eye className="w-5 h-5 text-green-500 mr-2" />
                Discovered Sources ({discoveredSources.length})
              </h2>

              {selectedSources.size > 0 && (
                <div className="flex items-center space-x-3">
                  <span className="text-sm text-gray-600">
                    {selectedSources.size} selected
                  </span>
                  <button
                    onClick={connectSelectedSources}
                    disabled={selectedSources.size === 0}
                    className="bg-green-600 hover:bg-green-700 disabled:bg-gray-300 text-white px-4 py-2 rounded-lg transition-colors inline-flex items-center space-x-2"
                  >
                    <Plus className="w-4 h-4" />
                    <span>Connect Selected</span>
                  </button>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {discoveredSources.map((source) => (
                <div
                  key={source.id}
                  className={`border-2 rounded-xl p-4 transition-all hover:shadow-md ${
                    selectedSources.has(source.id)
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center space-x-3">
                      <div className="text-2xl">{getSourceIcon(source.type)}</div>
                      <div>
                        <h3 className="font-semibold text-gray-900">{source.source_id}</h3>
                        <p className="text-sm text-gray-600">{source.type}</p>
                      </div>
                    </div>

                    <input
                      type="checkbox"
                      checked={selectedSources.has(source.id)}
                      onChange={(e) => {
                        const newSet = new Set(selectedSources);
                        if (e.target.checked) {
                          newSet.add(source.id);
                        } else {
                          newSet.delete(source.id);
                        }
                        setSelectedSources(newSet);
                      }}
                      className="rounded"
                    />
                  </div>

                  <p className="text-sm text-gray-700 mb-3">{source.reasoning}</p>

                  <div className="flex items-center justify-between mb-3">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(source.confidence)}`}>
                      {Math.round(source.confidence * 100)}% confidence
                    </span>

                    {connectedSources.has(source.id) && (
                      <span className="text-green-600 text-sm flex items-center">
                        <Check className="w-4 h-4 mr-1" />
                        Connected
                      </span>
                    )}

                    {failedSources.has(source.id) && (
                      <span className="text-red-600 text-sm flex items-center">
                        <X className="w-4 h-4 mr-1" />
                        Failed
                      </span>
                    )}
                  </div>

                  <div className="text-xs text-gray-500 mb-3">
                    {Object.entries(source.context).map(([key, value]) => (
                      <div key={key}>
                        <strong>{key}:</strong> {String(value)}
                      </div>
                    ))}
                  </div>

                  <div className="flex space-x-2">
                    <button
                      onClick={() => testConnection(source)}
                      disabled={testingConnections.has(source.id)}
                      className="flex-1 bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg transition-colors inline-flex items-center justify-center space-x-1 text-sm"
                    >
                      {testingConnections.has(source.id) ? (
                        <div className="animate-spin rounded-full h-3 w-3 border-2 border-gray-600 border-t-transparent" />
                      ) : (
                        <Play className="w-3 h-3" />
                      )}
                      <span>Test</span>
                    </button>

                    <button
                      onClick={() => connectToSource(source)}
                      disabled={connectedSources.has(source.id)}
                      className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-green-600 text-white px-3 py-2 rounded-lg transition-colors inline-flex items-center justify-center space-x-1 text-sm"
                    >
                      <Plus className="w-3 h-3" />
                      <span>{connectedSources.has(source.id) ? 'Connected' : 'Connect'}</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Scan Metadata */}
        {lastScanMetadata && (
          <div className="bg-white rounded-xl shadow-lg border border-white/20 p-6 mb-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <Zap className="w-5 h-5 text-yellow-500 mr-2" />
              Last Scan Results
            </h2>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{lastScanMetadata.total_candidates || 0}</div>
                <div className="text-sm text-gray-600">Total Candidates</div>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{discoveredSources.length}</div>
                <div className="text-sm text-gray-600">Sources Found</div>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">
                  {lastScanMetadata.scan_duration_ms ? `${Math.round(lastScanMetadata.scan_duration_ms / 1000)}s` : 'N/A'}
                </div>
                <div className="text-sm text-gray-600">Scan Duration</div>
              </div>

              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">
                  {lastScanMetadata.ml_enhanced ? 'Yes' : 'No'}
                </div>
                <div className="text-sm text-gray-600">ML Enhanced</div>
              </div>
            </div>
          </div>
        )}

        {/* Discovery History */}
        {discoveryHistory.length > 0 && (
          <div className="bg-white rounded-xl shadow-lg border border-white/20 p-6">
            <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
              <RefreshCw className="w-5 h-5 text-gray-500 mr-2" />
              Recent Discovery Sessions
            </h2>

            <div className="space-y-3">
              {discoveryHistory.map((session) => (
                <div key={session.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="text-sm">
                      <span className="font-medium">{session.sources_found} sources</span>
                      <span className="text-gray-500 ml-2">•</span>
                      <span className="text-gray-500 ml-2">{session.scan_mode} mode</span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {new Date(session.timestamp).toLocaleString()}
                    </div>
                  </div>

                  <div className="flex items-center space-x-3">
                    <span className="text-sm text-gray-600">
                      {Math.round(session.duration_ms / 1000)}s
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      session.success_rate >= 80 ? 'text-green-600 bg-green-100' : 'text-yellow-600 bg-yellow-100'
                    }`}>
                      {session.success_rate}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Empty State */}
        {!isScanning && discoveredSources.length === 0 && backendStatus === 'connected' && (
          <div className="bg-white rounded-xl shadow-lg border border-white/20 p-12 text-center">
            <Database className="w-16 h-16 text-gray-400 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-900 mb-2">No Data Sources Found</h3>
            <p className="text-gray-600 mb-6">
              Click "Start Discovery" to automatically scan for available data sources in your environment.
            </p>
            <button
              onClick={performDiscovery}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors inline-flex items-center space-x-2 font-medium"
            >
              <Search className="w-5 h-5" />
              <span>Start Smart Discovery</span>
            </button>
          </div>
        )}

        {/* System Capabilities Info */}
        {systemCapabilities && (
          <div className="mt-6 bg-blue-50 border border-blue-200 rounded-xl p-6">
            <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
              <Brain className="w-5 h-5 mr-2" />
              🚀 Smart Discovery Features Available
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
              {systemCapabilities.smart_features?.auto_data_discovery && (
                <div className="flex items-center space-x-2 text-blue-700">
                  <span>✅</span>
                  <span>Automated Discovery</span>
                </div>
              )}
              {systemCapabilities.smart_features?.intelligent_recommendations && (
                <div className="flex items-center space-x-2 text-blue-700">
                  <span>✅</span>
                  <span>Smart Recommendations</span>
                </div>
              )}
              {systemCapabilities.smart_features?.environment_scanning && (
                <div className="flex items-center space-x-2 text-blue-700">
                  <span>✅</span>
                  <span>Environment Scanning</span>
                </div>
              )}
              {systemCapabilities.smart_features?.learning_from_usage && (
                <div className="flex items-center space-x-2 text-blue-700">
                  <span>✅</span>
                  <span>Learning from Usage</span>
                </div>
              )}
              {systemCapabilities.smart_features?.policy_compliance && (
                <div className="flex items-center space-x-2 text-blue-700">
                  <span>✅</span>
                  <span>Policy Compliance</span>
                </div>
              )}
              {systemCapabilities.smart_features?.personalized_suggestions && (
                <div className="flex items-center space-x-2 text-blue-700">
                  <span>✅</span>
                  <span>Personalized Suggestions</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataDiscovery;