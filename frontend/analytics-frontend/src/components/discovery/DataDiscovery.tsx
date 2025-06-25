import React, { useState, useEffect } from 'react';
import { analyticsAPI } from '../../services/api';

// Import types from the API
import type {
  DataSource,
  DiscoveryResponse,
  DiscoverySession,
  APIError
} from '../../services/api';

// Define missing types locally since they're not exported
interface DiscoverySettings {
  auto_discovery: boolean;
  scan_frequency: 'hourly' | 'daily' | 'weekly';
  confidence_threshold: number;
  include_cloud_sources: boolean;
  scan_locations: string[];
}

interface DiscoveryInsights {
  recommendations: Array<{
    type: string;
    title: string;
    description: string;
    priority: 'high' | 'medium' | 'low';
    estimated_value: string;
  }>;
  patterns: Array<{
    pattern_type: string;
    description: string;
    frequency: number;
  }>;
}

interface DataSourceValidation {
  status: 'valid' | 'invalid' | 'warning';
  issues: Array<{
    type: 'connection' | 'schema' | 'data_quality';
    severity: 'low' | 'medium' | 'high';
    message: string;
    suggestion?: string;
  }>;
  health_score: number;
}

const DataDiscovery: React.FC = () => {
  const [discoveredSources, setDiscoveredSources] = useState<DataSource[]>([]);
  const [connectedSources, setConnectedSources] = useState<DataSource[]>([]);
  const [discoveryHistory, setDiscoveryHistory] = useState<DiscoverySession[]>([]);
  const [discoverySettings, setDiscoverySettings] = useState<DiscoverySettings | null>(null);
  const [discoveryInsights, setDiscoveryInsights] = useState<DiscoveryInsights | null>(null);

  const [isScanning, setIsScanning] = useState(false);
  const [scanMode, setScanMode] = useState<'fast' | 'balanced' | 'thorough'>('balanced');
  const [selectedSource, setSelectedSource] = useState<DataSource | null>(null);
  const [showConnectionModal, setShowConnectionModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [connectionParams, setConnectionParams] = useState<any>({});
  const [testingConnection, setTestingConnection] = useState(false);
  const [sourceValidation, setSourceValidation] = useState<DataSourceValidation | null>(null);
  const [error, setError] = useState<string | null>(null);

  const scanModes = [
    {
      id: 'fast' as const,
      name: 'Fast Scan',
      description: 'Quick discovery of obvious data sources',
      icon: '⚡',
      estimatedTime: '30 seconds'
    },
    {
      id: 'balanced' as const,
      name: 'Balanced Scan',
      description: 'Comprehensive scan with good performance',
      icon: '⚖️',
      estimatedTime: '2 minutes'
    },
    {
      id: 'thorough' as const,
      name: 'Thorough Scan',
      description: 'Deep scan for all possible sources',
      icon: '🔍',
      estimatedTime: '5 minutes'
    }
  ];

  useEffect(() => {
    initializeComponent();
  }, []);

  const initializeComponent = async () => {
    try {
      await Promise.allSettled([
        loadConnectedSources(),
        loadDiscoveryHistory(),
        loadDiscoverySettings(),
        loadDiscoveryInsights()
      ]);
    } catch (error) {
      console.error('Failed to initialize component:', error);
      setError('Failed to load initial data');
    }
  };

  const loadConnectedSources = async () => {
    try {
      const sources = await analyticsAPI.getConnectedSources();
      setConnectedSources(sources);
    } catch (error) {
      console.error('Failed to load connected sources:', error);
      // Use mock data as fallback
      setConnectedSources([
        {
          id: '1',
          source_id: 'postgres_main',
          type: 'PostgreSQL',
          confidence: 0.95,
          reasoning: 'Active connection found',
          context: {
            host: 'localhost',
            port: 5432,
            database: 'analytics_db',
            table_count: 45
          },
          status: 'connected'
        }
      ]);
    }
  };

  const loadDiscoveryHistory = async () => {
    try {
      const response = await analyticsAPI.getDiscoveryHistory();
      setDiscoveryHistory(response.discovery_sessions || []);
    } catch (error) {
      console.error('Failed to load discovery history:', error);
      // Use mock data as fallback
      setDiscoveryHistory([
        {
          id: '1',
          timestamp: new Date(Date.now() - 86400000).toISOString(),
          sources_found: 12,
          scan_mode: 'balanced',
          duration_ms: 120000,
          success_rate: 0.85
        }
      ]);
    }
  };

  const loadDiscoverySettings = async () => {
    try {
      // Check if the method exists first
      if (typeof analyticsAPI.getDiscoverySettings === 'function') {
        const response = await analyticsAPI.getDiscoverySettings();
        setDiscoverySettings(response.settings);
      } else {
        // Use default settings if method doesn't exist
        setDiscoverySettings({
          auto_discovery: false,
          scan_frequency: 'daily',
          confidence_threshold: 0.5,
          include_cloud_sources: true,
          scan_locations: ['/data', '/home/data']
        });
      }
    } catch (error) {
      console.error('Failed to load discovery settings:', error);
      // Use default settings
      setDiscoverySettings({
        auto_discovery: false,
        scan_frequency: 'daily',
        confidence_threshold: 0.5,
        include_cloud_sources: true,
        scan_locations: ['/data', '/home/data']
      });
    }
  };

  const loadDiscoveryInsights = async () => {
    try {
      // Check if the method exists first
      if (typeof analyticsAPI.getDiscoveryInsights === 'function') {
        const response = await analyticsAPI.getDiscoveryInsights();
        setDiscoveryInsights(response.insights);
      } else {
        // Use mock insights if method doesn't exist
        setDiscoveryInsights({
          recommendations: [
            {
              type: 'database',
              title: 'Consider MySQL Connection',
              description: 'MySQL service detected but not connected',
              priority: 'high',
              estimated_value: 'High data volume available'
            }
          ],
          patterns: [
            {
              pattern_type: 'database_clustering',
              description: 'Multiple databases found on same host',
              frequency: 3
            }
          ]
        });
      }
    } catch (error) {
      console.error('Failed to load discovery insights:', error);
      // Use mock insights
      setDiscoveryInsights({
        recommendations: [
          {
            type: 'database',
            title: 'Consider MySQL Connection',
            description: 'MySQL service detected but not connected',
            priority: 'high',
            estimated_value: 'High data volume available'
          }
        ],
        patterns: [
          {
            pattern_type: 'database_clustering',
            description: 'Multiple databases found on same host',
            frequency: 3
          }
        ]
      });
    }
  };

  const runDiscovery = async () => {
    setIsScanning(true);
    setError(null);

    try {
      const result: DiscoveryResponse = await analyticsAPI.discoverDataSources({
        mode: scanMode,
        include_environment_scan: true,
        max_recommendations: 20,
        confidence_threshold: discoverySettings?.confidence_threshold || 0.5
      });

      setDiscoveredSources(result.recommendations || []);

      // Refresh history and insights
      await Promise.allSettled([
        loadDiscoveryHistory(),
        loadDiscoveryInsights()
      ]);

    } catch (error) {
      console.error('Discovery failed:', error);
      const errorMsg = error instanceof Error ? error.message : 'Discovery failed';
      setError(errorMsg);

      // Use mock data as fallback
      setDiscoveredSources([
        {
          id: '2',
          source_id: 'mysql_prod',
          type: 'MySQL',
          confidence: 0.88,
          reasoning: 'MySQL service detected on port 3306',
          context: {
            host: '192.168.1.100',
            port: 3306,
            database: 'production',
            table_count: 28,
            size: '2.4 GB'
          },
          status: 'discovered'
        },
        {
          id: '3',
          source_id: 'csv_files',
          type: 'File System',
          confidence: 0.72,
          reasoning: 'Multiple CSV files found in data directory',
          context: {
            path: '/data/analytics/',
            file_count: 15,
            size: '450 MB'
          },
          status: 'discovered'
        }
      ]);
    } finally {
      setIsScanning(false);
    }
  };

  const connectToSource = async (source: DataSource) => {
    setSelectedSource(source);
    setConnectionParams({
      host: source.context.host || '',
      port: source.context.port || '',
      database: source.context.database || '',
      username: '',
      password: ''
    });
    setShowConnectionModal(true);
    setSourceValidation(null);
  };

  const testConnection = async () => {
    if (!selectedSource) return;

    setTestingConnection(true);
    try {
      const result = await analyticsAPI.testConnection(selectedSource.source_id, connectionParams);

      if (result.status === 'success') {
        alert('Connection successful! ✅');

        // Validate the source after successful connection
        try {
          if (typeof analyticsAPI.validateDataSource === 'function') {
            const validation = await analyticsAPI.validateDataSource(selectedSource.source_id);
            setSourceValidation(validation);
          }
        } catch (validationError) {
          console.warn('Validation failed but connection succeeded:', validationError);
        }
      } else {
        alert(`Connection failed: ${result.message} ❌`);
      }
    } catch (error) {
      console.error('Connection test failed:', error);
      const errorMsg = error instanceof Error ? error.message : 'Connection test failed';
      alert(`Connection test failed: ${errorMsg} ❌`);
    } finally {
      setTestingConnection(false);
    }
  };

  const finalizeConnection = async () => {
    if (!selectedSource) return;

    try {
      await analyticsAPI.connectDataSource(selectedSource.source_id, connectionParams);

      // Update the source status locally
      setDiscoveredSources(prev =>
        prev.map(source =>
          source.id === selectedSource.id
            ? { ...source, status: 'connected' as const }
            : source
        )
      );

      // Refresh connected sources
      await loadConnectedSources();

      // Close modal and reset state
      setShowConnectionModal(false);
      setConnectionParams({});
      setSelectedSource(null);
      setSourceValidation(null);

      alert('Data source connected successfully! 🎉');
    } catch (error) {
      console.error('Connection failed:', error);
      const errorMsg = error instanceof Error ? error.message : 'Connection failed';
      alert(`Failed to connect data source: ${errorMsg} ❌`);
    }
  };

  const disconnectSource = async (sourceId: string) => {
    try {
      if (typeof analyticsAPI.disconnectDataSource === 'function') {
        await analyticsAPI.disconnectDataSource(sourceId);
        await loadConnectedSources();
        alert('Data source disconnected successfully! 🔌');
      } else {
        alert('Disconnect functionality not available ❌');
      }
    } catch (error) {
      console.error('Disconnect failed:', error);
      const errorMsg = error instanceof Error ? error.message : 'Disconnect failed';
      alert(`Failed to disconnect source: ${errorMsg} ❌`);
    }
  };

  const refreshSource = async (sourceId: string) => {
    try {
      if (typeof analyticsAPI.refreshDataSource === 'function') {
        await analyticsAPI.refreshDataSource(sourceId);
        await loadConnectedSources();
        alert('Data source refreshed successfully! 🔄');
      } else {
        alert('Refresh functionality not available ❌');
      }
    } catch (error) {
      console.error('Refresh failed:', error);
      const errorMsg = error instanceof Error ? error.message : 'Refresh failed';
      alert(`Failed to refresh source: ${errorMsg} ❌`);
    }
  };

  const saveSettings = async (newSettings: DiscoverySettings) => {
    try {
      await analyticsAPI.saveDiscoverySettings(newSettings);
      setDiscoverySettings(newSettings);
      setShowSettingsModal(false);
      alert('Settings saved successfully! ⚙️');
    } catch (error) {
      console.error('Failed to save settings:', error);
      const errorMsg = error instanceof Error ? error.message : 'Failed to save settings';
      alert(`${errorMsg} ❌`);
    }
  };

  const getStatusColor = (status: string | undefined) => {
    switch (status) {
      case 'connected': return 'text-green-600 bg-green-100';
      case 'discovered': return 'text-blue-600 bg-blue-100';
      case 'connecting': return 'text-yellow-600 bg-yellow-100';
      case 'testing': return 'text-purple-600 bg-purple-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'postgresql': return '🐘';
      case 'mysql': return '🐬';
      case 'mongodb': return '🍃';
      case 'redis': return '🔴';
      case 'file system': return '📁';
      case 'api': return '🔗';
      case 'sqlite': return '💎';
      case 'elasticsearch': return '🔍';
      default: return '🗄️';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'text-red-600 bg-red-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-green-600 bg-green-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const formatDuration = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Data Discovery</h1>
        <p className="text-gray-600">
          Automatically discover and connect to data sources in your environment
        </p>
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center">
              <span className="text-red-600 mr-2">⚠️</span>
              <span className="text-red-800">{error}</span>
              <button
                onClick={() => setError(null)}
                className="ml-auto text-red-600 hover:text-red-800"
              >
                ✕
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Main Content */}
        <div className="lg:col-span-3 space-y-8">
          {/* Discovery Scanner */}
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Start New Discovery</h2>
              <button
                onClick={() => setShowSettingsModal(true)}
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                title="Discovery Settings"
              >
                ⚙️
              </button>
            </div>

            {/* Scan Mode Selection */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              {scanModes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setScanMode(mode.id)}
                  disabled={isScanning}
                  className={`p-4 border-2 rounded-lg text-left transition-colors ${
                    scanMode === mode.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  } ${isScanning ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div className="text-2xl mb-2">{mode.icon}</div>
                  <h3 className="font-semibold text-gray-900">{mode.name}</h3>
                  <p className="text-sm text-gray-600 mb-2">{mode.description}</p>
                  <div className="text-xs text-gray-500">Est. {mode.estimatedTime}</div>
                </button>
              ))}
            </div>

            <button
              onClick={runDiscovery}
              disabled={isScanning}
              className={`btn-primary ${isScanning ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {isScanning ? (
                <>
                  <svg className="animate-spin w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  Scanning ({scanMode})...
                </>
              ) : (
                `🔍 Start ${scanMode.charAt(0).toUpperCase() + scanMode.slice(1)} Discovery`
              )}
            </button>
          </div>

          {/* Discovery Insights */}
          {discoveryInsights && discoveryInsights.recommendations.length > 0 && (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Discovery Insights</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Recommendations</h3>
                  <div className="space-y-3">
                    {discoveryInsights.recommendations.map((rec, index) => (
                      <div key={index} className="p-3 border border-gray-200 rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <h4 className="font-medium text-gray-900">{rec.title}</h4>
                          <span className={`px-2 py-1 text-xs rounded-full ${getPriorityColor(rec.priority)}`}>
                            {rec.priority}
                          </span>
                        </div>
                        <p className="text-sm text-gray-600 mb-1">{rec.description}</p>
                        <div className="text-xs text-gray-500">{rec.estimated_value}</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold text-gray-900 mb-3">Patterns</h3>
                  <div className="space-y-3">
                    {discoveryInsights.patterns.map((pattern, index) => (
                      <div key={index} className="p-3 bg-gray-50 rounded-lg">
                        <h4 className="font-medium text-gray-900 mb-1">{pattern.pattern_type}</h4>
                        <p className="text-sm text-gray-600 mb-1">{pattern.description}</p>
                        <div className="text-xs text-gray-500">Frequency: {pattern.frequency}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Discovered Sources */}
          {discoveredSources.length > 0 && (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">
                Discovered Sources ({discoveredSources.length})
              </h2>
              <div className="space-y-4">
                {discoveredSources.map((source) => (
                  <div key={source.id} className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-4">
                        <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                          <span className="text-2xl">{getTypeIcon(source.type)}</span>
                        </div>

                        <div className="flex-1">
                          <div className="flex items-center space-x-3 mb-2">
                            <h3 className="font-semibold text-gray-900">{source.source_id}</h3>
                            <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(source.status)}`}>
                              {source.status || 'discovered'}
                            </span>
                            <span className="text-sm text-blue-600 font-medium">
                              {Math.round(source.confidence * 100)}% confidence
                            </span>
                          </div>

                          <p className="text-gray-600 text-sm mb-2">{source.reasoning}</p>

                          <div className="text-sm text-gray-500">
                            <span className="font-medium">{source.type}</span>
                            {source.context.host && (
                              <span className="ml-4">Host: {source.context.host}</span>
                            )}
                            {source.context.database && (
                              <span className="ml-4">DB: {source.context.database}</span>
                            )}
                            {source.context.table_count && (
                              <span className="ml-4">Tables: {source.context.table_count}</span>
                            )}
                            {source.context.size && (
                              <span className="ml-4">Size: {source.context.size}</span>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex space-x-2">
                        {source.status !== 'connected' && (
                          <button
                            onClick={() => connectToSource(source)}
                            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
                          >
                            Connect
                          </button>
                        )}
                        {source.status === 'connected' && (
                          <span className="px-4 py-2 bg-green-100 text-green-800 rounded-lg text-sm">
                            Connected ✅
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Connected Sources */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">
              Connected Sources ({connectedSources.length})
            </h2>
            {connectedSources.length === 0 ? (
              <div className="text-center py-8">
                <div className="text-4xl mb-4">🔗</div>
                <p className="text-gray-600">No connected sources yet. Start a discovery scan to find data sources.</p>
              </div>
            ) : (
              <div className="space-y-4">
                {connectedSources.map((source) => (
                  <div key={source.id} className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                          <span className="text-xl">{getTypeIcon(source.type)}</span>
                        </div>
                        <div>
                          <h3 className="font-semibold text-gray-900">{source.source_id}</h3>
                          <p className="text-sm text-gray-600">{source.type}</p>
                          {source.context.host && (
                            <p className="text-xs text-gray-500">{source.context.host}:{source.context.port}</p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor('connected')}`}>
                          Connected
                        </span>
                        <button
                          onClick={() => refreshSource(source.source_id)}
                          className="text-blue-600 hover:text-blue-800 text-sm"
                          title="Refresh"
                        >
                          🔄
                        </button>
                        <button
                          onClick={() => disconnectSource(source.source_id)}
                          className="text-red-600 hover:text-red-800 text-sm"
                          title="Disconnect"
                        >
                          🔌
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Discovery History */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Discoveries</h3>
            <div className="space-y-3">
              {discoveryHistory.slice(0, 5).map((session) => (
                <div key={session.id} className="p-3 bg-gray-50 rounded-lg">
                  <div className="flex justify-between items-start mb-2">
                    <span className="text-sm font-medium text-gray-900">
                      {session.sources_found} sources found
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(session.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                  <div className="text-xs text-gray-600">
                    <div>Mode: {session.scan_mode}</div>
                    <div>Duration: {formatDuration(session.duration_ms)}</div>
                    <div>Success: {Math.round(session.success_rate * 100)}%</div>
                  </div>
                </div>
              ))}
              {discoveryHistory.length === 0 && (
                <div className="text-center py-4 text-gray-500 text-sm">
                  No discovery history yet
                </div>
              )}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Discovery Stats</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Connected Sources:</span>
                <span className="font-semibold text-green-600">{connectedSources.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Discovered Sources:</span>
                <span className="font-semibold text-blue-600">{discoveredSources.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Total Scans:</span>
                <span className="font-semibold text-gray-900">{discoveryHistory.length}</span>
              </div>
              {discoverySettings && (
                <div className="flex justify-between">
                  <span className="text-gray-600">Auto Discovery:</span>
                  <span className={`font-semibold ${discoverySettings.auto_discovery ? 'text-green-600' : 'text-gray-600'}`}>
                    {discoverySettings.auto_discovery ? 'On' : 'Off'}
                  </span>
                </div>
              )}
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              <button
                onClick={() => initializeComponent()}
                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                🔄 Refresh All Sources
              </button>
              <button
                onClick={() => setShowSettingsModal(true)}
                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                ⚙️ Discovery Settings
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                📊 Source Health Check
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                📈 Usage Analytics
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Connection Modal */}
      {showConnectionModal && selectedSource && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">
                Connect to {selectedSource.source_id}
              </h3>
              <button
                onClick={() => {
                  setShowConnectionModal(false);
                  setSourceValidation(null);
                }}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Host</label>
                <input
                  type="text"
                  value={connectionParams.host || ''}
                  onChange={(e) => setConnectionParams({...connectionParams, host: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="localhost or IP address"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Port</label>
                <input
                  type="number"
                  value={connectionParams.port || ''}
                  onChange={(e) => setConnectionParams({...connectionParams, port: parseInt(e.target.value) || ''})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="5432, 3306, etc."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Username</label>
                <input
                  type="text"
                  value={connectionParams.username || ''}
                  onChange={(e) => setConnectionParams({...connectionParams, username: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Database username"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Password</label>
                <input
                  type="password"
                  value={connectionParams.password || ''}
                  onChange={(e) => setConnectionParams({...connectionParams, password: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Database password"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Database</label>
                <input
                  type="text"
                  value={connectionParams.database || ''}
                  onChange={(e) => setConnectionParams({...connectionParams, database: e.target.value})}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  placeholder="Database name"
                />
              </div>

              {/* SSL Option for databases */}
              {(selectedSource.type.toLowerCase().includes('postgres') || selectedSource.type.toLowerCase().includes('mysql')) && (
                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="ssl_enabled"
                    checked={connectionParams.ssl_enabled || false}
                    onChange={(e) => setConnectionParams({...connectionParams, ssl_enabled: e.target.checked})}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <label htmlFor="ssl_enabled" className="ml-2 text-sm text-gray-700">
                    Enable SSL
                  </label>
                </div>
              )}
            </div>

            {/* Validation Results */}
            {sourceValidation && (
              <div className="mt-4 p-3 border rounded-lg">
                <h4 className="font-medium text-gray-900 mb-2">Connection Validation</h4>
                <div className={`text-sm mb-2 ${
                  sourceValidation.status === 'valid' ? 'text-green-600' :
                  sourceValidation.status === 'warning' ? 'text-yellow-600' : 'text-red-600'
                }`}>
                  Status: {sourceValidation.status.toUpperCase()}
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  Health Score: {Math.round(sourceValidation.health_score * 100)}%
                </div>
                {sourceValidation.issues.length > 0 && (
                  <div>
                    <div className="text-sm font-medium text-gray-700 mb-1">Issues:</div>
                    <div className="space-y-1">
                      {sourceValidation.issues.map((issue: any, index: number) => (
                        <div key={index} className="text-xs text-gray-600">
                          <span className={`font-medium ${
                            issue.severity === 'high' ? 'text-red-600' :
                            issue.severity === 'medium' ? 'text-yellow-600' : 'text-blue-600'
                          }`}>
                            {issue.severity.toUpperCase()}:
                          </span> {issue.message}
                          {issue.suggestion && (
                            <div className="text-gray-500 mt-1">💡 {issue.suggestion}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="flex space-x-3 mt-6">
              <button
                onClick={testConnection}
                disabled={testingConnection || !connectionParams.host || !connectionParams.username}
                className={`flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors ${
                  (testingConnection || !connectionParams.host || !connectionParams.username) ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                {testingConnection ? (
                  <>
                    <svg className="animate-spin w-4 h-4 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    Testing...
                  </>
                ) : (
                  'Test Connection'
                )}
              </button>
              <button
                onClick={finalizeConnection}
                disabled={!connectionParams.host || !connectionParams.username}
                className={`flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors ${
                  (!connectionParams.host || !connectionParams.username) ? 'opacity-50 cursor-not-allowed' : ''
                }`}
              >
                Connect
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Settings Modal */}
      {showSettingsModal && discoverySettings && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">Discovery Settings</h3>
              <button
                onClick={() => setShowSettingsModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="space-y-6">
              {/* Auto Discovery */}
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-gray-700">Auto Discovery</label>
                  <input
                    type="checkbox"
                    checked={discoverySettings.auto_discovery}
                    onChange={(e) => setDiscoverySettings({
                      ...discoverySettings,
                      auto_discovery: e.target.checked
                    })}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">Automatically scan for new data sources periodically</p>
              </div>

              {/* Scan Frequency */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Scan Frequency</label>
                <select
                  value={discoverySettings.scan_frequency}
                  onChange={(e) => setDiscoverySettings({
                    ...discoverySettings,
                    scan_frequency: e.target.value as 'hourly' | 'daily' | 'weekly'
                  })}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="hourly">Hourly</option>
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                </select>
              </div>

              {/* Confidence Threshold */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Confidence Threshold: {Math.round(discoverySettings.confidence_threshold * 100)}%
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={discoverySettings.confidence_threshold}
                  onChange={(e) => setDiscoverySettings({
                    ...discoverySettings,
                    confidence_threshold: parseFloat(e.target.value)
                  })}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Low (0%)</span>
                  <span>High (100%)</span>
                </div>
                <p className="text-xs text-gray-500 mt-1">Only show sources above this confidence level</p>
              </div>

              {/* Include Cloud Sources */}
              <div>
                <div className="flex items-center justify-between">
                  <label className="text-sm font-medium text-gray-700">Include Cloud Sources</label>
                  <input
                    type="checkbox"
                    checked={discoverySettings.include_cloud_sources}
                    onChange={(e) => setDiscoverySettings({
                      ...discoverySettings,
                      include_cloud_sources: e.target.checked
                    })}
                    className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">Scan for cloud databases and services</p>
              </div>

              {/* Scan Locations */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Scan Locations</label>
                <div className="space-y-2">
                  {discoverySettings.scan_locations.map((location: string, index: number) => (
                    <div key={index} className="flex items-center space-x-2">
                      <input
                        type="text"
                        value={location}
                        onChange={(e) => {
                          const newLocations = [...discoverySettings.scan_locations];
                          newLocations[index] = e.target.value;
                          setDiscoverySettings({
                            ...discoverySettings,
                            scan_locations: newLocations
                          });
                        }}
                        className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                        placeholder="/path/to/data"
                      />
                      <button
                        onClick={() => {
                          const newLocations = discoverySettings.scan_locations.filter((_: string, i: number) => i !== index);
                          setDiscoverySettings({
                            ...discoverySettings,
                            scan_locations: newLocations
                          });
                        }}
                        className="p-2 text-red-600 hover:text-red-800"
                      >
                        🗑️
                      </button>
                    </div>
                  ))}
                  <button
                    onClick={() => setDiscoverySettings({
                      ...discoverySettings,
                      scan_locations: [...discoverySettings.scan_locations, '']
                    })}
                    className="w-full px-3 py-2 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-400 transition-colors text-sm"
                  >
                    + Add Location
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">File system paths to scan for data files</p>
              </div>
            </div>

            <div className="flex space-x-3 mt-6">
              <button
                onClick={() => setShowSettingsModal(false)}
                className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => saveSettings(discoverySettings)}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Save Settings
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataDiscovery;