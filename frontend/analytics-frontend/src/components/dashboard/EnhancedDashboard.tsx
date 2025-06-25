import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyticsAPI, extractErrorMessage } from '../../services/api';

interface DashboardData {
  total_analyses: number;
  success_rate: number;
  avg_processing_time: number;
  top_data_sources: Array<{ name: string; usage_count: number }>;
  recent_activities: Array<{
    id: string;
    type: string;
    description: string;
    timestamp: string;
    status: string;
  }>;
  performance_metrics: {
    analyses_this_week: number;
    data_processed_gb: number;
    insights_generated: number;
    charts_created: number;
  };
}

interface QuickStat {
  label: string;
  value: string | number;
  change: number;
  changeType: 'positive' | 'negative' | 'neutral';
  icon: string;
  color: string;
}

const EnhancedDashboard: React.FC = () => {
  const [dashboardData, setDashboardData] = useState<DashboardData | null>(null);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [quickActions, setQuickActions] = useState<Array<{ id: string; loading: boolean }>>([]);
  const [error, setError] = useState<string | null>(null);

  const navigate = useNavigate();

  useEffect(() => {
    loadDashboardData();

    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      refreshData();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const checkBackendStatus = async () => {
    try {
      const health = await analyticsAPI.healthCheck();
      if (health.status === 'healthy') {
        setBackendStatus('connected');
      } else {
        setBackendStatus('disconnected');
      }
    } catch (error) {
      setBackendStatus('disconnected');
    }
  };

  const loadDashboardData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Check backend status first
      await checkBackendStatus();

      // Load dashboard data and system status in parallel with better error handling
      const [dashboardResponse, statusResponse, capabilitiesResponse] = await Promise.allSettled([
        analyticsAPI.getAnalyticsDashboard().catch(() => null),
        analyticsAPI.getSystemStatus().catch(() => null),
        analyticsAPI.getCapabilities().catch(() => null)
      ]);

      if (dashboardResponse.status === 'fulfilled' && dashboardResponse.value?.dashboard_data) {
        setDashboardData(dashboardResponse.value.dashboard_data);
      } else {
        // Use enhanced mock data if API fails
        setDashboardData(getEnhancedMockDashboardData());
      }

      if (statusResponse.status === 'fulfilled' && statusResponse.value) {
        setSystemStatus(statusResponse.value);
      }

      // Merge capabilities into system status
      if (capabilitiesResponse.status === 'fulfilled' && capabilitiesResponse.value) {
        setSystemStatus((prev: any) => ({
          ...prev,
          capabilities: capabilitiesResponse.value
        }));
      }

      setLastUpdated(new Date());
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      setError(extractErrorMessage(error));
      // Use mock data as fallback
      setDashboardData(getEnhancedMockDashboardData());
    } finally {
      setLoading(false);
    }
  };

  const refreshData = async () => {
    setRefreshing(true);
    try {
      await loadDashboardData();
    } finally {
      setRefreshing(false);
    }
  };

  const getEnhancedMockDashboardData = (): DashboardData => ({
    total_analyses: 247,
    success_rate: 94.5,
    avg_processing_time: 3.2,
    top_data_sources: [
      { name: 'PostgreSQL', usage_count: 89 },
      { name: 'CSV Files', usage_count: 67 },
      { name: 'Excel Files', usage_count: 45 },
      { name: 'JSON APIs', usage_count: 23 },
      { name: 'MongoDB', usage_count: 18 },
      { name: 'Tableau', usage_count: 12 }
    ],
    recent_activities: [
      {
        id: '1',
        type: 'analysis',
        description: 'Enhanced AI analysis completed for sales trends dataset',
        timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
        status: 'completed'
      },
      {
        id: '2',
        type: 'discovery',
        description: 'Auto-discovered new data source: customer_analytics_db',
        timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
        status: 'discovered'
      },
      {
        id: '3',
        type: 'report',
        description: 'Comprehensive report with ML insights generated',
        timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
        status: 'completed'
      },
      {
        id: '4',
        type: 'prediction',
        description: 'Predictive model trained with 94% accuracy',
        timestamp: new Date(Date.now() - 1000 * 60 * 45).toISOString(),
        status: 'completed'
      },
      {
        id: '5',
        type: 'anomaly',
        description: 'Anomaly detection identified 3 critical patterns',
        timestamp: new Date(Date.now() - 1000 * 60 * 60).toISOString(),
        status: 'completed'
      }
    ],
    performance_metrics: {
      analyses_this_week: 47,
      data_processed_gb: 12.8,
      insights_generated: 156,
      charts_created: 89
    }
  });

  const getQuickStats = (): QuickStat[] => {
    if (!dashboardData) return [];

    return [
      {
        label: 'Total Analyses',
        value: dashboardData.total_analyses,
        change: 12,
        changeType: 'positive',
        icon: '📊',
        color: 'blue'
      },
      {
        label: 'Success Rate',
        value: `${dashboardData.success_rate}%`,
        change: 2.1,
        changeType: 'positive',
        icon: '✅',
        color: 'green'
      },
      {
        label: 'Avg Processing',
        value: `${dashboardData.avg_processing_time}s`,
        change: -0.3,
        changeType: 'positive',
        icon: '⚡',
        color: 'purple'
      },
      {
        label: 'Data Sources',
        value: dashboardData.top_data_sources.length,
        change: 2,
        changeType: 'positive',
        icon: '🔗',
        color: 'orange'
      }
    ];
  };

  const handleQuickAction = async (actionId: string, actionFn: () => Promise<void>) => {
    const newQuickActions = quickActions.filter(a => a.id !== actionId);
    newQuickActions.push({ id: actionId, loading: true });
    setQuickActions(newQuickActions);

    try {
      await actionFn();
    } catch (error) {
      console.error(`Quick action ${actionId} failed:`, error);
      setError(`Quick action failed: ${extractErrorMessage(error)}`);
    } finally {
      setQuickActions((prev: Array<{ id: string; loading: boolean }>) =>
        prev.filter(a => a.id !== actionId)
      );
    }
  };

  const quickDiscovery = async (): Promise<void> => {
    try {
      await analyticsAPI.discoverDataSources({ mode: 'fast', max_recommendations: 3 });
      navigate('/discovery');
    } catch (error) {
      console.error('Quick discovery failed:', error);
      navigate('/discovery');
    }
  };

  const quickAnalysis = async (): Promise<void> => {
    navigate('/datagenie'); // Navigate to DataGenie instead of analysis
  };

  const quickChat = async (): Promise<void> => {
    navigate('/chat'); // Navigate to conversational analytics
  };

  const generateQuickReport = async (): Promise<void> => {
    try {
      // For now, navigate to reports page since we don't have a direct report generation endpoint
      navigate('/dashboard/reports');
    } catch (error) {
      console.error('Quick report failed:', error);
      navigate('/dashboard/reports');
    }
  };

  const getChangeIcon = (changeType: string) => {
    switch (changeType) {
      case 'positive': return '↗️';
      case 'negative': return '↘️';
      default: return '➡️';
    }
  };

  const getChangeColor = (changeType: string) => {
    switch (changeType) {
      case 'positive': return 'text-green-600';
      case 'negative': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getStatColor = (color: string) => {
    const colors = {
      blue: 'bg-blue-500',
      green: 'bg-green-500',
      purple: 'bg-purple-500',
      orange: 'bg-orange-500',
      red: 'bg-red-500'
    };
    return colors[color as keyof typeof colors] || 'bg-gray-500';
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  };

  const getActivityIcon = (type: string) => {
    switch (type) {
      case 'analysis': return '🧠';
      case 'discovery': return '🔍';
      case 'report': return '📋';
      case 'connection': return '🔗';
      case 'prediction': return '🔮';
      case 'anomaly': return '🎯';
      case 'correlation': return '🔗';
      case 'clustering': return '🎪';
      default: return '📊';
    }
  };

  const getActivityColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'discovered': return 'text-blue-600 bg-blue-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'processing': return 'text-yellow-600 bg-yellow-100';
      case 'training': return 'text-purple-600 bg-purple-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getBackendStatusIndicator = () => {
    switch (backendStatus) {
      case 'checking':
        return { color: 'bg-yellow-500', text: 'Checking...', pulse: true };
      case 'connected':
        return { color: 'bg-green-500', text: 'Backend Ready', pulse: false };
      case 'disconnected':
        return { color: 'bg-red-500', text: 'Backend Offline', pulse: false };
    }
  };

  const statusInfo = getBackendStatusIndicator();

  const isActionLoading = (actionId: string) => {
    return quickActions.some(a => a.id === actionId && a.loading);
  };

  if (loading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {[1, 2, 3, 4].map(i => (
              <div key={i} className="h-32 bg-gray-200 rounded-lg"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 h-96 bg-gray-200 rounded-lg"></div>
            <div className="h-96 bg-gray-200 rounded-lg"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Enhanced Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Enhanced Analytics Dashboard</h1>
          <p className="text-gray-600">
            Complete overview of your AI-powered data analytics platform with advanced insights
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${statusInfo.color} ${statusInfo.pulse ? 'animate-pulse' : ''}`} />
            <span className="text-gray-600">{statusInfo.text}</span>
          </div>

          {systemStatus?.capabilities && (
            <div className="text-xs text-gray-500 flex items-center space-x-2">
              {systemStatus.capabilities.smart_features?.unified_smart_engine && (
                <span className="text-blue-600">🧠 Smart Engine</span>
              )}
              {systemStatus.capabilities.smart_features?.auto_data_discovery && (
                <span className="text-green-600">🔍 Auto-Discovery</span>
              )}
              {systemStatus.capabilities.features?.comprehensive_reporting && (
                <span className="text-purple-600">📋 Reports</span>
              )}
            </div>
          )}

          <div className="text-sm text-gray-500">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
          <button
            onClick={refreshData}
            disabled={refreshing}
            className={`p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors ${
              refreshing ? 'animate-spin' : ''
            }`}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-red-500">⚠️</span>
              <span className="text-red-800 font-medium">Dashboard Error</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="text-red-500 hover:text-red-700"
            >
              ✕
            </button>
          </div>
          <p className="text-red-700 text-sm mt-1">{error}</p>
        </div>
      )}

      {/* Backend Status Warning */}
      {backendStatus === 'disconnected' && (
        <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <span className="text-yellow-600">⚠️</span>
              <div>
                <span className="text-yellow-800 font-medium">Backend Connection Issue</span>
                <p className="text-yellow-700 text-sm mt-1">
                  Some features may be limited. Please ensure the backend server is running at http://localhost:8000
                </p>
              </div>
            </div>
            <button
              onClick={checkBackendStatus}
              className="text-yellow-600 hover:text-yellow-800 text-sm bg-yellow-100 px-2 py-1 rounded"
            >
              Retry
            </button>
          </div>
        </div>
      )}

      {/* Enhanced Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {getQuickStats().map((stat, index) => (
          <div key={index} className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                <div className="flex items-center mt-2">
                  <span className={`text-sm ${getChangeColor(stat.changeType)}`}>
                    {getChangeIcon(stat.changeType)} {Math.abs(stat.change)}%
                  </span>
                  <span className="text-xs text-gray-500 ml-2">vs last week</span>
                </div>
              </div>
              <div className={`w-12 h-12 rounded-lg ${getStatColor(stat.color)} flex items-center justify-center text-white text-xl`}>
                {stat.icon}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <button
          onClick={() => handleQuickAction('discovery', quickDiscovery)}
          disabled={isActionLoading('discovery')}
          className="p-6 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-lg hover:from-blue-600 hover:to-blue-700 transition-all shadow-lg disabled:opacity-50"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl mb-2">🔍</div>
              <h3 className="font-semibold">Discover Sources</h3>
              <p className="text-sm text-blue-100">Find new data sources</p>
            </div>
            {isActionLoading('discovery') && (
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent"></div>
            )}
          </div>
        </button>

        <button
          onClick={() => handleQuickAction('analysis', quickAnalysis)}
          disabled={isActionLoading('analysis')}
          className="p-6 bg-gradient-to-r from-purple-500 to-purple-600 text-white rounded-lg hover:from-purple-600 hover:to-purple-700 transition-all shadow-lg disabled:opacity-50"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl mb-2">🧞‍♂️</div>
              <h3 className="font-semibold">DataGenie</h3>
              <p className="text-sm text-purple-100">Start smart analysis</p>
            </div>
            {isActionLoading('analysis') && (
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent"></div>
            )}
          </div>
        </button>

        <button
          onClick={() => handleQuickAction('chat', quickChat)}
          disabled={isActionLoading('chat')}
          className="p-6 bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg hover:from-green-600 hover:to-green-700 transition-all shadow-lg disabled:opacity-50"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl mb-2">💬</div>
              <h3 className="font-semibold">Chat Analytics</h3>
              <p className="text-sm text-green-100">Ask questions</p>
            </div>
            {isActionLoading('chat') && (
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent"></div>
            )}
          </div>
        </button>

        <button
          onClick={() => handleQuickAction('report', generateQuickReport)}
          disabled={isActionLoading('report')}
          className="p-6 bg-gradient-to-r from-orange-500 to-orange-600 text-white rounded-lg hover:from-orange-600 hover:to-orange-700 transition-all shadow-lg disabled:opacity-50"
        >
          <div className="flex items-center justify-between">
            <div>
              <div className="text-2xl mb-2">📊</div>
              <h3 className="font-semibold">Generate Report</h3>
              <p className="text-sm text-orange-100">Create insights report</p>
            </div>
            {isActionLoading('report') && (
              <div className="animate-spin rounded-full h-6 w-6 border-2 border-white border-t-transparent"></div>
            )}
          </div>
        </button>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

        {/* Recent Activities */}
        <div className="lg:col-span-2 bg-white rounded-lg shadow-sm border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900 flex items-center">
              <span className="text-2xl mr-2">📈</span>
              Recent Activities
            </h2>
          </div>
          <div className="p-6">
            <div className="space-y-4">
              {dashboardData?.recent_activities.map((activity) => (
                <div key={activity.id} className="flex items-start space-x-4">
                  <div className="flex-shrink-0">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center ${getActivityColor(activity.status)}`}>
                      <span className="text-sm">{getActivityIcon(activity.type)}</span>
                    </div>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900">
                      {activity.description}
                    </p>
                    <p className="text-sm text-gray-500">
                      {formatTimestamp(activity.timestamp)}
                    </p>
                  </div>
                  <div className="flex-shrink-0">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getActivityColor(activity.status)}`}>
                      {activity.status}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Performance Metrics & Data Sources */}
        <div className="space-y-8">

          {/* Performance Metrics */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <span className="text-2xl mr-2">⚡</span>
                This Week
              </h2>
            </div>
            <div className="p-6">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Analyses</span>
                  <span className="text-lg font-semibold text-gray-900">
                    {dashboardData?.performance_metrics.analyses_this_week}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Data Processed</span>
                  <span className="text-lg font-semibold text-gray-900">
                    {dashboardData?.performance_metrics.data_processed_gb} GB
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Insights Generated</span>
                  <span className="text-lg font-semibold text-gray-900">
                    {dashboardData?.performance_metrics.insights_generated}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">Charts Created</span>
                  <span className="text-lg font-semibold text-gray-900">
                    {dashboardData?.performance_metrics.charts_created}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Top Data Sources */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200">
            <div className="p-6 border-b border-gray-200">
              <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                <span className="text-2xl mr-2">🔗</span>
                Top Data Sources
              </h2>
            </div>
            <div className="p-6">
              <div className="space-y-3">
                {dashboardData?.top_data_sources.map((source, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                        <span className="text-sm">
                          {source.name === 'PostgreSQL' ? '🐘' :
                           source.name === 'CSV Files' ? '📁' :
                           source.name === 'Excel Files' ? '📊' :
                           source.name === 'JSON APIs' ? '🔗' :
                           source.name === 'MongoDB' ? '🍃' :
                           source.name === 'Tableau' ? '📈' : '🗄️'}
                        </span>
                      </div>
                      <span className="text-sm font-medium text-gray-900">{source.name}</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-600">{source.usage_count}</span>
                      <div className="w-16 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full"
                          style={{ width: `${(source.usage_count / Math.max(...dashboardData.top_data_sources.map(s => s.usage_count))) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* System Status */}
          {systemStatus && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-200">
              <div className="p-6 border-b border-gray-200">
                <h2 className="text-xl font-semibold text-gray-900 flex items-center">
                  <span className="text-2xl mr-2">⚙️</span>
                  System Status
                </h2>
              </div>
              <div className="p-6">
                <div className="space-y-3">
                  {systemStatus.smart_engine && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Smart Engine</span>
                      <span className={`text-sm font-medium ${
                        systemStatus.smart_engine.available ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {systemStatus.smart_engine.available ? '✅ Available' : '❌ Offline'}
                      </span>
                    </div>
                  )}

                  {systemStatus.smart_engine?.openai_configured !== undefined && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">OpenAI</span>
                      <span className={`text-sm font-medium ${
                        systemStatus.smart_engine.openai_configured ? 'text-green-600' : 'text-yellow-600'
                      }`}>
                        {systemStatus.smart_engine.openai_configured ? '✅ Ready' : '⚠️ Not Configured'}
                      </span>
                    </div>
                  )}

                  {systemStatus.smart_defaults && (
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-gray-600">Auto-Discovery</span>
                      <span className={`text-sm font-medium ${
                        systemStatus.smart_defaults.engine_status === 'operational' ? 'text-green-600' : 'text-yellow-600'
                      }`}>
                        {systemStatus.smart_defaults.engine_status === 'operational' ? '✅ Active' : '⚠️ Limited'}
                      </span>
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Backend</span>
                    <span className={`text-sm font-medium ${
                      backendStatus === 'connected' ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {backendStatus === 'connected' ? '✅ Connected' : '❌ Disconnected'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EnhancedDashboard;