import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { analyticsAPI } from '../../services/api';

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
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date());
  const [quickActions, setQuickActions] = useState<Array<{ id: string; loading: boolean }>>([]);

  const navigate = useNavigate();

  useEffect(() => {
    loadDashboardData();

    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      refreshData();
    }, 30000);

    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      // Load dashboard data and system status in parallel
      const [dashboardResponse, statusResponse] = await Promise.allSettled([
        analyticsAPI.getAnalyticsDashboard(),
        analyticsAPI.getSystemStatus()
      ]);

      if (dashboardResponse.status === 'fulfilled') {
        setDashboardData(dashboardResponse.value.dashboard_data);
      } else {
        // Use mock data if API fails
        setDashboardData(getMockDashboardData());
      }

      if (statusResponse.status === 'fulfilled') {
        setSystemStatus(statusResponse.value);
      }

      setLastUpdated(new Date());
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      // Use mock data as fallback
      setDashboardData(getMockDashboardData());
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

  const getMockDashboardData = (): DashboardData => ({
    total_analyses: 247,
    success_rate: 94.5,
    avg_processing_time: 3.2,
    top_data_sources: [
      { name: 'PostgreSQL', usage_count: 89 },
      { name: 'CSV Files', usage_count: 67 },
      { name: 'Excel Files', usage_count: 45 },
      { name: 'JSON APIs', usage_count: 23 }
    ],
    recent_activities: [
      {
        id: '1',
        type: 'analysis',
        description: 'Sales trend analysis completed',
        timestamp: new Date(Date.now() - 1000 * 60 * 5).toISOString(),
        status: 'completed'
      },
      {
        id: '2',
        type: 'discovery',
        description: 'New data source discovered: customer_db',
        timestamp: new Date(Date.now() - 1000 * 60 * 15).toISOString(),
        status: 'discovered'
      },
      {
        id: '3',
        type: 'report',
        description: 'Executive summary report generated',
        timestamp: new Date(Date.now() - 1000 * 60 * 30).toISOString(),
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
        change: 1,
        changeType: 'positive',
        icon: '🔗',
        color: 'orange'
      }
    ];
  };

  const handleQuickAction = async (actionId: string, actionFn: () => Promise<void>) => {
    setQuickActions(prev => prev.map(a => a.id === actionId ? { ...a, loading: true } : a));

    try {
      await actionFn();
    } catch (error) {
      console.error(`Quick action ${actionId} failed:`, error);
    } finally {
      setQuickActions(prev => prev.map(a => a.id === actionId ? { ...a, loading: false } : a));
    }
  };

  const quickDiscovery = async () => {
    await analyticsAPI.discoverDataSources({ mode: 'fast', max_recommendations: 3 });
    navigate('/discovery');
  };

  const quickAnalysis = () => {
    navigate('/analysis');
  };

  const generateQuickReport = async () => {
    await analyticsAPI.generateReport('executive', {
      include_charts: true,
      format: 'pdf'
    });
    navigate('/reports');
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
      default: return '📊';
    }
  };

  const getActivityColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'discovered': return 'text-blue-600 bg-blue-100';
      case 'failed': return 'text-red-600 bg-red-100';
      case 'processing': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
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
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Analytics Dashboard</h1>
          <p className="text-gray-600">
            Complete overview of your data analytics platform
          </p>
        </div>

        <div className="flex items-center space-x-4">
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

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {getQuickStats().map((stat, index) => (
          <div key={index} className="card hover:shadow-lg transition-shadow">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-600">{stat.label}</p>
                <p className="text-2xl font-bold text-gray-900 mt-1">{stat.value}</p>
                <div className={`flex items-center mt-2 text-sm ${getChangeColor(stat.changeType)}`}>
                  <span className="mr-1">{getChangeIcon(stat.changeType)}</span>
                  <span>{Math.abs(stat.change)}</span>
                  <span className="text-gray-500 ml-1">vs last week</span>
                </div>
              </div>
              <div className={`w-12 h-12 rounded-lg ${getStatColor(stat.color)} flex items-center justify-center text-white text-xl`}>
                {stat.icon}
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        {/* Main Content Area */}
        <div className="lg:col-span-2 space-y-8">
          {/* Quick Actions */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-6">Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button
                onClick={() => handleQuickAction('discovery', quickDiscovery)}
                className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-center"
              >
                <div className="text-3xl mb-2">🔍</div>
                <h3 className="font-semibold text-gray-900">Quick Discovery</h3>
                <p className="text-sm text-gray-600">Find new data sources</p>
              </button>

              <button
                onClick={quickAnalysis}
                className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-green-400 hover:bg-green-50 transition-colors text-center"
              >
                <div className="text-3xl mb-2">🧠</div>
                <h3 className="font-semibold text-gray-900">Start Analysis</h3>
                <p className="text-sm text-gray-600">Ask questions about data</p>
              </button>

              <button
                onClick={() => handleQuickAction('report', generateQuickReport)}
                className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-colors text-center"
              >
                <div className="text-3xl mb-2">📊</div>
                <h3 className="font-semibold text-gray-900">Generate Report</h3>
                <p className="text-sm text-gray-600">Create executive summary</p>
              </button>
            </div>
          </div>

          {/* Performance Metrics */}
          {dashboardData?.performance_metrics && (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">This Week's Performance</h2>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {dashboardData.performance_metrics.analyses_this_week}
                  </div>
                  <div className="text-sm text-gray-600">Analyses</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {dashboardData.performance_metrics.data_processed_gb.toFixed(1)} GB
                  </div>
                  <div className="text-sm text-gray-600">Data Processed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {dashboardData.performance_metrics.insights_generated}
                  </div>
                  <div className="text-sm text-gray-600">Insights</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {dashboardData.performance_metrics.charts_created}
                  </div>
                  <div className="text-sm text-gray-600">Charts</div>
                </div>
              </div>
            </div>
          )}

          {/* Top Data Sources */}
          {dashboardData?.top_data_sources && (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Most Used Data Sources</h2>
              <div className="space-y-4">
                {dashboardData.top_data_sources.map((source, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                        <span className="text-blue-600 font-semibold">
                          {source.name.slice(0, 2).toUpperCase()}
                        </span>
                      </div>
                      <div>
                        <h3 className="font-medium text-gray-900">{source.name}</h3>
                        <p className="text-sm text-gray-500">{source.usage_count} analyses</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{
                            width: `${(source.usage_count / Math.max(...dashboardData.top_data_sources.map(s => s.usage_count))) * 100}%`
                          }}
                        />
                      </div>
                      <span className="text-sm font-medium text-gray-900">{source.usage_count}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* System Status */}
          {systemStatus && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">System Status</h3>
                <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                  systemStatus.status === 'healthy' 
                    ? 'text-green-600 bg-green-100' 
                    : 'text-red-600 bg-red-100'
                }`}>
                  {systemStatus.status}
                </span>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Uptime</span>
                  <span className="font-medium">
                    {Math.floor(systemStatus.uptime_seconds / 3600)}h
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Version</span>
                  <span className="font-medium">{systemStatus.version}</span>
                </div>
                {systemStatus.performance_metrics && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Success Rate</span>
                    <span className="font-medium text-green-600">
                      {systemStatus.performance_metrics.success_rate_24h?.toFixed(1)}%
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Recent Activity */}
          {dashboardData?.recent_activities && (
            <div className="card">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
              <div className="space-y-3">
                {dashboardData.recent_activities.map((activity) => (
                  <div key={activity.id} className="flex items-start space-x-3">
                    <div className="text-lg">{getActivityIcon(activity.type)}</div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {activity.description}
                      </p>
                      <div className="flex items-center justify-between mt-1">
                        <span className="text-xs text-gray-500">
                          {formatTimestamp(activity.timestamp)}
                        </span>
                        <span className={`px-2 py-1 text-xs rounded-full ${getActivityColor(activity.status)}`}>
                          {activity.status}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Quick Links */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Links</h3>
            <div className="space-y-2">
              <button
                onClick={() => navigate('/discovery')}
                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                🔍 Data Discovery
              </button>
              <button
                onClick={() => navigate('/analysis')}
                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                🧠 AI Analysis
              </button>
              <button
                onClick={() => navigate('/reports')}
                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                📋 Reports
              </button>
              <button
                onClick={() => navigate('/settings')}
                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                ⚙️ Settings
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnhancedDashboard;