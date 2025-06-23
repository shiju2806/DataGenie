import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp, AlertTriangle, Target, DollarSign, Activity, CheckCircle, XCircle, RefreshCw, Download, Wifi, WifiOff } from 'lucide-react';

const EnhancedDashboard = () => {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [autoInsights, setAutoInsights] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [debugInfo, setDebugInfo] = useState('');

  useEffect(() => {
    fetchDashboardData();
    // Check backend health first
    checkBackendHealth();
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });

      if (response.ok) {
        const health = await response.json();
        setConnectionStatus('connected');
        setDebugInfo(`Backend version: ${health.version}, Status: ${health.status}`);
      } else {
        setConnectionStatus('error');
        setDebugInfo(`Backend responded with status: ${response.status}`);
      }
    } catch (err) {
      setConnectionStatus('disconnected');
      setDebugInfo(`Cannot connect to backend: ${err.message}`);
    }
  };

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      console.log('🔄 Attempting to fetch dashboard data...');

      const response = await fetch('http://localhost:8000/api/dashboard', {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        // Add timeout
        signal: AbortSignal.timeout(10000)
      });

      console.log('📡 Response status:', response.status);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status} - ${response.statusText}`);
      }

      const data = await response.json();
      console.log('✅ Dashboard data received:', data);

      setDashboardData(data);
      setAutoInsights(data.auto_insights);
      setConnectionStatus('connected');

    } catch (error) {
      console.error('❌ Dashboard fetch error:', error);
      setError(error.message);
      setConnectionStatus('error');

      // Set mock data for development
      setMockData();
    } finally {
      setLoading(false);
    }
  };

  const setMockData = () => {
    console.log('🔧 Using mock data for development');
    const mockData = {
      summary: {
        data: [{
          total_policies: 15000,
          total_face_amount: 45000000000,
          average_issue_age: 42.5,
          smoker_rate: 0.18
        }]
      },
      financial: {
        data: [{
          total_claims: 12500000,
          loss_ratio: 0.082,
          net_income: 25000000,
          combined_ratio: 0.95
        }]
      },
      health_score: {
        overall_score: 82.5,
        status: "GOOD",
        color: "blue",
        components: {
          mortality_health: 85.0,
          financial_health: 80.0
        }
      },
      auto_insights: {
        alert_count: 2,
        opportunity_count: 3,
        data: [{
          alerts: [
            {
              severity: 'MEDIUM',
              message: 'TERM Male Non-Smoker showing elevated mortality',
              action: 'Review underwriting guidelines',
              impact: 'Potential 5% increase in claims'
            }
          ],
          opportunities: [
            {
              potential_value: 'HIGH',
              message: 'WHOLE life segment showing strong profitability',
              action: 'Increase marketing focus',
              estimated_impact: '+15% profit potential'
            }
          ],
          recommendations: [
            {
              priority: 'HIGH',
              recommendation: 'Expand profitable WHOLE life segment',
              rationale: 'Strong margins and low claims',
              timeline: '3-6 months',
              expected_outcome: '10-15% portfolio growth'
            }
          ]
        }]
      },
      last_updated: new Date().toISOString()
    };

    setDashboardData(mockData);
    setAutoInsights(mockData.auto_insights);
  };

  const ConnectionIndicator = () => {
    const statusConfig = {
      checking: { icon: RefreshCw, color: 'text-yellow-500', bg: 'bg-yellow-50', text: 'Checking connection...' },
      connected: { icon: Wifi, color: 'text-green-500', bg: 'bg-green-50', text: 'Connected to backend' },
      disconnected: { icon: WifiOff, color: 'text-red-500', bg: 'bg-red-50', text: 'Backend disconnected' },
      error: { icon: AlertTriangle, color: 'text-red-500', bg: 'bg-red-50', text: 'Connection error' }
    };

    const config = statusConfig[connectionStatus];
    const Icon = config.icon;

    return (
      <div className={`flex items-center space-x-2 px-3 py-2 rounded-lg ${config.bg}`}>
        <Icon className={`h-4 w-4 ${config.color} ${connectionStatus === 'checking' ? 'animate-spin' : ''}`} />
        <span className={`text-sm ${config.color}`}>{config.text}</span>
      </div>
    );
  };

  const HealthScoreIndicator = ({ score, status, color }) => {
    const colorClasses = {
      green: 'bg-green-500',
      blue: 'bg-blue-500',
      yellow: 'bg-yellow-500',
      red: 'bg-red-500'
    };

    return (
      <div className="flex items-center space-x-3">
        <div className={`w-16 h-16 rounded-full flex items-center justify-center text-2xl font-bold text-white ${colorClasses[color]}`}>
          {score}
        </div>
        <div>
          <div className="text-lg font-semibold">{status}</div>
          <div className="text-sm text-gray-500">Portfolio Health</div>
        </div>
      </div>
    );
  };

  const MetricCard = ({ icon: Icon, title, value, subtitle, color = "blue" }) => {
    const colorClasses = {
      blue: 'text-blue-600',
      red: 'text-red-600',
      green: 'text-green-600',
      purple: 'text-purple-600'
    };

    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className={`text-2xl font-bold ${colorClasses[color]}`}>{value}</p>
            {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
          </div>
          <Icon className={`h-8 w-8 ${colorClasses[color]}`} />
        </div>
      </div>
    );
  };

  // Loading state
  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="text-gray-600">Loading dashboard insights...</p>
          <ConnectionIndicator />
          {debugInfo && (
            <div className="text-xs text-gray-500 max-w-md">
              Debug: {debugInfo}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Error state with debugging info
  if (error && !dashboardData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center space-y-6 max-w-2xl mx-auto p-8">
          <div className="text-red-500">
            <AlertTriangle className="h-16 w-16 mx-auto mb-4" />
            <h2 className="text-2xl font-bold">Dashboard Connection Failed</h2>
          </div>

          <div className="bg-white rounded-lg shadow p-6 text-left">
            <h3 className="font-semibold mb-3">🔍 Debugging Information:</h3>
            <div className="space-y-2 text-sm">
              <div><strong>Error:</strong> {error}</div>
              <div><strong>Expected URL:</strong> http://localhost:8000/api/dashboard</div>
              <div><strong>Status:</strong> <ConnectionIndicator /></div>
              {debugInfo && <div><strong>Details:</strong> {debugInfo}</div>}
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h3 className="font-semibold text-blue-900 mb-2">🛠️ Troubleshooting Steps:</h3>
            <ol className="text-left text-sm text-blue-800 space-y-1">
              <li>1. Ensure backend is running: <code className="bg-blue-100 px-1 rounded">python main.py</code></li>
              <li>2. Check if port 8000 is available</li>
              <li>3. Test API directly: <a href="http://localhost:8000" target="_blank" rel="noopener noreferrer" className="underline">http://localhost:8000</a></li>
              <li>4. Verify CORS settings in backend</li>
              <li>5. Check browser console for detailed errors</li>
            </ol>
          </div>

          <div className="flex space-x-4">
            <button
              onClick={fetchDashboardData}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry Connection
            </button>
            <button
              onClick={setMockData}
              className="px-6 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
            >
              Use Demo Data
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Success state - render dashboard
  const summary = dashboardData?.summary?.data?.[0] || {};
  const financial = dashboardData?.financial?.data?.[0] || {};
  const healthScore = dashboardData?.health_score || { overall_score: 75, status: 'GOOD', color: 'blue', components: { mortality_health: 75, financial_health: 75 }};

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header with connection status */}
      <div className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Portfolio Dashboard</h1>
              <p className="text-gray-600">Real-time insights and analytics</p>
            </div>
            <div className="flex items-center space-x-4">
              <ConnectionIndicator />
              <button
                onClick={fetchDashboardData}
                className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                <RefreshCw className="h-4 w-4 mr-2" />
                Refresh
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        {/* Health Score & Key Metrics */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-1 bg-white rounded-lg shadow p-6">
            <HealthScoreIndicator
              score={healthScore.overall_score}
              status={healthScore.status}
              color={healthScore.color}
            />
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm">
                <span>Mortality Health</span>
                <span className="font-medium">{healthScore.components.mortality_health}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span>Financial Health</span>
                <span className="font-medium">{healthScore.components.financial_health}</span>
              </div>
            </div>
          </div>

          <MetricCard
            icon={BarChart3}
            title="Total Policies"
            value={summary.total_policies?.toLocaleString() || 'N/A'}
            subtitle={summary.total_face_amount ? `$${(summary.total_face_amount / 1e9).toFixed(1)}B face amount` : ''}
            color="blue"
          />

          <MetricCard
            icon={DollarSign}
            title="Claims Paid"
            value={financial.total_claims ? `$${(financial.total_claims / 1e6).toFixed(1)}M` : 'N/A'}
            subtitle={financial.loss_ratio ? `${(financial.loss_ratio * 100).toFixed(1)}% loss ratio` : ''}
            color="red"
          />

          <MetricCard
            icon={TrendingUp}
            title="Net Income"
            value={financial.net_income ? `$${(financial.net_income / 1e6).toFixed(1)}M` : 'N/A'}
            subtitle={financial.combined_ratio ? `${financial.combined_ratio.toFixed(2)} combined ratio` : ''}
            color="green"
          />
        </div>

        {/* Auto-Insights Section */}
        {autoInsights && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Alerts */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                  <AlertTriangle className="h-5 w-5 text-red-500 mr-2" />
                  Active Alerts ({autoInsights.alert_count || 0})
                </h2>
              </div>
              <div className="p-6 space-y-4">
                {autoInsights.data?.[0]?.alerts?.length > 0 ? (
                  autoInsights.data[0].alerts.map((alert, index) => (
                    <div key={index} className="border-l-4 border-yellow-500 bg-yellow-50 p-4">
                      <h4 className="font-medium text-gray-900">{alert.message}</h4>
                      <p className="text-sm text-gray-600 mt-1">{alert.action}</p>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8">
                    <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-2" />
                    <p className="text-gray-500">No active alerts</p>
                  </div>
                )}
              </div>
            </div>

            {/* Opportunities */}
            <div className="bg-white rounded-lg shadow">
              <div className="px-6 py-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center">
                  <Target className="h-5 w-5 text-green-500 mr-2" />
                  Opportunities ({autoInsights.opportunity_count || 0})
                </h2>
              </div>
              <div className="p-6 space-y-4">
                {autoInsights.data?.[0]?.opportunities?.length > 0 ? (
                  autoInsights.data[0].opportunities.map((opportunity, index) => (
                    <div key={index} className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900">{opportunity.message}</h4>
                      <p className="text-sm text-gray-600 mt-1">{opportunity.action}</p>
                    </div>
                  ))
                ) : (
                  <div className="text-center py-8">
                    <Target className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-gray-500">No opportunities identified</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Debug Info (only show in development) */}
        {error && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <p className="text-sm text-yellow-800">
              ⚠️ Running in demo mode due to connection issues. Start your backend to see live data.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default EnhancedDashboard;