
import React, { useState, useEffect } from 'react';
import { analyticsAPI } from '../../services/api';

interface APITestResult {
  status: 'success' | 'error' | 'loading';
  data?: any;
  error?: string;
  responseTime?: number;
}

interface UserPreferences {
  default_analysis_mode: 'fast' | 'balanced' | 'thorough';
  auto_charts: boolean;
  email_notifications: boolean;
  data_retention_days: number;
  theme: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  auto_discovery: boolean;
}

const Settings: React.FC = () => {
  const [activeTab, setActiveTab] = useState('general');
  const [preferences, setPreferences] = useState<UserPreferences>({
    default_analysis_mode: 'balanced',
    auto_discovery: true,
    auto_charts: true,
    email_notifications: false,
    data_retention_days: 90,
    theme: 'light',
    language: 'en',
    timezone: 'UTC'
  });
  const [apiTests, setApiTests] = useState<{ [key: string]: APITestResult }>({});
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [isSaving, setIsSaving] = useState(false);

  const tabs = [
    { id: 'general', name: 'General', icon: '⚙️' },
    { id: 'preferences', name: 'Preferences', icon: '🎨' },
    { id: 'data', name: 'Data Sources', icon: '📊' },
    { id: 'api', name: 'API Testing', icon: '🔌' },
    { id: 'security', name: 'Security', icon: '🔒' },
  ];

  useEffect(() => {
    loadPreferences();
    if (activeTab === 'api') {
      runHealthCheck();
    }
  }, [activeTab]);

  const loadPreferences = () => {
    try {
      const saved = localStorage.getItem('user_preferences');
      if (saved) {
        setPreferences({ ...preferences, ...JSON.parse(saved) });
      }
    } catch (error) {
      console.error('Failed to load preferences:', error);
    }
  };

  const savePreferences = async () => {
    setIsSaving(true);
    try {
      localStorage.setItem('user_preferences', JSON.stringify(preferences));
      
      // Also save to backend if available - use the correct property names
      try {
        const backendPreferences = {
          default_analysis_mode: preferences.default_analysis_mode,
          auto_charts: preferences.auto_charts,
          email_notifications: preferences.email_notifications,
          data_retention_days: preferences.data_retention_days
        };
        await analyticsAPI.updateUserPreferences(backendPreferences);
      } catch (error) {
        console.log('Backend save failed, using local storage only');
      }
      
      setTimeout(() => setIsSaving(false), 500);
    } catch (error) {
      console.error('Failed to save preferences:', error);
      setIsSaving(false);
    }
  };

  const updatePreference = (key: keyof UserPreferences, value: any) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
  };

  const runAPITest = async (testName: string, testFunction: () => Promise<any>) => {
    const startTime = Date.now();
    setApiTests(prev => ({ ...prev, [testName]: { status: 'loading' } }));

    try {
      const result = await testFunction();
      const responseTime = Date.now() - startTime;
      
      setApiTests(prev => ({
        ...prev,
        [testName]: {
          status: 'success',
          data: result,
          responseTime
        }
      }));
    } catch (error: any) {
      const responseTime = Date.now() - startTime;
      setApiTests(prev => ({
        ...prev,
        [testName]: {
          status: 'error',
          error: error.message || 'Test failed',
          responseTime
        }
      }));
    }
  };

  const runHealthCheck = () => runAPITest('health', () => analyticsAPI.healthCheck());
  const runSystemStatus = () => runAPITest('status', () => analyticsAPI.getSystemStatus());
  const runDataDiscovery = () => runAPITest('discovery', () => analyticsAPI.discoverDataSources({ mode: 'fast' }));
  const runRecommendations = () => runAPITest('recommendations', () => analyticsAPI.getRecommendations());
  const runPing = () => runAPITest('ping', () => analyticsAPI.ping());

  const getTestStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-green-600 bg-green-100';
      case 'error': return 'text-red-600 bg-red-100';
      case 'loading': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTestIcon = (status: string) => {
    switch (status) {
      case 'success': return '✅';
      case 'error': return '❌';
      case 'loading': return '⏳';
      default: return '⚪';
    }
  };

  const APITestButton: React.FC<{
    label: string;
    description: string;
    testKey: string;
    onTest: () => void;
  }> = ({ label, description, testKey, onTest }) => {
    const test = apiTests[testKey];
    
    return (
      <div className="p-4 border border-gray-200 rounded-lg">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="font-semibold text-gray-900">{label}</h3>
            <p className="text-sm text-gray-600">{description}</p>
          </div>
          <button
            onClick={onTest}
            disabled={test?.status === 'loading'}
            className={`btn-primary text-sm px-4 py-2 ${
              test?.status === 'loading' ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {test?.status === 'loading' ? (
              <svg className="animate-spin w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            ) : (
              'Test'
            )}
          </button>
        </div>

        {test && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className={`px-2 py-1 text-xs rounded-full ${getTestStatusColor(test.status)}`}>
                {getTestIcon(test.status)} {test.status}
              </span>
              {test.responseTime && (
                <span className="text-xs text-gray-500">{test.responseTime}ms</span>
              )}
            </div>
            
            {test.error && (
              <div className="text-sm text-red-600 bg-red-50 p-2 rounded">
                {test.error}
              </div>
            )}
            
            {test.status === 'success' && (
              <div className="text-sm text-green-600 bg-green-50 p-2 rounded">
                ✅ Test passed - Check console for details
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Settings</h1>
        <p className="text-gray-600">
          Configure your analytics platform preferences and test integrations
        </p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-8">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-2 px-1 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.icon} {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <div className="lg:col-span-3">
          {/* General Settings */}
          {activeTab === 'general' && (
            <div className="space-y-6">
              <div className="card">
                <h2 className="text-xl font-semibold text-gray-900 mb-6">General Settings</h2>
                
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Default Analysis Mode
                    </label>
                    <select 
                      value={preferences.default_analysis_mode}
                      onChange={(e) => updatePreference('default_analysis_mode', e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="fast">Fast - Quick analysis with basic insights</option>
                      <option value="balanced">Balanced - Good balance of speed and depth</option>
                      <option value="thorough">Thorough - Comprehensive analysis (slower)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Data Retention Period
                    </label>
                    <select 
                      value={preferences.data_retention_days}
                      onChange={(e) => updatePreference('data_retention_days', parseInt(e.target.value))}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value={30}>30 days</option>
                      <option value={90}>90 days</option>
                      <option value={180}>6 months</option>
                      <option value={365}>1 year</option>
                    </select>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="autoDiscovery"
                        checked={preferences.auto_discovery}
                        onChange={(e) => updatePreference('auto_discovery', e.target.checked)}
                        className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                      />
                      <label htmlFor="autoDiscovery" className="ml-3 text-sm font-medium text-gray-700">
                        Enable automatic data source discovery
                      </label>
                    </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="autoCharts"
                        checked={preferences.auto_charts}
                        onChange={(e) => updatePreference('auto_charts', e.target.checked)}
                        className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                      />
                      <label htmlFor="autoCharts" className="ml-3 text-sm font-medium text-gray-700">
                        Include charts by default in analysis
                      </label>
                    </div>

                    <div className="flex items-center">
                      <input
                        type="checkbox"
                        id="emailNotifications"
                        checked={preferences.email_notifications}
                        onChange={(e) => updatePreference('email_notifications', e.target.checked)}
                        className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                      />
                      <label htmlFor="emailNotifications" className="ml-3 text-sm font-medium text-gray-700">
                        Send email notifications for completed analyses
                      </label>
                    </div>
                  </div>
                </div>

                <div className="mt-6 pt-6 border-t border-gray-200">
                  <button
                    onClick={savePreferences}
                    disabled={isSaving}
                    className={`btn-primary ${isSaving ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {isSaving ? (
                      <>
                        <svg className="animate-spin w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        Saving...
                      </>
                    ) : (
                      'Save Settings'
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Preferences */}
          {activeTab === 'preferences' && (
            <div className="space-y-6">
              <div className="card">
                <h2 className="text-xl font-semibold text-gray-900 mb-6">User Preferences</h2>
                
                <div className="space-y-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Theme
                    </label>
                    <select 
                      value={preferences.theme}
                      onChange={(e) => updatePreference('theme', e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                      <option value="auto">Auto (System)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Language
                    </label>
                    <select 
                      value={preferences.language}
                      onChange={(e) => updatePreference('language', e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="en">English</option>
                      <option value="es">Español</option>
                      <option value="fr">Français</option>
                      <option value="de">Deutsch</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Timezone
                    </label>
                    <select 
                      value={preferences.timezone}
                      onChange={(e) => updatePreference('timezone', e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="UTC">UTC</option>
                      <option value="America/New_York">Eastern Time</option>
                      <option value="America/Chicago">Central Time</option>
                      <option value="America/Denver">Mountain Time</option>
                      <option value="America/Los_Angeles">Pacific Time</option>
                      <option value="Europe/London">London</option>
                      <option value="Europe/Paris">Paris</option>
                      <option value="Asia/Tokyo">Tokyo</option>
                    </select>
                  </div>
                </div>

                <div className="mt-6 pt-6 border-t border-gray-200">
                  <button
                    onClick={savePreferences}
                    disabled={isSaving}
                    className={`btn-primary ${isSaving ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    Save Preferences
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Data Sources */}
          {activeTab === 'data' && (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Data Source Settings</h2>
              <div className="space-y-4">
                <div className="p-4 bg-gray-50 rounded-lg">
                  <h3 className="font-medium text-gray-900 mb-2">Connected Sources</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">PostgreSQL Database</span>
                      <span className="text-green-600 font-medium">Connected</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">CSV Files (3)</span>
                      <span className="text-green-600 font-medium">Active</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-gray-700">Excel Workbooks (2)</span>
                      <span className="text-green-600 font-medium">Active</span>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-blue-50 rounded-lg">
                  <h3 className="font-medium text-blue-900 mb-2">Data Source Permissions</h3>
                  <div className="space-y-2">
                    <label className="flex items-center">
                      <input type="checkbox" className="mr-2" defaultChecked />
                      <span className="text-sm text-blue-800">Allow automatic schema detection</span>
                    </label>
                    <label className="flex items-center">
                      <input type="checkbox" className="mr-2" defaultChecked />
                      <span className="text-sm text-blue-800">Enable data sampling for previews</span>
                    </label>
                    <label className="flex items-center">
                      <input type="checkbox" className="mr-2" />
                      <span className="text-sm text-blue-800">Cache query results</span>
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* API Testing */}
          {activeTab === 'api' && (
            <div className="space-y-6">
              <div className="card">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-semibold text-gray-900">API Integration Test</h2>
                  <div className="text-sm text-gray-500">
                    Base URL: {analyticsAPI.getBaseURL()}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <APITestButton
                    label="Health Check"
                    description="Basic connectivity test"
                    testKey="health"
                    onTest={runHealthCheck}
                  />

                  <APITestButton
                    label="System Status"
                    description="Detailed system information"
                    testKey="status"
                    onTest={runSystemStatus}
                  />

                  <APITestButton
                    label="Data Discovery"
                    description="Test source discovery functionality"
                    testKey="discovery"
                    onTest={runDataDiscovery}
                  />

                  <APITestButton
                    label="Recommendations"
                    description="AI recommendation engine test"
                    testKey="recommendations"
                    onTest={runRecommendations}
                  />

                  <APITestButton
                    label="Ping Test"
                    description="Simple response time test"
                    testKey="ping"
                    onTest={runPing}
                  />

                  <div className="p-4 border border-gray-200 rounded-lg">
                    <h3 className="font-semibold text-gray-900 mb-2">Connection Info</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-600">User ID:</span>
                        <span className="font-mono text-xs">{analyticsAPI.getUserId()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Connected:</span>
                        <span className={analyticsAPI.isConnected() ? 'text-green-600' : 'text-red-600'}>
                          {analyticsAPI.isConnected() ? 'Yes' : 'No'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Security */}
          {activeTab === 'security' && (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-6">Security Settings</h2>
              <div className="space-y-6">
                <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                  <h3 className="font-medium text-yellow-900 mb-2">⚠️ Security Notice</h3>
                  <p className="text-sm text-yellow-800">
                    This is a demo application. In production, implement proper authentication,
                    data encryption, and access controls.
                  </p>
                </div>

                <div className="space-y-4">
                  <div>
                    <h3 className="font-medium text-gray-900 mb-3">Data Privacy</h3>
                    <div className="space-y-2">
                      <label className="flex items-center">
                        <input type="checkbox" className="mr-2" defaultChecked />
                        <span className="text-sm text-gray-700">Encrypt data at rest</span>
                      </label>
                      <label className="flex items-center">
                        <input type="checkbox" className="mr-2" defaultChecked />
                        <span className="text-sm text-gray-700">Use HTTPS for all connections</span>
                      </label>
                      <label className="flex items-center">
                        <input type="checkbox" className="mr-2" />
                        <span className="text-sm text-gray-700">Enable audit logging</span>
                      </label>
                    </div>
                  </div>

                  <div>
                    <h3 className="font-medium text-gray-900 mb-3">Access Control</h3>
                    <div className="space-y-2">
                      <label className="flex items-center">
                        <input type="checkbox" className="mr-2" defaultChecked />
                        <span className="text-sm text-gray-700">Require authentication</span>
                      </label>
                      <label className="flex items-center">
                        <input type="checkbox" className="mr-2" />
                        <span className="text-sm text-gray-700">Enable two-factor authentication</span>
                      </label>
                      <label className="flex items-center">
                        <input type="checkbox" className="mr-2" />
                        <span className="text-sm text-gray-700">Role-based permissions</span>
                      </label>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                🔄 Reset to Defaults
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                📥 Export Settings
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                📤 Import Settings
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                🗑️ Clear All Data
              </button>
            </div>
          </div>

          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">System Information</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Version:</span>
                <span className="font-medium">5.0.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Build:</span>
                <span className="font-medium">2024.01.15</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Environment:</span>
                <span className="font-medium">Development</span>
              </div>
            </div>
          </div>

          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Support</h3>
            <div className="space-y-2">
              <button className="w-full text-left px-3 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg">
                📚 Documentation
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg">
                💬 Contact Support
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg">
                🐛 Report Bug
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Settings;
