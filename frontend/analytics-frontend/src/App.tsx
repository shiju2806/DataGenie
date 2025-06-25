import React, { useState, useEffect, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';

// Main Layout Components
import Header from './components/ui/Header';
import Sidebar from './components/ui/Sidebar';

// Page Components
import EnhancedDashboard from './components/dashboard/EnhancedDashboard';
import DataDiscovery from './components/discovery/DataDiscovery';
import AnalysisInterface from './components/analysis/AnalysisInterface';
import AdvancedAnalytics from './components/analytics/AdvancedAnalytics';
import Reports from './components/dashboard/Reports';
import Settings from './components/ui/Settings';

// Import the enhanced DataGenie component
import DataGenie from './components/DataGenie';

// NEW: Import the Conversational Analytics component
import ConversationalAnalytics from './components/analytics/ConversationalAnalytics';

// Import API initialization
import { initializeAPI, logEvent, getEnvironmentInfo } from './services/api';

// Types for better type safety
interface ApiCapabilities {
  enableAdvancedAnalytics?: boolean;
  enableConversationMode?: boolean;
  enableFileUpload?: boolean;
  enableDataSourceDiscovery?: boolean;
  [key: string]: any;
}

type BackendStatus = 'checking' | 'connected' | 'partial' | 'disconnected';
type InitializationStatus = 'loading' | 'success' | 'partial' | 'failed';

// Welcome/Landing Page - Enhanced with conversational analytics
const WelcomePage: React.FC = () => {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('checking');
  const [apiCapabilities, setApiCapabilities] = useState<ApiCapabilities | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const checkBackendStatus = useCallback(async () => {
    try {
      const result = await initializeAPI();
      setApiCapabilities(result.capabilities);

      if (result.status === 'success') {
        setBackendStatus('connected');
      } else if (result.status === 'partial') {
        setBackendStatus('partial');
      } else {
        setBackendStatus('disconnected');
      }

      // Reset retry count on success
      setRetryCount(0);
    } catch (error) {
      console.error('Backend check failed:', error);
      setBackendStatus('disconnected');
    }
  }, []);

  useEffect(() => {
    checkBackendStatus();
  }, [checkBackendStatus]);

  const handleRetryConnection = useCallback(() => {
    setRetryCount(prev => prev + 1);
    setBackendStatus('checking');
    checkBackendStatus();
  }, [checkBackendStatus]);

  const getStatusIndicator = () => {
    switch (backendStatus) {
      case 'checking':
        return {
          color: 'bg-yellow-500',
          text: retryCount > 0 ? 'Retrying Connection...' : 'Checking Backend...',
          pulse: true
        };
      case 'connected':
        return { color: 'bg-green-500', text: 'Backend Connected', pulse: false };
      case 'partial':
        return { color: 'bg-yellow-500', text: 'Partial Connection', pulse: false };
      case 'disconnected':
        return { color: 'bg-red-500', text: 'Backend Disconnected', pulse: false };
    }
  };

  const statusInfo = getStatusIndicator();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 flex items-center justify-center">
      <div className="max-w-4xl mx-auto px-6 py-12 text-center">
        <div className="mb-8">
          <div className="text-6xl mb-4">🧞‍♂️</div>
          <h1 className="text-5xl font-bold text-gray-900 mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              DataGenie
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            AI-powered data analysis with automated source discovery, natural language queries, and advanced analytics
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <div className="p-6 bg-white rounded-xl shadow-lg border hover:shadow-xl transition-shadow duration-300">
            <div className="text-3xl mb-3">🔍</div>
            <h3 className="font-semibold text-gray-900 mb-2">Smart Discovery</h3>
            <p className="text-sm text-gray-600">Automatically find and connect to data sources</p>
          </div>

          <div className="p-6 bg-white rounded-xl shadow-lg border hover:shadow-xl transition-shadow duration-300">
            <div className="text-3xl mb-3">💬</div>
            <h3 className="font-semibold text-gray-900 mb-2">Natural Language</h3>
            <p className="text-sm text-gray-600">Ask questions in plain English</p>
          </div>

          <div className="p-6 bg-white rounded-xl shadow-lg border hover:shadow-xl transition-shadow duration-300">
            <div className="text-3xl mb-3">📊</div>
            <h3 className="font-semibold text-gray-900 mb-2">Interactive Charts</h3>
            <p className="text-sm text-gray-600">Dynamic visualizations and dashboards</p>
          </div>

          <div className="p-6 bg-white rounded-xl shadow-lg border hover:shadow-xl transition-shadow duration-300">
            <div className="text-3xl mb-3">🧠</div>
            <h3 className="font-semibold text-gray-900 mb-2">Advanced Analytics</h3>
            <p className="text-sm text-gray-600">ML-powered predictions, anomaly detection, and statistical insights</p>
          </div>
        </div>

        {/* Enhanced Feature Highlights */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
          <div className="p-6 bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl border border-purple-200">
            <h3 className="font-semibold text-purple-900 mb-4 flex items-center justify-center">
              🔮 Advanced Analytics Features
            </h3>
            <ul className="text-sm text-purple-700 space-y-2 text-left">
              <li>• Predictive modeling and forecasting</li>
              <li>• Anomaly detection and pattern analysis</li>
              <li>• Correlation and causal analysis</li>
              <li>• Statistical insights and distributions</li>
              <li>• Clustering and segmentation</li>
            </ul>
          </div>

          <div className="p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border border-green-200">
            <h3 className="font-semibold text-green-900 mb-4 flex items-center justify-center">
              ⚡ Intelligent Automation
            </h3>
            <ul className="text-sm text-green-700 space-y-2 text-left">
              <li>• Automatic data source discovery</li>
              <li>• Smart chart suggestions</li>
              <li>• Adaptive analysis based on data</li>
              <li>• Natural language understanding</li>
              <li>• Real-time insights generation</li>
            </ul>
          </div>
        </div>

        {/* Updated Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8">
          <a
            href="/chat"
            className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-medium px-8 py-3 rounded-lg transition-all duration-300 text-lg inline-flex items-center justify-center space-x-2 no-underline shadow-lg hover:shadow-xl transform hover:scale-105"
          >
            <span>💬</span>
            <span>Chat with Data</span>
          </a>
          <a
            href="/datagenie"
            className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-8 py-3 rounded-lg transition-colors text-lg inline-flex items-center justify-center space-x-2 no-underline"
          >
            <span>🧞‍♂️</span>
            <span>Launch DataGenie</span>
          </a>
          <a
            href="/dashboard"
            className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium px-8 py-3 rounded-lg transition-colors text-lg no-underline"
          >
            Classic Dashboard
          </a>
        </div>

        {/* Enhanced Platform Status */}
        <div className="p-6 bg-white bg-opacity-50 backdrop-blur-sm rounded-xl border mb-8">
          <h3 className="font-semibold text-gray-900 mb-4">Platform Status</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="flex items-center justify-center space-x-2">
              <div className={`w-2 h-2 ${statusInfo.color} rounded-full ${statusInfo.pulse ? 'animate-pulse' : ''}`}></div>
              <span className="text-gray-700">{statusInfo.text}</span>
            </div>
            <div className="flex items-center justify-center space-x-2">
              <div className={`w-2 h-2 ${apiCapabilities?.enableAdvancedAnalytics ? 'bg-green-500' : 'bg-yellow-500'} rounded-full`}></div>
              <span className="text-gray-700">Enhanced Analytics</span>
            </div>
            <div className="flex items-center justify-center space-x-2">
              <div className={`w-2 h-2 ${apiCapabilities?.enableConversationMode ? 'bg-green-500' : 'bg-yellow-500'} rounded-full`}></div>
              <span className="text-gray-700">AI Engine Ready</span>
            </div>
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <span className="text-gray-700">Advanced Mode</span>
            </div>
          </div>

          {(backendStatus === 'disconnected' || backendStatus === 'partial') && (
            <div className={`mt-4 p-3 border rounded-lg ${
              backendStatus === 'disconnected' 
                ? 'bg-red-50 border-red-200' 
                : 'bg-yellow-50 border-yellow-200'
            }`}>
              <p className={`text-sm ${
                backendStatus === 'disconnected' ? 'text-red-700' : 'text-yellow-700'
              }`}>
                {backendStatus === 'disconnected' ? (
                  <>
                    ⚠️ Backend server is not running. Some features may be limited.
                    <br />
                    Please start the backend server at <code className="bg-red-100 px-1 rounded">http://localhost:8000</code>
                  </>
                ) : (
                  <>
                    ⚠️ Partial backend connectivity. Some advanced features may be limited.
                  </>
                )}
              </p>

              {retryCount < 3 && (
                <button
                  onClick={handleRetryConnection}
                  className="mt-2 px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
                  disabled={false}
                >
                  Retry Connection
                </button>
              )}
            </div>
          )}

          {backendStatus === 'checking' && (
            <div className="mt-4 p-3 border rounded-lg bg-blue-50 border-blue-200">
              <p className="text-sm text-blue-700">
                🔄 Retrying connection...
              </p>
            </div>
          )}
        </div>

        <div className="text-center">
          <p className="text-sm text-gray-500 mb-2">
            Powered by mathematical engines, knowledge frameworks, and machine learning
          </p>
          <p className="text-xs text-gray-400">
            v2.0 Enhanced Edition with Conversational Analytics • {getEnvironmentInfo().isDevelopment ? 'Development' : 'Production'} Mode
          </p>
        </div>
      </div>
    </div>
  );
};

// Enhanced Error Boundary Component
interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

class ErrorBoundary extends React.Component<
  React.PropsWithChildren<{}>,
  ErrorBoundaryState
> {
  constructor(props: React.PropsWithChildren<{}>) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('App Error Boundary caught an error:', error, errorInfo);
    logEvent('app_error', { error: error.message, stack: error.stack });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-red-50 via-white to-red-50 flex items-center justify-center">
          <div className="max-w-md mx-auto text-center p-6">
            <div className="text-6xl mb-4">⚠️</div>
            <h1 className="text-2xl font-bold text-gray-900 mb-4">Something went wrong</h1>
            <p className="text-gray-600 mb-6">
              We're sorry, but DataGenie encountered an unexpected error.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
            >
              Reload Application
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Main App Component
const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [apiInitialized, setApiInitialized] = useState(false);
  const [initializationStatus, setInitializationStatus] = useState<InitializationStatus>('loading');

  const initializeApp = useCallback(async () => {
    try {
      console.log('🚀 Initializing DataGenie application...');

      const result = await initializeAPI();

      // Map API statuses to initialization statuses
      const statusMapping: Record<string, InitializationStatus> = {
        'success': 'success',
        'partial': 'partial',
        'failed': 'failed'
      };

      const mappedStatus = statusMapping[result.status] || 'failed';
      setInitializationStatus(mappedStatus);
      setApiInitialized(true);

      logEvent('app_initialized', {
        status: result.status,
        capabilities: result.capabilities,
        environment: getEnvironmentInfo()
      });

      console.log('✅ DataGenie application initialized:', result.status);
    } catch (error) {
      console.error('❌ Failed to initialize DataGenie application:', error);
      setInitializationStatus('failed');
      setApiInitialized(true); // Still allow app to work with limited functionality

      logEvent('app_initialization_failed', { error: error });
    }
  }, []);

  useEffect(() => {
    initializeApp();
  }, [initializeApp]);

  // Show loading screen during initialization
  if (!apiInitialized) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">🧞‍♂️</div>
          <h1 className="text-2xl font-bold text-gray-900 mb-4">
            <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              DataGenie
            </span>
          </h1>
          <div className="flex items-center justify-center space-x-2 text-gray-600 mb-4">
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
            <span>Initializing platform...</span>
          </div>

          {initializationStatus === 'failed' && (
            <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg max-w-md mx-auto">
              <p className="text-yellow-700 text-sm">
                ⚠️ Some features may be limited. Continuing with basic functionality...
              </p>
            </div>
          )}

          {initializationStatus === 'partial' && (
            <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg max-w-md mx-auto">
              <p className="text-blue-700 text-sm">
                ℹ️ Partial initialization complete. Some advanced features may be limited.
              </p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <Router>
        <div className="min-h-screen bg-gray-50">
          <Routes>
            {/* Welcome/Landing Page */}
            <Route path="/" element={<WelcomePage />} />

            {/* NEW: Conversational Analytics Interface - Standalone Route */}
            <Route path="/chat" element={<ConversationalAnalytics />} />

            {/* Enhanced DataGenie Interface - Standalone Route */}
            <Route path="/datagenie" element={<DataGenie />} />

            {/* Existing Application Routes with Sidebar/Header */}
            <Route
              path="/dashboard/*"
              element={
                <div className="flex h-screen">
                  <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
                  <div className="flex-1 flex flex-col overflow-hidden">
                    <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
                    <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
                      <Routes>
                        <Route index element={<EnhancedDashboard />} />
                        <Route path="reports" element={<Reports />} />
                      </Routes>
                    </main>
                  </div>
                </div>
              }
            />

            <Route
              path="/discovery"
              element={
                <div className="flex h-screen">
                  <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
                  <div className="flex-1 flex flex-col overflow-hidden">
                    <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
                    <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
                      <DataDiscovery />
                    </main>
                  </div>
                </div>
              }
            />

            <Route
              path="/analysis"
              element={
                <div className="flex h-screen">
                  <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
                  <div className="flex-1 flex flex-col overflow-hidden">
                    <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
                    <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
                      <AnalysisInterface />
                    </main>
                  </div>
                </div>
              }
            />

            <Route
              path="/advanced"
              element={
                <div className="flex h-screen">
                  <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
                  <div className="flex-1 flex flex-col overflow-hidden">
                    <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
                    <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
                      <AdvancedAnalytics />
                    </main>
                  </div>
                </div>
              }
            />

            <Route
              path="/settings"
              element={
                <div className="flex h-screen">
                  <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
                  <div className="flex-1 flex flex-col overflow-hidden">
                    <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
                    <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
                      <Settings />
                    </main>
                  </div>
                </div>
              }
            />

            {/* Redirect unknown routes to home */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </div>
      </Router>
    </ErrorBoundary>
  );
};

export default App;