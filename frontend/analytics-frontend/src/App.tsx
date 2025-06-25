import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import {
  Home,
  Database,
  MessageCircle,
  FileText,
  Settings,
  Menu,
  X,
  BarChart3,
  Brain,
  Zap,
  RefreshCw,
  AlertCircle,
  CheckCircle
} from 'lucide-react';
import { analyticsAPI, initializeAPI } from './services/api';
import EnhancedDashboard from './components/dashboard/EnhancedDashboard';
import DataGenie from './components/DataGenie';
import ConversationalAnalytics from './components/analytics/ConversationalAnalytics';
import DataDiscovery from './components/discovery/DataDiscovery';
import Reports from './components/dashboard/Reports';

// Navigation component
const Navigation: React.FC<{ isOpen: boolean; onToggle: () => void }> = ({ isOpen, onToggle }) => {
  const location = useLocation();

  const navItems = [
    { path: '/', icon: Home, label: 'Dashboard', description: 'Analytics overview' },
    { path: '/datagenie', icon: Brain, label: 'DataGenie', description: 'Smart analysis' },
    { path: '/chat', icon: MessageCircle, label: 'Chat Analytics', description: 'Ask questions' },
    { path: '/discovery', icon: Database, label: 'Data Discovery', description: 'Find sources' },
    { path: '/reports', icon: FileText, label: 'Reports', description: 'Generate reports' },
  ];

  const isActivePath = (path: string) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onToggle}
        />
      )}

      {/* Sidebar */}
      <div className={`
        fixed top-0 left-0 h-full bg-white border-r border-gray-200 shadow-lg z-50 transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0 lg:static lg:z-auto
        w-64
      `}>
        {/* Header */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <BarChart3 className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900">Analytics Platform</h1>
                <p className="text-xs text-gray-600">Enhanced v5.1.0</p>
              </div>
            </div>

            <button
              onClick={onToggle}
              className="lg:hidden p-1 rounded-lg hover:bg-gray-100"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = isActivePath(item.path);

            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => window.innerWidth < 1024 && onToggle()}
                className={`
                  block p-3 rounded-lg transition-colors group
                  ${isActive 
                    ? 'bg-blue-50 text-blue-700 border border-blue-200' 
                    : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                  }
                `}
              >
                <div className="flex items-center space-x-3">
                  <Icon className={`w-5 h-5 ${isActive ? 'text-blue-600' : 'text-gray-500 group-hover:text-gray-700'}`} />
                  <div>
                    <div className="font-medium">{item.label}</div>
                    <div className="text-xs text-gray-500">{item.description}</div>
                  </div>
                </div>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200">
          <div className="flex items-center space-x-2 text-sm text-gray-600">
            <Zap className="w-4 h-4 text-green-500" />
            <span>All systems operational</span>
          </div>
          <div className="mt-2 text-xs text-gray-500">
            Smart Engine • Auto-Discovery • AI Reports
          </div>
        </div>
      </div>
    </>
  );
};

// Enhanced System Status component (improved from AnalysisInterface)
const SystemStatus: React.FC = () => {
  const [status, setStatus] = useState<{
    backend: 'checking' | 'connected' | 'disconnected';
    initialization: 'pending' | 'success' | 'partial' | 'failed';
    capabilities: any;
    error?: string;
    lastChecked?: Date;
  }>({
    backend: 'checking',
    initialization: 'pending',
    capabilities: null
  });

  const [showDetails, setShowDetails] = useState(false);

  useEffect(() => {
    initializeSystem();
    // Auto-refresh status every 30 seconds
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const initializeSystem = async () => {
    try {
      console.log('🚀 Initializing Enhanced Analytics Platform...');

      // Initialize API and check capabilities
      const initResult = await initializeAPI();

      setStatus(prev => ({
        ...prev,
        backend: initResult.status === 'failed' ? 'disconnected' : 'connected',
        initialization: initResult.status,
        capabilities: initResult.capabilities,
        error: initResult.status === 'failed' ? 'Backend connection failed' : undefined,
        lastChecked: new Date()
      }));

      console.log('✅ Platform initialization complete:', initResult.status);

    } catch (error) {
      console.error('❌ Platform initialization failed:', error);
      setStatus(prev => ({
        ...prev,
        backend: 'disconnected',
        initialization: 'failed',
        capabilities: null,
        error: 'System initialization failed',
        lastChecked: new Date()
      }));
    }
  };

  const checkBackendStatus = async () => {
    try {
      const [health, capabilities] = await Promise.allSettled([
        analyticsAPI.healthCheck(),
        analyticsAPI.getCapabilities()
      ]);

      const isConnected = health.status === 'fulfilled' && health.value.status === 'healthy';

      setStatus(prev => ({
        ...prev,
        backend: isConnected ? 'connected' : 'disconnected',
        capabilities: capabilities.status === 'fulfilled' ? capabilities.value : prev.capabilities,
        lastChecked: new Date(),
        error: isConnected ? undefined : 'Backend health check failed'
      }));
    } catch (error) {
      setStatus(prev => ({
        ...prev,
        backend: 'disconnected',
        lastChecked: new Date(),
        error: 'Backend connection failed'
      }));
    }
  };

  const getStatusColor = () => {
    if (status.backend === 'disconnected') return 'bg-red-500';
    if (status.initialization === 'partial') return 'bg-yellow-500';
    if (status.initialization === 'success') return 'bg-green-500';
    return 'bg-gray-500';
  };

  const getStatusText = () => {
    if (status.backend === 'disconnected') return 'Offline Mode';
    if (status.initialization === 'partial') return 'Limited Features';
    if (status.initialization === 'success') return 'All Systems Ready';
    return 'Initializing...';
  };

  const getStatusIcon = () => {
    if (status.backend === 'disconnected') return AlertCircle;
    if (status.initialization === 'success') return CheckCircle;
    return Zap;
  };

  const StatusIcon = getStatusIcon();

  if (status.initialization === 'failed') {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
        <div className="flex items-start space-x-3">
          <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-sm font-medium text-red-800">System Initialization Failed</h3>
            <p className="mt-1 text-sm text-red-700">{status.error}</p>
            <div className="mt-3 space-x-2">
              <button
                onClick={initializeSystem}
                className="bg-red-600 hover:bg-red-700 text-white px-3 py-1 rounded text-sm inline-flex items-center space-x-2"
              >
                <RefreshCw className="w-3 h-3" />
                <span>Retry Initialization</span>
              </button>
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-red-600 hover:text-red-800 text-sm"
              >
                {showDetails ? 'Hide' : 'Show'} Details
              </button>
            </div>
            {showDetails && (
              <div className="mt-3 p-3 bg-red-100 rounded text-xs text-red-800">
                <strong>Troubleshooting:</strong>
                <ul className="mt-1 space-y-1 list-disc list-inside">
                  <li>Ensure backend server is running at http://localhost:8000</li>
                  <li>Check network connectivity</li>
                  <li>Verify server configuration</li>
                  <li>Review browser console for errors</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 mb-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`w-3 h-3 rounded-full ${getStatusColor()} ${status.backend === 'checking' ? 'animate-pulse' : ''}`} />
          <StatusIcon className={`w-4 h-4 ${
            status.backend === 'disconnected' ? 'text-red-500' :
            status.initialization === 'success' ? 'text-green-500' : 'text-yellow-500'
          }`} />
          <div>
            <span className="text-sm font-medium text-gray-900">{getStatusText()}</span>
            {status.capabilities && (
              <div className="text-xs text-gray-500 mt-1">
                Backend: {status.backend === 'connected' ? 'Connected' : 'Offline'} •
                Features: {Object.values(status.capabilities.features || {}).filter(Boolean).length} active
                {status.lastChecked && (
                  <> • Last checked: {status.lastChecked.toLocaleTimeString()}</>
                )}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={checkBackendStatus}
            className="text-gray-400 hover:text-gray-600 p-1 rounded-lg hover:bg-gray-100"
            title="Refresh Status"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-gray-400 hover:text-gray-600 text-sm"
          >
            {showDetails ? 'Hide' : 'Details'}
          </button>
        </div>
      </div>

      {showDetails && status.capabilities && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-xs">
            {status.capabilities.smart_features?.unified_smart_engine && (
              <div className="flex items-center space-x-2 text-blue-600">
                <CheckCircle className="w-3 h-3" />
                <span>Smart Engine</span>
              </div>
            )}
            {status.capabilities.smart_features?.auto_data_discovery && (
              <div className="flex items-center space-x-2 text-green-600">
                <CheckCircle className="w-3 h-3" />
                <span>Auto-Discovery</span>
              </div>
            )}
            {status.capabilities.features?.comprehensive_reporting && (
              <div className="flex items-center space-x-2 text-purple-600">
                <CheckCircle className="w-3 h-3" />
                <span>Smart Reports</span>
              </div>
            )}
            {status.capabilities.features?.chart_intelligence && (
              <div className="flex items-center space-x-2 text-orange-600">
                <CheckCircle className="w-3 h-3" />
                <span>Chart Intelligence</span>
              </div>
            )}
            {status.capabilities.features?.adaptive_query_processing && (
              <div className="flex items-center space-x-2 text-indigo-600">
                <CheckCircle className="w-3 h-3" />
                <span>Adaptive Processing</span>
              </div>
            )}
            {status.capabilities.smart_features?.llm_powered_query_understanding && (
              <div className="flex items-center space-x-2 text-pink-600">
                <CheckCircle className="w-3 h-3" />
                <span>AI Understanding</span>
              </div>
            )}
          </div>

          <div className="mt-3 text-xs text-gray-500">
            <strong>Available Formats:</strong> {status.capabilities.data_formats?.join(', ') || 'CSV, Excel, JSON'}
            <br />
            <strong>Analysis Types:</strong> {status.capabilities.analysis_types?.slice(0, 3).join(', ') || 'Trend, Distribution, Comparison'}
            {status.capabilities.analysis_types?.length > 3 && ` +${status.capabilities.analysis_types.length - 3} more`}
          </div>

          {status.error && (
            <div className="mt-3 p-2 bg-red-50 border border-red-200 rounded text-xs text-red-700">
              <strong>Error:</strong> {status.error}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Main App component
const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <Navigation isOpen={sidebarOpen} onToggle={toggleSidebar} />

        {/* Main Content */}
        <div className="lg:ml-64">
          {/* Mobile Header */}
          <div className="lg:hidden bg-white border-b border-gray-200 px-4 py-3">
            <div className="flex items-center justify-between">
              <button
                onClick={toggleSidebar}
                className="p-2 rounded-lg hover:bg-gray-100"
              >
                <Menu className="w-5 h-5 text-gray-600" />
              </button>

              <div className="flex items-center space-x-2">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                <span className="font-semibold text-gray-900">Analytics Platform</span>
              </div>

              <div className="w-9"></div> {/* Spacer for centering */}
            </div>
          </div>

          {/* Page Content */}
          <main className="min-h-screen">
            <div className="p-6">
              {/* Enhanced System Status */}
              <SystemStatus />

              {/* Routes */}
              <Routes>
                <Route path="/" element={<EnhancedDashboard />} />
                <Route path="/datagenie" element={<DataGenie />} />
                <Route path="/chat" element={<ConversationalAnalytics />} />
                <Route path="/discovery" element={<DataDiscovery />} />
                <Route path="/reports" element={<Reports />} />

                {/* Legacy redirects */}
                <Route path="/dashboard" element={<Navigate to="/" replace />} />
                <Route path="/dashboard/reports" element={<Navigate to="/reports" replace />} />
                <Route path="/analysis" element={<Navigate to="/datagenie" replace />} />

                {/* Catch all - redirect to dashboard */}
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </div>
          </main>
        </div>
      </div>
    </Router>
  );
};

export default App;