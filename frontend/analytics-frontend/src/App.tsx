import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

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

// Import the new unified DataGenie component
import DataGenie from './components/DataGenie';

// Welcome/Landing Page - Updated for DataGenie
const WelcomePage: React.FC = () => {
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
            AI-powered data analysis with automated source discovery and natural language queries
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <div className="card text-center hover:shadow-lg transition-shadow duration-300">
            <div className="text-3xl mb-3">🔍</div>
            <h3 className="font-semibold text-gray-900 mb-2">Smart Discovery</h3>
            <p className="text-sm text-gray-600">Automatically find and connect to data sources</p>
          </div>

          <div className="card text-center hover:shadow-lg transition-shadow duration-300">
            <div className="text-3xl mb-3">💬</div>
            <h3 className="font-semibold text-gray-900 mb-2">Natural Language</h3>
            <p className="text-sm text-gray-600">Ask questions in plain English</p>
          </div>

          <div className="card text-center hover:shadow-lg transition-shadow duration-300">
            <div className="text-3xl mb-3">📊</div>
            <h3 className="font-semibold text-gray-900 mb-2">Interactive Charts</h3>
            <p className="text-sm text-gray-600">Dynamic visualizations and dashboards</p>
          </div>

          <div className="card text-center hover:shadow-lg transition-shadow duration-300">
            <div className="text-3xl mb-3">🧠</div>
            <h3 className="font-semibold text-gray-900 mb-2">AI Insights</h3>
            <p className="text-sm text-gray-600">Machine learning powered analytics</p>
          </div>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <button
            className="btn-primary px-8 py-3 text-lg"
            onClick={() => window.location.href = '/datagenie'}
          >
            Launch DataGenie
          </button>
          <button
            className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium px-8 py-3 rounded-lg transition-colors text-lg"
            onClick={() => window.location.href = '/dashboard'}
          >
            Classic Dashboard
          </button>
        </div>

        <div className="mt-12 p-6 bg-white/50 backdrop-blur-sm rounded-xl border border-white/20">
          <h3 className="font-semibold text-gray-900 mb-4">Platform Status</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-gray-700">Backend Connected</span>
            </div>
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-gray-700">Smart Defaults Active</span>
            </div>
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="text-gray-700">AI Engine Ready</span>
            </div>
            <div className="flex items-center justify-center space-x-2">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <span className="text-gray-700">Learning Mode</span>
            </div>
          </div>
        </div>

        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            Powered by advanced mathematical engines, domain knowledge frameworks, and smart defaults
          </p>
        </div>
      </div>
    </div>
  );
};

// Main App Component
const App: React.FC = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Routes>
          {/* Welcome/Landing Page */}
          <Route path="/" element={<WelcomePage />} />

          {/* NEW: Unified DataGenie Interface */}
          <Route path="/datagenie" element={<DataGenie />} />

          {/* Existing Application Routes with Sidebar/Header */}
          <Route
            path="/*"
            element={
              <div className="flex h-screen">
                {/* Sidebar */}
                <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

                {/* Main Content Area */}
                <div className="flex-1 flex flex-col overflow-hidden">
                  {/* Header */}
                  <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />

                  {/* Page Content */}
                  <main className="flex-1 overflow-x-hidden overflow-y-auto bg-gray-50">
                    <Routes>
                      <Route path="/dashboard" element={<EnhancedDashboard />} />
                      <Route path="/discovery" element={<DataDiscovery />} />
                      <Route path="/analysis" element={<AnalysisInterface />} />
                      <Route path="/advanced" element={<AdvancedAnalytics />} />
                      <Route path="/reports" element={<Reports />} />
                      <Route path="/settings" element={<Settings />} />
                    </Routes>
                  </main>
                </div>
              </div>
            }
          />
        </Routes>
      </div>
    </Router>
  );
};

export default App;