
import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

interface NavItem {
  id: string;
  label: string;
  path: string;
  icon: string;
  badge?: string;
  badgeColor?: string;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const navItems: NavItem[] = [
    {
      id: 'dashboard',
      label: 'Dashboard',
      path: '/dashboard',
      icon: '📊',
    },
    {
      id: 'discovery',
      label: 'Data Discovery',
      path: '/discovery',
      icon: '🔍',
      badge: 'Auto',
      badgeColor: 'bg-green-100 text-green-800'
    },
    {
      id: 'analysis',
      label: 'AI Analysis',
      path: '/analysis',
      icon: '🧠',
      badge: 'AI',
      badgeColor: 'bg-blue-100 text-blue-800'
    },
    {
      id: 'advanced',
      label: 'Advanced Analytics',
      path: '/advanced',
      icon: '🔮',
      badge: 'ML',
      badgeColor: 'bg-purple-100 text-purple-800'
    },
    {
      id: 'reports',
      label: 'Reports',
      path: '/reports',
      icon: '📋',
    },
    {
      id: 'settings',
      label: 'Settings',
      path: '/settings',
      icon: '⚙️',
    },
  ];

  const handleNavigation = (path: string) => {
    navigate(path);
    onClose(); // Close sidebar on mobile after navigation
  };

  const isActivePath = (path: string) => {
    return location.pathname === path;
  };

  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 lg:hidden"
          onClick={onClose}
        >
          <div className="absolute inset-0 bg-gray-600 opacity-75"></div>
        </div>
      )}

      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out
        lg:translate-x-0 lg:static lg:inset-0
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
      `}>
        <div className="flex flex-col h-full">
          {/* Sidebar Header */}
          <div className="flex items-center justify-between p-4 border-b border-gray-200">
            <div className="flex items-center space-x-3">
              <div className="text-2xl">📊</div>
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Analytics</h2>
                <p className="text-xs text-gray-500">v5.0.0</p>
              </div>
            </div>

            {/* Close button for mobile */}
            <button
              onClick={onClose}
              className="lg:hidden p-1 rounded-md text-gray-400 hover:text-gray-600 hover:bg-gray-100"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          {/* Navigation */}
          <nav className="flex-1 px-4 py-6 space-y-2">
            {navItems.map((item) => {
              const isActive = isActivePath(item.path);

              return (
                <button
                  key={item.id}
                  onClick={() => handleNavigation(item.path)}
                  className={`
                    w-full flex items-center justify-between px-3 py-3 rounded-lg text-left transition-colors duration-200
                    ${isActive 
                      ? 'bg-blue-50 text-blue-700 border border-blue-200' 
                      : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                    }
                  `}
                >
                  <div className="flex items-center space-x-3">
                    <span className="text-lg">{item.icon}</span>
                    <span className="font-medium">{item.label}</span>
                  </div>

                  {item.badge && (
                    <span className={`
                      px-2 py-1 text-xs font-medium rounded-full
                      ${item.badgeColor || 'bg-gray-100 text-gray-800'}
                    `}>
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>

          {/* Quick Actions */}
          <div className="p-4 border-t border-gray-200">
            <div className="space-y-3">
              <h3 className="text-sm font-medium text-gray-900">Quick Actions</h3>

              <button 
                onClick={() => handleNavigation('/discovery')}
                className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <span>📁</span>
                <span>Upload Data</span>
              </button>

              <button 
                onClick={() => handleNavigation('/discovery')}
                className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <span>🔗</span>
                <span>Connect Source</span>
              </button>

              <button 
                onClick={() => handleNavigation('/analysis')}
                className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <span>📈</span>
                <span>New Analysis</span>
              </button>

              <button 
                onClick={() => handleNavigation('/advanced')}
                className="w-full flex items-center space-x-3 px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
              >
                <span>🔮</span>
                <span>Train ML Model</span>
              </button>
            </div>
          </div>

          {/* System Status */}
          <div className="p-4 bg-gray-50">
            <div className="space-y-2">
              <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide">System Status</h4>

              <div className="space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Backend</span>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-green-600 font-medium">Online</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Smart Defaults</span>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-green-600 font-medium">Active</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-600">Data Sources</span>
                  <span className="text-gray-700 font-medium">3 Connected</span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-600">AI Engine</span>
                  <div className="flex items-center space-x-1">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-green-600 font-medium">Ready</span>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-600">ML Models</span>
                  <span className="text-purple-700 font-medium">2 Trained</span>
                </div>
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-200">
            <div className="flex items-center justify-between text-xs text-gray-500">
              <span>© 2024 Analytics Platform</span>
              <button className="text-blue-600 hover:text-blue-800">
                Help
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;