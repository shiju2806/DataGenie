import React, { useState, useEffect } from 'react';

interface Report {
  id: string;
  title: string;
  description: string;
  type: 'executive' | 'operational' | 'analytical';
  status: 'draft' | 'generating' | 'completed';
  createdAt: string;
  size?: string;
  tags?: string[];
}

const Reports: React.FC = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [selectedType, setSelectedType] = useState<string>('all');
  const [isGenerating, setIsGenerating] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newReportType, setNewReportType] = useState<'executive' | 'operational' | 'analytical'>('executive');

  const reportTypes = [
    { id: 'all', name: 'All Reports', icon: '📋' },
    { id: 'executive', name: 'Executive Summary', icon: '👔' },
    { id: 'operational', name: 'Operational', icon: '⚙️' },
    { id: 'analytical', name: 'Analytical', icon: '📊' }
  ];

  const reportTemplates = [
    {
      type: 'executive' as const,
      title: 'Executive Summary Report',
      description: 'High-level insights and KPIs for leadership team',
      icon: '👔',
      estimatedTime: '5-10 minutes',
      sections: ['Executive Overview', 'Key Metrics', 'Strategic Insights', 'Recommendations']
    },
    {
      type: 'operational' as const,
      title: 'Operational Performance Report',
      description: 'Detailed operational metrics and performance indicators',
      icon: '⚙️',
      estimatedTime: '10-15 minutes',
      sections: ['Operational Metrics', 'Performance Trends', 'Efficiency Analysis', 'Action Items']
    },
    {
      type: 'analytical' as const,
      title: 'Deep Analytics Report',
      description: 'Comprehensive data analysis with statistical insights',
      icon: '📊',
      estimatedTime: '15-25 minutes',
      sections: ['Data Analysis', 'Statistical Models', 'Correlations', 'Predictive Insights']
    }
  ];

  useEffect(() => {
    loadSavedReports();
  }, []);

  const loadSavedReports = () => {
    try {
      const saved = localStorage.getItem('analytics_reports');
      if (saved) {
        const savedReports = JSON.parse(saved);
        setReports(savedReports);
      } else {
        // Initialize with sample reports
        const sampleReports: Report[] = [
          {
            id: '1',
            title: 'Q4 Sales Performance Analysis',
            description: 'Comprehensive analysis of Q4 sales trends, performance metrics, and forecasts',
            type: 'executive',
            status: 'completed',
            createdAt: '2024-01-15',
            size: '2.3 MB',
            tags: ['sales', 'quarterly', 'performance']
          },
          {
            id: '2',
            title: 'Customer Segmentation Deep Dive',
            description: 'Detailed customer analysis with behavioral patterns and recommendations',
            type: 'analytical',
            status: 'completed',
            createdAt: '2024-01-10',
            size: '1.8 MB',
            tags: ['customers', 'segmentation', 'behavior']
          },
          {
            id: '3',
            title: 'Operational Efficiency Report',
            description: 'Monthly operational metrics and process optimization opportunities',
            type: 'operational',
            status: 'completed',
            createdAt: '2024-01-05',
            size: '1.2 MB',
            tags: ['operations', 'efficiency', 'monthly']
          }
        ];
        setReports(sampleReports);
        saveReports(sampleReports);
      }
    } catch (error) {
      console.error('Failed to load reports:', error);
    }
  };

  const saveReports = (reportsToSave: Report[]) => {
    try {
      localStorage.setItem('analytics_reports', JSON.stringify(reportsToSave));
    } catch (error) {
      console.error('Failed to save reports:', error);
    }
  };

  const generateNewReport = async (template: typeof reportTemplates[0]) => {
    setIsGenerating(true);
    setShowCreateModal(false);

    const newReport: Report = {
      id: Date.now().toString(),
      title: template.title,
      description: template.description,
      type: template.type,
      status: 'generating',
      createdAt: new Date().toISOString().split('T')[0],
      tags: ['auto-generated', 'latest']
    };

    const updatedReports = [newReport, ...reports];
    setReports(updatedReports);
    saveReports(updatedReports);

    // Simulate report generation
    setTimeout(() => {
      const completedReports = updatedReports.map(r =>
        r.id === newReport.id
          ? {
              ...r,
              status: 'completed' as const,
              size: `${(Math.random() * 3 + 0.5).toFixed(1)} MB`
            }
          : r
      );
      setReports(completedReports);
      saveReports(completedReports);
      setIsGenerating(false);
    }, 3000 + Math.random() * 2000);
  };

  const deleteReport = (reportId: string) => {
    const updatedReports = reports.filter(r => r.id !== reportId);
    setReports(updatedReports);
    saveReports(updatedReports);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100';
      case 'generating': return 'text-blue-600 bg-blue-100';
      case 'draft': return 'text-yellow-600 bg-yellow-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'executive': return '👔';
      case 'operational': return '⚙️';
      case 'analytical': return '📊';
      default: return '📋';
    }
  };

  const filteredReports = selectedType === 'all'
    ? reports
    : reports.filter(r => r.type === selectedType);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Comprehensive Reports</h1>
        <p className="text-gray-600">
          Generate and manage detailed analytics reports with AI-powered insights
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        <div className="lg:col-span-3">
          {/* Report Generation */}
          <div className="card mb-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Generate New Report</h2>
              <button
                onClick={() => setShowCreateModal(true)}
                disabled={isGenerating}
                className={`btn-primary flex items-center space-x-2 ${isGenerating ? 'opacity-50' : ''}`}
              >
                {isGenerating ? (
                  <>
                    <svg className="animate-spin w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                    <span>Generating...</span>
                  </>
                ) : (
                  <>
                    <span>📊</span>
                    <span>Create Report</span>
                  </>
                )}
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {reportTemplates.map((template, index) => (
                <button
                  key={index}
                  onClick={() => generateNewReport(template)}
                  disabled={isGenerating}
                  className={`p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors cursor-pointer text-left ${
                    isGenerating ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  <div className="text-center mb-3">
                    <div className="text-3xl mb-2">{template.icon}</div>
                    <h3 className="font-semibold text-gray-900">{template.type.charAt(0).toUpperCase() + template.type.slice(1)}</h3>
                    <p className="text-sm text-gray-600">{template.description}</p>
                  </div>
                  <div className="text-xs text-gray-500 text-center">
                    Est. time: {template.estimatedTime}
                  </div>
                </button>
              ))}
            </div>

            {isGenerating && (
              <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <div className="flex items-center space-x-3">
                  <svg className="animate-spin w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <div>
                    <p className="text-blue-900 font-medium">Generating Report...</p>
                    <p className="text-blue-700 text-sm">AI is analyzing your data and creating insights</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Reports List */}
          <div className="card">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-xl font-semibold text-gray-900">Your Reports</h2>
              <div className="flex space-x-2">
                {reportTypes.map((type) => (
                  <button
                    key={type.id}
                    onClick={() => setSelectedType(type.id)}
                    className={`px-3 py-1 text-sm rounded-full transition-colors ${
                      selectedType === type.id
                        ? 'bg-blue-100 text-blue-700'
                        : 'text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    {type.icon} {type.name}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-4">
              {filteredReports.length === 0 ? (
                <div className="text-center py-12">
                  <div className="text-4xl mb-4">📄</div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">No Reports Found</h3>
                  <p className="text-gray-600">Create your first report to get started with analytics insights.</p>
                </div>
              ) : (
                filteredReports.map((report) => (
                  <div key={report.id} className="p-4 border border-gray-200 rounded-lg hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between">
                      <div className="flex items-start space-x-4">
                        <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                          <span className="text-2xl">{getTypeIcon(report.type)}</span>
                        </div>

                        <div className="flex-1">
                          <h3 className="font-semibold text-gray-900 mb-1">{report.title}</h3>
                          <p className="text-gray-600 text-sm mb-2">{report.description}</p>

                          <div className="flex items-center space-x-4 text-sm text-gray-500">
                            <span>Created: {report.createdAt}</span>
                            {report.size && <span>Size: {report.size}</span>}
                            <span className="capitalize">{report.type}</span>
                          </div>

                          {report.tags && (
                            <div className="flex items-center space-x-2 mt-2">
                              {report.tags.map((tag, index) => (
                                <span
                                  key={index}
                                  className="px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center space-x-3">
                        <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(report.status)}`}>
                          {report.status === 'generating' && (
                            <svg className="inline animate-spin w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                            </svg>
                          )}
                          {report.status}
                        </span>

                        {report.status === 'completed' && (
                          <div className="flex space-x-2">
                            <button className="text-blue-600 hover:text-blue-800 text-sm font-medium">
                              📥 Download
                            </button>
                            <button className="text-gray-600 hover:text-gray-800 text-sm font-medium">
                              📤 Share
                            </button>
                            <button className="text-green-600 hover:text-green-800 text-sm font-medium">
                              👁️ View
                            </button>
                            <button
                              onClick={() => deleteReport(report.id)}
                              className="text-red-600 hover:text-red-800 text-sm font-medium"
                            >
                              🗑️ Delete
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Report Statistics</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Reports</span>
                <span className="font-semibold">{reports.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">This Month</span>
                <span className="font-semibold text-blue-600">
                  {reports.filter(r => {
                    const reportDate = new Date(r.createdAt);
                    const now = new Date();
                    return reportDate.getMonth() === now.getMonth() && reportDate.getFullYear() === now.getFullYear();
                  }).length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Completed</span>
                <span className="font-semibold text-green-600">
                  {reports.filter(r => r.status === 'completed').length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">In Progress</span>
                <span className="font-semibold text-yellow-600">
                  {reports.filter(r => r.status === 'generating').length}
                </span>
              </div>
            </div>
          </div>

          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              <button
                onClick={() => setShowCreateModal(true)}
                className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg"
              >
                📊 Custom Report
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                📅 Schedule Report
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                📧 Email Report
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                ⚙️ Report Settings
              </button>
              <button className="w-full text-left px-3 py-2 text-sm text-gray-700 hover:bg-gray-50 rounded-lg">
                📈 Analytics Trends
              </button>
            </div>
          </div>

          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Activity</h3>
            <div className="space-y-3 text-sm">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-gray-600">Report completed: Q4 Analysis</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span className="text-gray-600">New data source connected</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                <span className="text-gray-600">Report scheduled for tomorrow</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Create Report Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">Create New Report</h3>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="space-y-4">
              {reportTemplates.map((template, index) => (
                <button
                  key={index}
                  onClick={() => generateNewReport(template)}
                  className="w-full p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors text-left"
                >
                  <div className="flex items-start space-x-4">
                    <div className="text-3xl">{template.icon}</div>
                    <div className="flex-1">
                      <h4 className="font-semibold text-gray-900 mb-1">{template.title}</h4>
                      <p className="text-gray-600 text-sm mb-2">{template.description}</p>
                      <div className="text-xs text-gray-500">
                        <span>Estimated time: {template.estimatedTime}</span>
                        <span className="mx-2">•</span>
                        <span>Sections: {template.sections.join(', ')}</span>
                      </div>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Reports;