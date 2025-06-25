import React, { useState, useEffect } from 'react';
import { FileText, Download, Share, Calendar, Filter, Eye, TrendingUp, BarChart3, PieChart, RefreshCw, Plus, Settings, AlertCircle } from 'lucide-react';
import { analyticsAPI, extractErrorMessage } from '../../services/api';

interface Report {
  id: string;
  title: string;
  description: string;
  type: 'executive' | 'detailed' | 'technical' | 'custom';
  created_at: string;
  last_updated: string;
  author: string;
  status: 'draft' | 'published' | 'archived';
  tags: string[];
  metrics: {
    views: number;
    downloads: number;
    shares: number;
  };
  size_mb: number;
  format: 'pdf' | 'html' | 'excel' | 'powerpoint';
  thumbnail?: string;
}

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  type: 'executive' | 'detailed' | 'technical' | 'custom';
  sections: string[];
  estimated_time: string;
  complexity: 'simple' | 'moderate' | 'complex';
}

const Reports: React.FC = () => {
  const [reports, setReports] = useState<Report[]>([]);
  const [templates, setTemplates] = useState<ReportTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState<Set<string>>(new Set());
  const [filterType, setFilterType] = useState<'all' | 'executive' | 'detailed' | 'technical' | 'custom'>('all');
  const [filterStatus, setFilterStatus] = useState<'all' | 'draft' | 'published' | 'archived'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'created_at' | 'last_updated' | 'title' | 'views'>('last_updated');
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [systemCapabilities, setSystemCapabilities] = useState<any>(null);
  const [showTemplateModal, setShowTemplateModal] = useState(false);
  const [selectedTemplate, setSelectedTemplate] = useState<ReportTemplate | null>(null);

  useEffect(() => {
    initializeReports();
  }, []);

  const initializeReports = async () => {
    try {
      setLoading(true);
      setBackendStatus('checking');

      // Check backend status
      const health = await analyticsAPI.healthCheck();
      if (health.status === 'healthy') {
        setBackendStatus('connected');
      } else {
        setBackendStatus('disconnected');
      }

      // Load capabilities
      try {
        const capabilities = await analyticsAPI.getCapabilities();
        setSystemCapabilities(capabilities);
      } catch (error) {
        console.warn('Could not load capabilities:', error);
      }

      // Load reports and templates
      await loadReports();
      await loadTemplates();

    } catch (error) {
      console.error('Failed to initialize reports:', error);
      setBackendStatus('disconnected');
      setError('Failed to initialize reports. Using offline mode.');
      // Load mock data for offline functionality
      loadMockData();
    } finally {
      setLoading(false);
    }
  };

  const loadReports = async () => {
    try {
      // In a real implementation, this would fetch from the backend
      // For now, load from localStorage or use mock data
      const savedReports = localStorage.getItem('analytics_reports');
      if (savedReports) {
        setReports(JSON.parse(savedReports));
      } else {
        loadMockReports();
      }
    } catch (error) {
      console.error('Failed to load reports:', error);
      loadMockReports();
    }
  };

  const loadTemplates = async () => {
    try {
      // Load report templates
      const mockTemplates: ReportTemplate[] = [
        {
          id: 'executive_summary',
          name: 'Executive Summary',
          description: 'High-level overview with key metrics and insights for leadership',
          type: 'executive',
          sections: ['Executive Summary', 'Key Metrics', 'Strategic Insights', 'Recommendations'],
          estimated_time: '5-10 minutes',
          complexity: 'simple'
        },
        {
          id: 'detailed_analysis',
          name: 'Detailed Analytics Report',
          description: 'Comprehensive analysis with statistical insights and visualizations',
          type: 'detailed',
          sections: ['Data Overview', 'Statistical Analysis', 'Trend Analysis', 'Correlations', 'Predictions'],
          estimated_time: '15-20 minutes',
          complexity: 'moderate'
        },
        {
          id: 'technical_deep_dive',
          name: 'Technical Deep Dive',
          description: 'Technical report with methodology, algorithms, and detailed findings',
          type: 'technical',
          sections: ['Methodology', 'Data Processing', 'Algorithm Analysis', 'Technical Findings', 'Appendices'],
          estimated_time: '20-30 minutes',
          complexity: 'complex'
        },
        {
          id: 'custom_dashboard',
          name: 'Custom Dashboard Report',
          description: 'Customizable report with user-defined sections and metrics',
          type: 'custom',
          sections: ['Custom Section 1', 'Custom Section 2', 'User-defined Metrics'],
          estimated_time: '10-15 minutes',
          complexity: 'moderate'
        }
      ];
      setTemplates(mockTemplates);
    } catch (error) {
      console.error('Failed to load templates:', error);
    }
  };

  const loadMockReports = () => {
    const mockReports: Report[] = [
      {
        id: '1',
        title: 'Q4 2024 Sales Performance Analysis',
        description: 'Comprehensive analysis of Q4 sales data with regional breakdowns and trend analysis',
        type: 'executive',
        created_at: '2024-01-15T10:30:00Z',
        last_updated: '2024-01-16T14:22:00Z',
        author: 'AI Analytics Engine',
        status: 'published',
        tags: ['sales', 'q4', 'performance', 'regional'],
        metrics: { views: 156, downloads: 42, shares: 8 },
        size_mb: 2.3,
        format: 'pdf'
      },
      {
        id: '2',
        title: 'Customer Behavior Insights Report',
        description: 'Deep dive into customer behavior patterns and segmentation analysis',
        type: 'detailed',
        created_at: '2024-01-14T09:15:00Z',
        last_updated: '2024-01-14T16:45:00Z',
        author: 'DataGenie Smart Engine',
        status: 'published',
        tags: ['customer', 'behavior', 'segmentation', 'patterns'],
        metrics: { views: 89, downloads: 23, shares: 5 },
        size_mb: 4.7,
        format: 'html'
      },
      {
        id: '3',
        title: 'Predictive Model Performance Evaluation',
        description: 'Technical analysis of ML model performance with accuracy metrics and optimization recommendations',
        type: 'technical',
        created_at: '2024-01-13T11:20:00Z',
        last_updated: '2024-01-15T13:10:00Z',
        author: 'Mathematical Engine',
        status: 'draft',
        tags: ['machine learning', 'prediction', 'model', 'performance'],
        metrics: { views: 34, downloads: 12, shares: 2 },
        size_mb: 1.9,
        format: 'pdf'
      },
      {
        id: '4',
        title: 'Market Trend Analysis Dashboard',
        description: 'Real-time market trends with competitive analysis and opportunity identification',
        type: 'custom',
        created_at: '2024-01-12T08:30:00Z',
        last_updated: '2024-01-16T12:15:00Z',
        author: 'Smart Defaults Engine',
        status: 'published',
        tags: ['market', 'trends', 'competitive', 'dashboard'],
        metrics: { views: 203, downloads: 67, shares: 15 },
        size_mb: 3.1,
        format: 'excel'
      }
    ];
    setReports(mockReports);
  };

  const loadMockData = () => {
    loadMockReports();
    // Templates are already loaded in loadTemplates
  };

  const generateReport = async (template: ReportTemplate) => {
    const newSet = new Set(generating);
    newSet.add(template.id);
    setGenerating(newSet);
    setError(null);

    try {
      console.log(`📊 Generating ${template.name} report...`);

      // Simulate report generation
      await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 3000));

      const newReport: Report = {
        id: Date.now().toString(),
        title: `${template.name} - ${new Date().toLocaleDateString()}`,
        description: template.description,
        type: template.type,
        created_at: new Date().toISOString(),
        last_updated: new Date().toISOString(),
        author: 'AI Analytics Engine',
        status: 'draft',
        tags: ['auto-generated', template.type, 'new'],
        metrics: { views: 0, downloads: 0, shares: 0 },
        size_mb: Math.random() * 5 + 1,
        format: 'pdf'
      };

      setReports(prev => [newReport, ...prev]);

      // Save to localStorage
      const updatedReports = [newReport, ...reports];
      localStorage.setItem('analytics_reports', JSON.stringify(updatedReports));

      console.log(`✅ Report generated: ${newReport.title}`);
      setShowTemplateModal(false);

    } catch (error) {
      console.error('Report generation failed:', error);
      setError(`Failed to generate ${template.name}: ${extractErrorMessage(error)}`);
    } finally {
      const finalSet = new Set(generating);
      finalSet.delete(template.id);
      setGenerating(finalSet);
    }
  };

  const downloadReport = async (report: Report) => {
    try {
      console.log(`📥 Downloading ${report.title}...`);

      // In a real implementation, this would download from the backend
      // For demo, create a mock download
      const content = `
Analytics Report: ${report.title}
Generated: ${new Date(report.created_at).toLocaleString()}
Author: ${report.author}
Type: ${report.type}

Description:
${report.description}

This is a mock report generated by the Enhanced Analytics Platform.
In a production environment, this would contain the actual report content
with charts, insights, and detailed analysis.

Tags: ${report.tags.join(', ')}
Status: ${report.status}
Format: ${report.format.toUpperCase()}
      `;

      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${report.title.replace(/[^a-z0-9]/gi, '_')}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      // Update download count
      setReports(prev => prev.map(r =>
        r.id === report.id
          ? { ...r, metrics: { ...r.metrics, downloads: r.metrics.downloads + 1 } }
          : r
      ));

    } catch (error) {
      console.error('Download failed:', error);
      setError(`Failed to download ${report.title}: ${extractErrorMessage(error)}`);
    }
  };

  const shareReport = async (report: Report) => {
    try {
      const shareData = {
        title: report.title,
        text: report.description,
        url: `${window.location.origin}/reports/${report.id}`
      };

      if (navigator.share) {
        await navigator.share(shareData);
      } else {
        // Fallback: copy to clipboard
        await navigator.clipboard.writeText(
          `${report.title}\n${report.description}\n${shareData.url}`
        );
        alert('Report link copied to clipboard!');
      }

      // Update share count
      setReports(prev => prev.map(r =>
        r.id === report.id
          ? { ...r, metrics: { ...r.metrics, shares: r.metrics.shares + 1 } }
          : r
      ));

    } catch (error) {
      console.error('Share failed:', error);
      setError(`Failed to share ${report.title}: ${extractErrorMessage(error)}`);
    }
  };

  const viewReport = (report: Report) => {
    // Update view count
    setReports(prev => prev.map(r =>
      r.id === report.id
        ? { ...r, metrics: { ...r.metrics, views: r.metrics.views + 1 } }
        : r
    ));

    // In a real implementation, this would navigate to the report view
    console.log(`👁️ Viewing report: ${report.title}`);
    alert(`Viewing ${report.title}\n\nIn a full implementation, this would open the detailed report view.`);
  };

  const filteredReports = reports
    .filter(report => {
      if (filterType !== 'all' && report.type !== filterType) return false;
      if (filterStatus !== 'all' && report.status !== filterStatus) return false;
      if (searchQuery && !report.title.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !report.description.toLowerCase().includes(searchQuery.toLowerCase()) &&
          !report.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()))) {
        return false;
      }
      return true;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'title':
          return a.title.localeCompare(b.title);
        case 'created_at':
          return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
        case 'views':
          return b.metrics.views - a.metrics.views;
        case 'last_updated':
        default:
          return new Date(b.last_updated).getTime() - new Date(a.last_updated).getTime();
      }
    });

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'published': return 'text-green-600 bg-green-100';
      case 'draft': return 'text-yellow-600 bg-yellow-100';
      case 'archived': return 'text-gray-600 bg-gray-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'executive': return '👔';
      case 'detailed': return '📊';
      case 'technical': return '🔧';
      case 'custom': return '🎨';
      default: return '📄';
    }
  };

  const getFormatIcon = (format: string) => {
    switch (format) {
      case 'pdf': return '📄';
      case 'html': return '🌐';
      case 'excel': return '📊';
      case 'powerpoint': return '📑';
      default: return '📄';
    }
  };

  const getBackendStatusIndicator = () => {
    switch (backendStatus) {
      case 'checking':
        return { color: 'bg-yellow-500', text: 'Checking...', pulse: true };
      case 'connected':
        return { color: 'bg-green-500', text: 'Reports Ready', pulse: false };
      case 'disconnected':
        return { color: 'bg-red-500', text: 'Offline Mode', pulse: false };
    }
  };

  const statusInfo = getBackendStatusIndicator();

  if (loading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[1, 2, 3, 4, 5, 6].map(i => (
              <div key={i} className="h-64 bg-gray-200 rounded-lg"></div>
            ))}
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
          <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center">
            <FileText className="w-8 h-8 text-blue-500 mr-3" />
            Analytics Reports
          </h1>
          <p className="text-gray-600">
            Generate, manage, and share comprehensive analytics reports with AI-powered insights
          </p>
        </div>

        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${statusInfo.color} ${statusInfo.pulse ? 'animate-pulse' : ''}`} />
            <span className="text-gray-600">{statusInfo.text}</span>
          </div>

          {systemCapabilities?.features?.comprehensive_reporting && (
            <div className="text-xs text-gray-500">
              <span className="text-purple-600">📋 Smart Reports</span>
            </div>
          )}

          <button
            onClick={() => setShowTemplateModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors inline-flex items-center space-x-2 font-medium"
          >
            <Plus className="w-4 h-4" />
            <span>Generate Report</span>
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-start space-x-3">
            <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
            <div className="flex-1">
              <h3 className="text-sm font-medium text-red-800">Report Error</h3>
              <p className="mt-1 text-sm text-red-700">{error}</p>
            </div>
            <button
              onClick={() => setError(null)}
              className="flex-shrink-0 text-red-400 hover:text-red-600"
            >
              ✕
            </button>
          </div>
        </div>
      )}

      {/* Filters and Search */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Search</label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search reports..."
              className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Type</label>
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">All Types</option>
              <option value="executive">Executive</option>
              <option value="detailed">Detailed</option>
              <option value="technical">Technical</option>
              <option value="custom">Custom</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Status</label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value as any)}
              className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="all">All Status</option>
              <option value="published">Published</option>
              <option value="draft">Draft</option>
              <option value="archived">Archived</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="last_updated">Last Updated</option>
              <option value="created_at">Created Date</option>
              <option value="title">Title</option>
              <option value="views">Most Viewed</option>
            </select>
          </div>
        </div>
      </div>

      {/* Reports Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {filteredReports.map((report) => (
          <div key={report.id} className="bg-white rounded-lg shadow-sm border border-gray-200 hover:shadow-lg transition-shadow">
            <div className="p-6">
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="text-2xl">{getTypeIcon(report.type)}</div>
                  <div>
                    <h3 className="font-semibold text-gray-900 line-clamp-2">{report.title}</h3>
                    <p className="text-sm text-gray-600">{report.author}</p>
                  </div>
                </div>
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(report.status)}`}>
                  {report.status}
                </span>
              </div>

              <p className="text-sm text-gray-700 mb-4 line-clamp-3">{report.description}</p>

              <div className="flex flex-wrap gap-1 mb-4">
                {report.tags.slice(0, 3).map((tag) => (
                  <span key={tag} className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full">
                    {tag}
                  </span>
                ))}
                {report.tags.length > 3 && (
                  <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs rounded-full">
                    +{report.tags.length - 3}
                  </span>
                )}
              </div>

              <div className="grid grid-cols-3 gap-4 mb-4 text-center text-sm">
                <div>
                  <div className="font-semibold text-gray-900">{report.metrics.views}</div>
                  <div className="text-gray-600">Views</div>
                </div>
                <div>
                  <div className="font-semibold text-gray-900">{report.metrics.downloads}</div>
                  <div className="text-gray-600">Downloads</div>
                </div>
                <div>
                  <div className="font-semibold text-gray-900">{report.metrics.shares}</div>
                  <div className="text-gray-600">Shares</div>
                </div>
              </div>

              <div className="flex items-center justify-between text-xs text-gray-500 mb-4">
                <span>Updated: {new Date(report.last_updated).toLocaleDateString()}</span>
                <div className="flex items-center space-x-1">
                  <span>{getFormatIcon(report.format)}</span>
                  <span>{report.size_mb.toFixed(1)} MB</span>
                </div>
              </div>

              <div className="flex space-x-2">
                <button
                  onClick={() => viewReport(report)}
                  className="flex-1 bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-lg transition-colors inline-flex items-center justify-center space-x-1 text-sm"
                >
                  <Eye className="w-4 h-4" />
                  <span>View</span>
                </button>

                <button
                  onClick={() => downloadReport(report)}
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg transition-colors"
                  title="Download"
                >
                  <Download className="w-4 h-4" />
                </button>

                <button
                  onClick={() => shareReport(report)}
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-2 rounded-lg transition-colors"
                  title="Share"
                >
                  <Share className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Empty State */}
      {filteredReports.length === 0 && !loading && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No Reports Found</h3>
          <p className="text-gray-600 mb-6">
            {searchQuery || filterType !== 'all' || filterStatus !== 'all'
              ? 'No reports match your current filters. Try adjusting your search criteria.'
              : 'Get started by generating your first analytics report using our smart templates.'
            }
          </p>
          <button
            onClick={() => setShowTemplateModal(true)}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg transition-colors inline-flex items-center space-x-2 font-medium"
          >
            <Plus className="w-5 h-5" />
            <span>Generate First Report</span>
          </button>
        </div>
      )}

      {/* Template Selection Modal */}
      {showTemplateModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold text-gray-900">Select Report Template</h2>
                <button
                  onClick={() => setShowTemplateModal(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ✕
                </button>
              </div>
              <p className="text-gray-600 mt-2">
                Choose a template to generate a comprehensive analytics report
              </p>
            </div>

            <div className="p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {templates.map((template) => (
                  <div
                    key={template.id}
                    className={`border-2 rounded-xl p-6 cursor-pointer transition-all hover:shadow-md ${
                      selectedTemplate?.id === template.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                    onClick={() => setSelectedTemplate(template)}
                  >
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div className="text-3xl">{getTypeIcon(template.type)}</div>
                        <div>
                          <h3 className="font-semibold text-gray-900">{template.name}</h3>
                          <p className="text-sm text-gray-600 capitalize">{template.type} Report</p>
                        </div>
                      </div>
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                        template.complexity === 'simple' ? 'text-green-600 bg-green-100' :
                        template.complexity === 'moderate' ? 'text-yellow-600 bg-yellow-100' :
                        'text-red-600 bg-red-100'
                      }`}>
                        {template.complexity}
                      </span>
                    </div>

                    <p className="text-sm text-gray-700 mb-4">{template.description}</p>

                    <div className="space-y-2 mb-4">
                      <h4 className="text-sm font-medium text-gray-900">Sections:</h4>
                      <ul className="text-sm text-gray-600 space-y-1">
                        {template.sections.map((section, index) => (
                          <li key={index} className="flex items-center space-x-2">
                            <span className="w-1.5 h-1.5 bg-blue-500 rounded-full"></span>
                            <span>{section}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="flex items-center justify-between text-sm text-gray-600">
                      <span>📅 {template.estimated_time}</span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          generateReport(template);
                        }}
                        disabled={generating.has(template.id)}
                        className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white px-4 py-2 rounded-lg transition-colors inline-flex items-center space-x-2"
                      >
                        {generating.has(template.id) ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                            <span>Generating...</span>
                          </>
                        ) : (
                          <>
                            <Plus className="w-4 h-4" />
                            <span>Generate</span>
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Reports;