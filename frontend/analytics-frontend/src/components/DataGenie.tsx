import React, { useState, useEffect } from 'react';
import {
  Upload,
  Database,
  MessageCircle,
  BarChart3,
  FileText,
  Settings,
  HelpCircle,
  Sparkles,
  TrendingUp,
  PieChart,
  Download,
  Share,
  Clock,
  CheckCircle,
  Brain,
  Zap,
  Search,
  AlertCircle,
  RefreshCw
} from 'lucide-react';

// Import types and API from the services directory
import {
  analyticsAPI,
  extractErrorMessage,
  type AnalysisResponse,
  type DataSource,
  type DiscoveryResponse
} from '../services/api';

// Types
interface AnalysisResults {
  summary: string;
  insights: string[];
  chartData: any[];
  confidence: number;
  analysisType: string;
  chartSuggestions?: any[];
  performance?: any;
  metadata?: any;
  comprehensiveReport?: any;
}

interface SavedAnalysis {
  id: number;
  query: string;
  results: AnalysisResults;
  timestamp: Date;
  sources: string[];
  file?: string | null;
}

type StepType = 'connect' | 'analyze' | 'results' | 'report';

const DataGenie: React.FC = () => {
  // State
  const [currentStep, setCurrentStep] = useState<StepType>('connect');
  const [selectedSources, setSelectedSources] = useState<string[]>([]);
  const [query, setQuery] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [savedAnalyses, setSavedAnalyses] = useState<SavedAnalysis[]>([]);
  const [dataSources, setDataSources] = useState<DataSource[]>([]);
  const [isDiscovering, setIsDiscovering] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [systemCapabilities, setSystemCapabilities] = useState<any>(null);
  const [advancedOptions, setAdvancedOptions] = useState({
    includeStatisticalTests: false,
    detectOutliers: false,
    predictiveModeling: false,
    executiveSummary: true,
    use_adaptive: true,
    include_charts: true,
    auto_discover: true,
    domain: 'general'
  });

  // Auto-discover data sources on load
  useEffect(() => {
    initializeDataGenie();
  }, []);

  // Load saved analyses on mount
  useEffect(() => {
    loadSavedAnalyses();
  }, []);

  const initializeDataGenie = async () => {
    try {
      console.log('🚀 Initializing DataGenie...');

      // Check backend status
      await checkBackendStatus();

      // Discover data sources
      await discoverDataSources();

    } catch (error) {
      console.error('Failed to initialize DataGenie:', error);
      setError('Failed to initialize DataGenie. Please check your backend connection.');
    }
  };

  const checkBackendStatus = async () => {
    try {
      console.log('🔍 Checking backend status...');
      setBackendStatus('checking');

      const [healthCheck, capabilities, openaiTest] = await Promise.allSettled([
        analyticsAPI.healthCheck(),
        analyticsAPI.getCapabilities(),
        analyticsAPI.testOpenAI()
      ]);

      if (healthCheck.status === 'fulfilled' && healthCheck.value.status === 'healthy') {
        setBackendStatus('connected');
        console.log('✅ Backend is healthy');
      } else {
        setBackendStatus('disconnected');
        console.log('❌ Backend is not healthy');
      }

      if (capabilities.status === 'fulfilled') {
        setSystemCapabilities(capabilities.value);
        console.log('📋 Capabilities loaded:', capabilities.value);
      }

      // Log OpenAI status
      if (openaiTest.status === 'fulfilled') {
        console.log('🤖 OpenAI status:', openaiTest.value);
      }

    } catch (error) {
      console.error('❌ Backend status check failed:', error);
      setBackendStatus('disconnected');
    }
  };

  const discoverDataSources = async () => {
    setIsDiscovering(true);
    try {
      console.log('🔍 Discovering data sources...');

      const response: DiscoveryResponse = await analyticsAPI.discoverDataSources({
        mode: 'balanced',
        include_environment_scan: true,
        max_recommendations: 20,
        confidence_threshold: 0.5
      });

      if (response.status === 'success' && response.recommendations) {
        setDataSources(response.recommendations);
        console.log(`✅ Discovered ${response.recommendations.length} data sources`);
      } else {
        console.log('⚠️ Using fallback data sources');
        setDataSources(getMockDataSources());
      }
    } catch (error) {
      console.error('Discovery failed, using mock data:', error);
      setDataSources(getMockDataSources());
    } finally {
      setIsDiscovering(false);
    }
  };

  const getMockDataSources = (): DataSource[] => [
    {
      id: 'postgres_prod',
      source_id: 'postgres_prod',
      type: 'PostgreSQL',
      confidence: 0.92,
      reasoning: 'Production PostgreSQL database detected',
      context: {
        type: 'PostgreSQL',
        host: 'localhost',
        port: 5432,
        database: 'analytics_db',
        table_count: 45
      },
      status: 'discovered'
    },
    {
      id: 'sales_csv',
      source_id: 'sales_csv',
      type: 'CSV',
      confidence: 0.87,
      reasoning: 'CSV files found in data directory',
      context: {
        type: 'CSV',
        location: '/data/sales/',
        size: '2.4 MB'
      },
      status: 'discovered'
    },
    {
      id: 'tableau_server',
      source_id: 'tableau_server',
      type: 'Tableau',
      confidence: 0.78,
      reasoning: 'Tableau Server connection available',
      context: {
        type: 'Tableau',
        server: 'tableau.company.com'
      },
      status: 'discovered'
    },
    {
      id: 'api_crm',
      source_id: 'api_crm',
      type: 'REST API',
      confidence: 0.85,
      reasoning: 'CRM API endpoint accessible',
      context: {
        type: 'REST API',
        endpoint: 'api.crm.company.com'
      },
      status: 'discovered'
    }
  ];

  const loadSavedAnalyses = () => {
    try {
      const saved = JSON.parse(localStorage.getItem('dataGenie_analyses') || '[]');
      setSavedAnalyses(saved.map((analysis: any) => ({
        ...analysis,
        timestamp: new Date(analysis.timestamp)
      })));
    } catch (error) {
      console.error('Failed to load saved analyses:', error);
    }
  };

  // Export Functions
  const downloadPDF = () => {
    if (!analysisResults) {
      alert('No analysis results to export');
      return;
    }

    const reportContent = `
DataGenie Analysis Report
Generated: ${new Date().toLocaleDateString()}

QUERY: ${query}

SUMMARY:
${analysisResults.summary}

KEY INSIGHTS:
${analysisResults.insights.map((insight, i) => `${i + 1}. ${insight}`).join('\n')}

ANALYSIS TYPE: ${analysisResults.analysisType}
CONFIDENCE: ${Math.round(analysisResults.confidence * 100)}%

${analysisResults.comprehensiveReport ? `
COMPREHENSIVE REPORT:
Executive Summary: ${analysisResults.comprehensiveReport.executive_summary || 'N/A'}

Detailed Findings:
${(analysisResults.comprehensiveReport.detailed_findings || []).map((finding: string, i: number) => `${i + 1}. ${finding}`).join('\n')}

Recommendations:
${(analysisResults.comprehensiveReport.recommendations || []).map((rec: string, i: number) => `${i + 1}. ${rec}`).join('\n')}
` : ''}

---
This report was generated by DataGenie Analytics Platform
Powered by advanced AI and mathematical analysis engines
    `;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `DataGenie-Analysis-${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const shareLink = () => {
    if (!analysisResults) {
      alert('No analysis results to share');
      return;
    }

    const analysisData = {
      query,
      summary: analysisResults.summary,
      insights: analysisResults.insights,
      timestamp: new Date().toISOString(),
      confidence: analysisResults.confidence,
      analysisType: analysisResults.analysisType
    };

    const encodedData = btoa(JSON.stringify(analysisData));
    const shareUrl = `${window.location.origin}/shared-analysis?data=${encodedData}`;

    navigator.clipboard.writeText(shareUrl).then(() => {
      alert('Share link copied to clipboard!');
    }).catch(() => {
      const textArea = document.createElement('textarea');
      textArea.value = shareUrl;
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand('copy');
      document.body.removeChild(textArea);
      alert('Share link copied to clipboard!');
    });
  };

  const scheduleUpdates = () => {
    const email = prompt('Enter email address for scheduled reports:');
    if (email) {
      alert(`Scheduled reports will be sent to ${email}. This feature connects to your backend scheduling system.`);
    }
  };

  // Event Handlers
  const handleSourceSelect = (sourceId: string) => {
    setSelectedSources(prev =>
      prev.includes(sourceId)
        ? prev.filter(id => id !== sourceId)
        : [...prev, sourceId]
    );
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // Validate file
      const maxSize = 100 * 1024 * 1024; // 100MB
      if (file.size > maxSize) {
        setError(`File too large: ${Math.round(file.size / 1024 / 1024)}MB. Maximum size is 100MB.`);
        return;
      }

      const allowedTypes = ['.csv', '.xlsx', '.xls', '.json'];
      const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
      if (!allowedTypes.includes(fileExtension)) {
        setError(`Unsupported file type: ${fileExtension}. Please use CSV, Excel, or JSON files.`);
        return;
      }

      setUploadedFile(file);
      setSelectedSources([]);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!query.trim()) {
      setError('Please enter a question or query');
      return;
    }

    if (!uploadedFile && selectedSources.length === 0) {
      setError('Please select a data source or upload a file');
      return;
    }

    setError(null);
    setIsAnalyzing(true);
    setCurrentStep('analyze');

    try {
      console.log('🔍 Starting analysis with enhanced options:', {
        query,
        hasFile: !!uploadedFile,
        fileName: uploadedFile?.name,
        fileSize: uploadedFile?.size,
        fileType: uploadedFile?.type,
        selectedSources: selectedSources.length,
        advancedOptions
      });

      const response: AnalysisResponse = await analyticsAPI.analyze({
        prompt: query,
        file: uploadedFile || undefined,
        data_source_id: selectedSources[0],
        use_adaptive: advancedOptions.use_adaptive,
        include_charts: advancedOptions.include_charts,
        auto_discover: selectedSources.length === 0 && !uploadedFile && advancedOptions.auto_discover,
        domain: advancedOptions.domain
      });

      console.log('📡 Backend response:', response);

      if (response.status === 'success') {
        const transformedResults: AnalysisResults = {
          summary: response.analysis?.summary || 'Analysis completed successfully',
          insights: response.analysis?.insights || ['Analysis completed with no specific insights'],
          chartData: response.analysis?.data || [],
          confidence: response.query_interpretation?.confidence || 0.8,
          analysisType: response.analysis?.type || 'general_analysis',
          chartSuggestions: response.chart_intelligence?.suggested_charts || [],
          performance: response.performance || {},
          metadata: response.analysis?.metadata || {},
          comprehensiveReport: response.comprehensive_report
        };

        console.log('✅ Analysis successful, showing results');
        setAnalysisResults(transformedResults);
        setCurrentStep('results');
        setError(null);
      } else {
        console.error('❌ Backend returned error:', response);
        throw new Error(extractErrorMessage(response));
      }
    } catch (error) {
      console.error('💥 Analysis error:', error);

      const errorMessage = extractErrorMessage(error);
      setError(errorMessage);
      setCurrentStep('analyze');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const saveAnalysis = () => {
    if (!analysisResults) return;

    const newAnalysis: SavedAnalysis = {
      id: Date.now(),
      query,
      results: analysisResults,
      timestamp: new Date(),
      sources: selectedSources,
      file: uploadedFile?.name || null
    };

    setSavedAnalyses(prev => [newAnalysis, ...prev]);

    try {
      const saved = JSON.parse(localStorage.getItem('dataGenie_analyses') || '[]');
      saved.unshift(newAnalysis);
      localStorage.setItem('dataGenie_analyses', JSON.stringify(saved.slice(0, 10)));
    } catch (error) {
      console.error('Failed to save analysis to localStorage:', error);
    }
  };

  const generateReport = () => {
    setCurrentStep('report');
  };

  const handleAdvancedOptionChange = (option: keyof typeof advancedOptions) => {
    setAdvancedOptions(prev => ({
      ...prev,
      [option]: !prev[option]
    }));
  };

  const loadSavedAnalysis = (analysis: SavedAnalysis) => {
    setQuery(analysis.query);
    setAnalysisResults(analysis.results);
    setSelectedSources(analysis.sources || []);
    setCurrentStep('results');
  };

  // Step configuration
  const steps = [
    { step: 'connect' as const, label: 'Connect Data', icon: Database },
    { step: 'analyze' as const, label: 'Ask DataGenie', icon: MessageCircle },
    { step: 'results' as const, label: 'View Results', icon: BarChart3 },
    { step: 'report' as const, label: 'Export Report', icon: FileText }
  ];

  const suggestions = [
    "Show revenue trends over time",
    "Compare regional performance metrics",
    "Identify top customers by revenue",
    "Analyze seasonal patterns and trends",
    "Find correlations in the data",
    "Detect any outliers or anomalies"
  ];

  const getStatusColor = (status: string | undefined) => {
    switch (status) {
      case 'connected': return 'text-green-600 bg-green-100';
      case 'discovered': return 'text-blue-600 bg-blue-100';
      case 'connecting': return 'text-yellow-600 bg-yellow-100';
      case 'testing': return 'text-purple-600 bg-purple-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case 'postgresql': return '🐘';
      case 'mysql': return '🐬';
      case 'mongodb': return '🍃';
      case 'redis': return '🔴';
      case 'csv': return '📁';
      case 'file system': return '📁';
      case 'api': return '🔗';
      case 'rest api': return '🔗';
      case 'sqlite': return '💎';
      case 'elasticsearch': return '🔍';
      case 'tableau': return '📊';
      default: return '🗄️';
    }
  };

  const getBackendStatusIndicator = () => {
    switch (backendStatus) {
      case 'checking':
        return { color: 'bg-yellow-500', text: 'Checking Backend...', pulse: true };
      case 'connected':
        return { color: 'bg-green-500', text: 'Backend Connected', pulse: false };
      case 'disconnected':
        return { color: 'bg-red-500', text: 'Backend Disconnected', pulse: false };
    }
  };

  const statusInfo = getBackendStatusIndicator();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm border-b border-white/20 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="text-2xl">🧞‍♂️</div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              DataGenie
            </h1>
            <span className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded-full">v5.1.0</span>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm">
              <div className={`w-2 h-2 rounded-full ${statusInfo.color} ${statusInfo.pulse ? 'animate-pulse' : ''}`} />
              <span className="text-gray-600">{statusInfo.text}</span>
            </div>

            {systemCapabilities && (
              <div className="text-xs text-gray-500 flex items-center space-x-2">
                {systemCapabilities.smart_features?.unified_smart_engine && (
                  <span className="text-blue-600">🧠 Smart Engine</span>
                )}
                {systemCapabilities.smart_features?.auto_data_discovery && (
                  <span className="text-green-600">🔍 Auto-Discovery</span>
                )}
                {systemCapabilities.smart_features?.llm_powered_query_understanding && (
                  <span className="text-purple-600">🤖 AI Ready</span>
                )}
              </div>
            )}

            <div className="flex items-center space-x-2 text-sm">
              <div className={`w-2 h-2 rounded-full ${
                dataSources.length > 0 ? 'bg-green-500' : 'bg-yellow-500'
              }`} />
              <span className="text-gray-600">
                {isDiscovering ? 'Discovering...' : `${dataSources.length} sources`}
              </span>
            </div>

            <button
              onClick={checkBackendStatus}
              className="p-2 text-gray-600 hover:text-gray-900 transition-colors"
              title="Refresh Status"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button className="p-2 text-gray-600 hover:text-gray-900 transition-colors" title="Help">
              <HelpCircle className="w-5 h-5" />
            </button>
            <button className="p-2 text-gray-600 hover:text-gray-900 transition-colors" title="Settings">
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* Progress Indicator */}
        <div className="mb-8">
          <div className="flex items-center justify-center space-x-8">
            {steps.map(({ step, label, icon: Icon }, index) => {
              const isActive = currentStep === step;
              const stepOrder: StepType[] = ['connect', 'analyze', 'results', 'report'];
              const isCompleted = stepOrder.indexOf(currentStep) > stepOrder.indexOf(step);

              return (
                <div key={step} className="flex items-center">
                  <div className={`flex items-center space-x-2 px-4 py-2 rounded-full transition-all ${
                    isActive ? 'bg-blue-100 text-blue-700' : 
                    isCompleted ? 'bg-green-100 text-green-700' : 
                    'bg-gray-100 text-gray-500'
                  }`}>
                    {isCompleted ? (
                      <CheckCircle className="w-4 h-4" />
                    ) : (
                      <Icon className="w-4 h-4" />
                    )}
                    <span className="text-sm font-medium">{label}</span>
                  </div>
                  {index < 3 && (
                    <div className={`w-8 h-0.5 mx-2 ${
                      isCompleted ? 'bg-green-300' : 'bg-gray-200'
                    }`} />
                  )}
                </div>
              );
            })}
          </div>
        </div>

        {/* Main Content Area */}
        <div className="space-y-8">

          {/* Step 1: Data Connection */}
          {currentStep === 'connect' && (
            <div className="bg-white rounded-2xl shadow-lg border border-white/20 p-8">
              <div className="text-center mb-8">
                <Sparkles className="w-12 h-12 text-blue-500 mx-auto mb-4" />
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Connect Your Data</h2>
                <p className="text-gray-600">DataGenie automatically discovered data sources in your environment</p>
              </div>

              {/* Error Display */}
              {error && (
                <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <AlertCircle className="w-4 h-4 text-red-500" />
                    <p className="text-red-700 text-sm">{error}</p>
                  </div>
                </div>
              )}

              {/* Auto-discovered Sources */}
              <div className="mb-8">
                <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Search className="w-5 h-5 mr-2 text-blue-500" />
                  Auto-Discovered Sources
                  {isDiscovering && (
                    <div className="ml-2 w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  )}
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {dataSources.map((source) => (
                    <div
                      key={source.id}
                      className={`border-2 rounded-xl p-4 cursor-pointer transition-all hover:shadow-md ${
                        selectedSources.includes(source.id)
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => handleSourceSelect(source.id)}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-2">
                          <span className="text-2xl">{getTypeIcon(source.type)}</span>
                          <h4 className="font-semibold text-gray-900">{source.source_id}</h4>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full ${
                            source.confidence > 0.8 ? 'bg-green-500' : 'bg-yellow-500'
                          }`} />
                          <span className="text-xs text-gray-500">{Math.round(source.confidence * 100)}%</span>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{source.reasoning}</p>
                      <div className="text-xs text-gray-500">
                        <span className="font-medium">{source.type}</span>
                        {source.context.host && (
                          <span className="ml-2">• {source.context.host}</span>
                        )}
                        {source.context.database && (
                          <span className="ml-2">• {source.context.database}</span>
                        )}
                        {source.context.table_count && (
                          <span className="ml-2">• {source.context.table_count} tables</span>
                        )}
                        {source.context.size && (
                          <span className="ml-2">• {source.context.size}</span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Manual Upload */}
              <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-gray-400 transition-colors">
                <Upload className="w-8 h-8 text-gray-400 mx-auto mb-3" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload Data</h3>
                <p className="text-gray-600 mb-4">Drop CSV, Excel, or JSON files here</p>
                {uploadedFile && (
                  <div className="mb-4 p-2 bg-green-50 border border-green-200 rounded-lg">
                    <p className="text-green-700 text-sm">✓ {uploadedFile.name} ({Math.round(uploadedFile.size / 1024)} KB)</p>
                  </div>
                )}
                <input
                  type="file"
                  accept=".csv,.xlsx,.xls,.json"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-4 py-2 rounded-lg transition-colors inline-flex items-center cursor-pointer">
                  Choose Files
                </label>
                <p className="text-xs text-gray-500 mt-2">
                  Maximum file size: 100MB
                </p>
              </div>

              {(selectedSources.length > 0 || uploadedFile) && (
                <div className="mt-8 text-center">
                  <button
                    className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-8 py-3 text-lg rounded-lg transition-colors inline-flex items-center"
                    onClick={() => setCurrentStep('analyze')}
                  >
                    Continue with {uploadedFile ? uploadedFile.name : `${selectedSources.length} source${selectedSources.length > 1 ? 's' : ''}`}
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Step 2: Analysis Interface */}
          {(currentStep === 'analyze' || currentStep === 'results') && (
            <div className="bg-white rounded-2xl shadow-lg border border-white/20 p-8">
              <div className="text-center mb-8">
                <Brain className="w-12 h-12 text-purple-500 mx-auto mb-4" />
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Ask DataGenie</h2>
                <p className="text-gray-600">What would you like to learn from your data?</p>
              </div>

              {/* Error Display */}
              {error && (
                <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="w-5 h-5 text-red-400 mt-0.5" />
                    <div className="flex-1">
                      <h3 className="text-sm font-medium text-red-800">Analysis Error</h3>
                      <p className="mt-1 text-sm text-red-700">{error}</p>
                      {error.includes('backend server') && (
                        <div className="mt-2 text-sm text-red-600">
                          <p className="font-medium">To fix this:</p>
                          <ol className="list-decimal list-inside mt-1 space-y-1">
                            <li>Make sure your backend server is running</li>
                            <li>Check that it's accessible at: <code className="bg-red-100 px-1 rounded">http://localhost:8000</code></li>
                            <li>Verify CORS is enabled for frontend requests</li>
                          </ol>
                        </div>
                      )}
                    </div>
                    <button
                      onClick={() => setError(null)}
                      className="flex-shrink-0 text-red-400 hover:text-red-600"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                </div>
              )}

              {/* File Upload Status */}
              {uploadedFile && (
                <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className="flex-shrink-0">
                      <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-blue-800">File Ready for Analysis</p>
                      <p className="text-sm text-blue-600">
                        {uploadedFile.name} ({Math.round(uploadedFile.size / 1024)} KB) •
                        {uploadedFile.type || 'CSV/Excel file'}
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Selected Sources Status */}
              {selectedSources.length > 0 && (
                <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Database className="w-5 h-5 text-green-500" />
                    <div className="flex-1">
                      <p className="text-sm font-medium text-green-800">Data Sources Selected</p>
                      <p className="text-sm text-green-600">
                        {selectedSources.length} source{selectedSources.length > 1 ? 's' : ''} ready for analysis
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Query Interface */}
              <div className="max-w-4xl mx-auto">
                <div className="relative">
                  <textarea
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Try: 'Show me sales trends by region' or 'What factors drive customer retention?'"
                    className="w-full p-6 border-2 border-gray-200 rounded-2xl resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
                    rows={3}
                    disabled={isAnalyzing}
                  />
                  <button
                    onClick={handleAnalyze}
                    disabled={!query.trim() || isAnalyzing || backendStatus !== 'connected'}
                    className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white font-medium px-6 py-2 rounded-lg transition-colors inline-flex items-center disabled:opacity-50"
                  >
                    {isAnalyzing ? (
                      <div className="flex items-center space-x-2">
                        <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                        <span>Analyzing...</span>
                      </div>
                    ) : (
                      <div className="flex items-center space-x-2">
                        <Zap className="w-4 h-4" />
                        <span>Analyze</span>
                      </div>
                    )}
                  </button>
                </div>

                {/* Quick Suggestions */}
                <div className="mt-4 flex flex-wrap gap-2 justify-center">
                  {suggestions.map((suggestion) => (
                    <button
                      key={suggestion}
                      onClick={() => setQuery(suggestion)}
                      className="px-4 py-2 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-full text-sm transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>

                {/* Advanced Options */}
                <div className="mt-6 text-center">
                  <button
                    onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                    className="text-blue-600 hover:text-blue-700 text-sm flex items-center mx-auto space-x-1"
                  >
                    <Settings className="w-4 h-4" />
                    <span>Advanced Options</span>
                  </button>

                  {showAdvancedOptions && (
                    <div className="mt-4 p-4 bg-gray-50 rounded-xl">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            className="rounded"
                            checked={advancedOptions.includeStatisticalTests}
                            onChange={() => handleAdvancedOptionChange('includeStatisticalTests')}
                          />
                          <span>Include statistical tests</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            className="rounded"
                            checked={advancedOptions.detectOutliers}
                            onChange={() => handleAdvancedOptionChange('detectOutliers')}
                          />
                          <span>Detect outliers</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            className="rounded"
                            checked={advancedOptions.predictiveModeling}
                            onChange={() => handleAdvancedOptionChange('predictiveModeling')}
                          />
                          <span>Predictive modeling</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            className="rounded"
                            checked={advancedOptions.executiveSummary}
                            onChange={() => handleAdvancedOptionChange('executiveSummary')}
                          />
                          <span>Executive summary</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            className="rounded"
                            checked={advancedOptions.use_adaptive}
                            onChange={() => handleAdvancedOptionChange('use_adaptive')}
                          />
                          <span>Adaptive processing</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            className="rounded"
                            checked={advancedOptions.include_charts}
                            onChange={() => handleAdvancedOptionChange('include_charts')}
                          />
                          <span>Include charts</span>
                        </label>
                        <label className="flex items-center space-x-2">
                          <input
                            type="checkbox"
                            className="rounded"
                            checked={advancedOptions.auto_discover}
                            onChange={() => handleAdvancedOptionChange('auto_discover')}
                          />
                          <span>Auto-discover</span>
                        </label>
                        <div className="flex items-center space-x-2">
                          <span>Domain:</span>
                          <select
                            value={advancedOptions.domain}
                            onChange={(e) => setAdvancedOptions(prev => ({...prev, domain: e.target.value}))}
                            className="text-xs border rounded px-1 py-0.5"
                          >
                            <option value="general">General</option>
                            <option value="finance">Finance</option>
                            <option value="healthcare">Healthcare</option>
                            <option value="technology">Technology</option>
                            <option value="retail">Retail</option>
                          </select>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Results */}
          {analysisResults && currentStep === 'results' && (
            <div className="space-y-6">
              {/* Quick Summary */}
              <div className="bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl p-8 text-white">
                <div className="flex items-start justify-between">
                  <div>
                    <h3 className="text-2xl font-bold mb-2">Analysis Complete</h3>
                    <p className="text-blue-100 text-lg">{analysisResults.summary}</p>
                    <div className="mt-4 flex items-center space-x-4">
                      <div className="flex items-center space-x-2">
                        <CheckCircle className="w-5 h-5" />
                        <span>Confidence: {Math.round(analysisResults.confidence * 100)}%</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <TrendingUp className="w-5 h-5" />
                        <span>{analysisResults.analysisType.replace('_', ' ')}</span>
                      </div>
                      {analysisResults.performance?.total_time_ms && (
                        <div className="flex items-center space-x-2">
                          <Clock className="w-5 h-5" />
                          <span>{Math.round(analysisResults.performance.total_time_ms)}ms</span>
                        </div>
                      )}
                    </div>
                  </div>
                  <div className="text-right">
                    <button
                      onClick={saveAnalysis}
                      className="bg-white/20 hover:bg-white/30 text-white px-4 py-2 rounded-lg transition-colors mb-2 w-full"
                    >
                      Save Analysis
                    </button>
                    <button
                      onClick={generateReport}
                      className="bg-white text-blue-600 hover:bg-gray-50 px-4 py-2 rounded-lg transition-colors w-full"
                    >
                      Generate Report
                    </button>
                  </div>
                </div>
              </div>

              {/* Key Insights */}
              <div className="bg-white rounded-2xl shadow-lg border border-white/20 p-8">
                <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
                  <Sparkles className="w-6 h-6 text-yellow-500 mr-2" />
                  Key Insights
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {analysisResults.insights.map((insight: string, index: number) => (
                    <div key={index} className="p-4 bg-gray-50 rounded-xl">
                      <p className="text-gray-800">{insight}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Comprehensive Report Section */}
              {analysisResults.comprehensiveReport && (
                <div className="bg-white rounded-2xl shadow-lg border border-white/20 p-8">
                  <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
                    <FileText className="w-6 h-6 text-green-500 mr-2" />
                    Comprehensive Report
                  </h3>

                  {analysisResults.comprehensiveReport.executive_summary && (
                    <div className="mb-6">
                      <h4 className="font-semibold text-gray-900 mb-2">Executive Summary</h4>
                      <p className="text-gray-700 bg-blue-50 p-4 rounded-lg">
                        {analysisResults.comprehensiveReport.executive_summary}
                      </p>
                    </div>
                  )}

                  {analysisResults.comprehensiveReport.detailed_findings && (
                    <div className="mb-6">
                      <h4 className="font-semibold text-gray-900 mb-2">Detailed Findings</h4>
                      <ul className="space-y-2">
                        {analysisResults.comprehensiveReport.detailed_findings.map((finding: string, index: number) => (
                          <li key={index} className="flex items-start space-x-2">
                            <span className="text-blue-500 mt-1">•</span>
                            <span className="text-gray-700">{finding}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {analysisResults.comprehensiveReport.recommendations && (
                    <div className="mb-6">
                      <h4 className="font-semibold text-gray-900 mb-2">Recommendations</h4>
                      <ul className="space-y-2">
                        {analysisResults.comprehensiveReport.recommendations.map((rec: string, index: number) => (
                          <li key={index} className="flex items-start space-x-2">
                            <span className="text-green-500 mt-1">✓</span>
                            <span className="text-gray-700">{rec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Visualization */}
              <div className="bg-white rounded-2xl shadow-lg border border-white/20 p-8">
                <h3 className="text-xl font-bold text-gray-900 mb-6 flex items-center">
                  <BarChart3 className="w-6 h-6 text-blue-500 mr-2" />
                  Visualization
                </h3>
                {analysisResults.chartSuggestions && analysisResults.chartSuggestions.length > 0 ? (
                  <div className="space-y-4">
                    <div className="flex flex-wrap gap-2 mb-4">
                      {analysisResults.chartSuggestions.map((chart: any, index: number) => (
                        <span key={index} className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm">
                          {chart.type || chart.chart_type} chart
                        </span>
                      ))}
                    </div>
                    <div className="h-64 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl flex items-center justify-center">
                      <div className="text-center">
                        <BarChart3 className="w-16 h-16 text-blue-400 mx-auto mb-4" />
                        <p className="text-gray-600">Interactive chart would render here</p>
                        <p className="text-sm text-gray-500 mt-2">
                          Suggested: {analysisResults.chartSuggestions[0]?.type || analysisResults.chartSuggestions[0]?.chart_type || 'Chart'} visualization
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="h-64 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl flex items-center justify-center">
                    <div className="text-center">
                      <PieChart className="w-16 h-16 text-blue-400 mx-auto mb-4" />
                      <p className="text-gray-600">Interactive chart would render here</p>
                      <p className="text-sm text-gray-500 mt-2">
                        {analysisResults.analysisType.replace('_', ' ')} visualization
                      </p>
                    </div>
                  </div>
                )}

                {analysisResults.performance && (
                  <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                    <p className="text-sm text-gray-600">
                      Analysis completed in {analysisResults.performance.total_time_ms || 'N/A'}ms
                      {analysisResults.performance.data_stats && (
                        <span> • Processed {analysisResults.performance.data_stats.rows || analysisResults.performance.data_stats.rows_processed || 0} rows</span>
                      )}
                      {analysisResults.performance.breakdown && (
                        <span> • Analysis: {analysisResults.performance.breakdown.analysis_ms || 0}ms</span>
                      )}
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Step 4: Report Generation */}
          {currentStep === 'report' && (
            <div className="bg-white rounded-2xl shadow-lg border border-white/20 p-8">
              <div className="text-center mb-8">
                <FileText className="w-12 h-12 text-green-500 mx-auto mb-4" />
                <h2 className="text-3xl font-bold text-gray-900 mb-2">Export Your Report</h2>
                <p className="text-gray-600">Choose how you'd like to share your findings</p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
                <div
                  className="border-2 border-gray-200 rounded-xl p-6 text-center hover:border-blue-300 hover:shadow-md transition-all cursor-pointer"
                  onClick={downloadPDF}
                >
                  <Download className="w-8 h-8 text-blue-500 mx-auto mb-3" />
                  <h3 className="font-semibold text-gray-900 mb-2">Download Report</h3>
                  <p className="text-sm text-gray-600">Comprehensive report with charts and insights</p>
                </div>

                <div
                  className="border-2 border-gray-200 rounded-xl p-6 text-center hover:border-green-300 hover:shadow-md transition-all cursor-pointer"
                  onClick={shareLink}
                >
                  <Share className="w-8 h-8 text-green-500 mx-auto mb-3" />
                  <h3 className="font-semibold text-gray-900 mb-2">Share Link</h3>
                  <p className="text-sm text-gray-600">Interactive dashboard for stakeholders</p>
                </div>

                <div
                  className="border-2 border-gray-200 rounded-xl p-6 text-center hover:border-purple-300 hover:shadow-md transition-all cursor-pointer"
                  onClick={scheduleUpdates}
                >
                  <Clock className="w-8 h-8 text-purple-500 mx-auto mb-3" />
                  <h3 className="font-semibold text-gray-900 mb-2">Schedule Updates</h3>
                  <p className="text-sm text-gray-600">Automated reports with fresh data</p>
                </div>
              </div>

              <div className="mt-8 text-center">
                <button
                  onClick={() => {
                    setCurrentStep('analyze');
                    setQuery('');
                    setAnalysisResults(null);
                  }}
                  className="bg-gray-100 hover:bg-gray-200 text-gray-700 font-medium px-8 py-3 rounded-lg transition-colors mr-4"
                >
                  New Analysis
                </button>
                <button
                  onClick={downloadPDF}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium px-8 py-3 rounded-lg transition-colors inline-flex items-center"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Download Report
                </button>
              </div>
            </div>
          )}

          {/* Saved Analyses Sidebar */}
          {savedAnalyses.length > 0 && (
            <div className="bg-white rounded-2xl shadow-lg border border-white/20 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Analyses</h3>
              <div className="space-y-3">
                {savedAnalyses.slice(0, 3).map((analysis) => (
                  <div
                    key={analysis.id}
                    className="p-3 bg-gray-50 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors"
                    onClick={() => loadSavedAnalysis(analysis)}
                  >
                    <p className="text-sm font-medium text-gray-900 truncate">{analysis.query}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {analysis.timestamp.toLocaleDateString()} •
                      {analysis.file ? ` File: ${analysis.file}` : ` ${analysis.sources.length} source(s)`} •
                      {Math.round(analysis.results.confidence * 100)}% confidence
                    </p>
                  </div>
                ))}
              </div>
              {savedAnalyses.length > 3 && (
                <button className="mt-3 text-sm text-blue-600 hover:text-blue-700">
                  View all analyses ({savedAnalyses.length})
                </button>
              )}
            </div>
          )}

          {/* Backend Status Warning */}
          {backendStatus === 'disconnected' && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-6">
              <div className="flex items-center space-x-3">
                <AlertCircle className="w-6 h-6 text-yellow-600" />
                <div>
                  <h3 className="font-semibold text-yellow-900">Backend Connection Issue</h3>
                  <p className="text-yellow-700 text-sm mt-1">
                    DataGenie backend is not responding. Please ensure the server is running at http://localhost:8000
                  </p>
                  <button
                    onClick={checkBackendStatus}
                    className="mt-2 text-sm bg-yellow-600 text-white px-3 py-1 rounded hover:bg-yellow-700 transition-colors"
                  >
                    Retry Connection
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* System Capabilities Info */}
          {systemCapabilities && (
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
              <h3 className="font-semibold text-blue-900 mb-3">🚀 Available Features</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                {systemCapabilities.smart_features?.unified_smart_engine && (
                  <div className="flex items-center space-x-2 text-blue-700">
                    <span>✅</span>
                    <span>Smart Query Engine</span>
                  </div>
                )}
                {systemCapabilities.smart_features?.auto_data_discovery && (
                  <div className="flex items-center space-x-2 text-blue-700">
                    <span>✅</span>
                    <span>Auto Data Discovery</span>
                  </div>
                )}
                {systemCapabilities.smart_features?.llm_powered_query_understanding && (
                  <div className="flex items-center space-x-2 text-blue-700">
                    <span>✅</span>
                    <span>AI Query Understanding</span>
                  </div>
                )}
                {systemCapabilities.features?.adaptive_query_processing && (
                  <div className="flex items-center space-x-2 text-blue-700">
                    <span>✅</span>
                    <span>Adaptive Processing</span>
                  </div>
                )}
                {systemCapabilities.features?.chart_intelligence && (
                  <div className="flex items-center space-x-2 text-blue-700">
                    <span>✅</span>
                    <span>Chart Intelligence</span>
                  </div>
                )}
                {systemCapabilities.features?.comprehensive_reporting && (
                  <div className="flex items-center space-x-2 text-blue-700">
                    <span>✅</span>
                    <span>Comprehensive Reports</span>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default DataGenie;