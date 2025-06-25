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
  Search
} from 'lucide-react';

// Types
interface DataSource {
  id: string;
  name: string;
  type: string;
  confidence: number;
  status: string;
}

interface AnalysisResults {
  summary: string;
  insights: string[];
  chartData: any[];
  confidence: number;
  analysisType: string;
  chartSuggestions?: any[];
  performance?: any;
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

// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Fixed API Service Functions
const apiService = {
  async discoverSources() {
    try {
      const response = await fetch(`${API_BASE_URL}/discover-sources/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer demo-token`
        }
      });
      return await response.json();
    } catch (error) {
      console.error('Error discovering sources:', error);
      return { status: 'error', discovered_sources: [], recommendations: [] };
    }
  },

  async analyzeData(query: string, file: File | null = null, selectedSources: string[] = []) {
    try {
      console.log('📡 Starting analysis request...');
      console.log('📝 Query:', query);
      console.log('📄 File:', file ? `${file.name} (${file.size} bytes)` : 'None');
      console.log('🔗 Sources:', selectedSources);

      // STEP 1: Test basic connectivity
      console.log('🔍 Testing backend health...');
      try {
        const healthResponse = await fetch(`${API_BASE_URL}/health/`);
        const healthData = await healthResponse.json();
        console.log('❤️ Backend health:', healthData);

        if (!healthData.status || healthData.status !== 'healthy') {
          throw new Error('Backend is not healthy');
        }
      } catch (healthError) {
        console.error('💔 Backend health check failed:', healthError);
        throw new Error('Backend server is not accessible. Please ensure it\'s running on http://localhost:8000');
      }

      // STEP 2: Test a simple GET request first
      console.log('🧪 Testing basic API with GET request...');
      try {
        const statusResponse = await fetch(`${API_BASE_URL}/`);
        const statusData = await statusResponse.json();
        console.log('🏠 Root endpoint response:', statusData);
      } catch (statusError) {
        console.error('🚫 Basic API test failed:', statusError);
      }

      // STEP 3: Test the simpler endpoint first
      console.log('🧪 Testing simpler upload endpoint...');
      try {
        const testFormData = new FormData();
        testFormData.append('prompt', query);
        if (file) {
          testFormData.append('file', file, file.name);
        }
        testFormData.append('use_adaptive', 'true');
        testFormData.append('include_charts', 'true');
        testFormData.append('auto_discover', file ? 'false' : 'true');

        const testResponse = await fetch(`${API_BASE_URL}/test-upload/`, {
          method: 'POST',
          body: testFormData
        });

        console.log('🧪 Test endpoint response:', testResponse.status);

        if (testResponse.ok) {
          const testData = await testResponse.json();
          console.log('🧪 Test endpoint data:', testData);

          if (testData.status === 'success') {
            console.log('✅ Test upload successful! File is valid.');
          } else {
            console.log('❌ Test upload failed:', testData.error);
          }
        } else {
          const testError = await testResponse.text();
          console.log('❌ Test endpoint failed:', testError);
        }
      } catch (testError) {
        console.error('🧪 Test endpoint error:', testError);
      }

      // STEP 4: Prepare the actual analysis request
      console.log('📦 Preparing FormData request...');

      const formData = new FormData();

      // FIXED: Send everything as form data
      formData.append('prompt', query);
      formData.append('use_adaptive', 'true');
      formData.append('include_charts', 'true');
      formData.append('auto_discover', file ? 'false' : 'true');

      if (file) {
        console.log('📄 Adding file to FormData...');
        formData.append('file', file, file.name);
        console.log(`📄 File details: ${file.name}, ${file.size} bytes, ${file.type}`);
      }

      if (selectedSources.length > 0) {
        formData.append('data_source_id', selectedSources[0]);
      }

      // Debug: Log what we're sending
      console.log('📋 Request details:');
      console.log(`  URL: ${API_BASE_URL}/analyze/`);
      console.log(`  Method: POST`);
      console.log(`  Has file: ${!!file}`);
      console.log(`  Prompt length: ${query.length}`);

      // STEP 5: Make the actual request
      console.log('🚀 Sending analysis request...');

      const response = await fetch(`${API_BASE_URL}/analyze/`, {
        method: 'POST',
        body: formData
        // No headers - let browser handle Content-Type for FormData
      });

      console.log('📬 Response received:');
      console.log(`  Status: ${response.status} ${response.statusText}`);
      console.log(`  OK: ${response.ok}`);

      // Log response headers
      const responseHeaders: Record<string, string> = {};
      response.headers.forEach((value, key) => {
        responseHeaders[key] = value;
      });
      console.log('  Headers:', responseHeaders);

      // Handle response
      if (!response.ok) {
        console.error('❌ Request failed with status:', response.status);

        let errorDetails;
        const contentType = response.headers.get('content-type') || '';

        console.log('📄 Response content-type:', contentType);

        if (contentType.includes('application/json')) {
          try {
            errorDetails = await response.json();
            console.error('📄 Error response (JSON):', errorDetails);
          } catch (jsonError) {
            console.error('❌ Failed to parse JSON error response:', jsonError);
            errorDetails = { error: 'Failed to parse error response' };
          }
        } else {
          try {
            const textError = await response.text();
            console.error('📄 Error response (Text):', textError);
            errorDetails = { error: textError };
          } catch (textError) {
            console.error('❌ Failed to read error response:', textError);
            errorDetails = { error: 'Failed to read error response' };
          }
        }

        // Create user-friendly error message
        let userMessage = 'Analysis failed';

        if (response.status === 422) {
          userMessage = 'The request format is invalid. Please check your file format and try again.';

          // Add specific suggestions based on error details
          if (errorDetails.detail && typeof errorDetails.detail === 'string') {
            if (errorDetails.detail.includes('file')) {
              userMessage += ' Make sure your file is a valid CSV or Excel file.';
            }
            if (errorDetails.detail.includes('prompt')) {
              userMessage += ' Make sure your question is not empty.';
            }
          }
        } else if (response.status === 400) {
          userMessage = 'Bad request. Please verify your file and query are valid.';
        } else if (response.status === 500) {
          userMessage = 'Server error. Please try again in a moment.';
        } else {
          userMessage = `Request failed with status ${response.status}. Please try again.`;
        }

        throw new Error(userMessage);
      }

      // Parse successful response
      console.log('✅ Request successful, parsing response...');
      let result;

      try {
        result = await response.json();
        console.log('✅ Response parsed successfully');
        console.log('📊 Response summary:', {
          status: result.status,
          hasAnalysis: !!result.analysis,
          hasData: !!result.analysis?.data,
          dataLength: result.analysis?.data?.length || 0
        });
      } catch (parseError) {
        console.error('❌ Failed to parse successful response:', parseError);
        throw new Error('Received invalid response from server');
      }

      return result;

    } catch (error) {
      console.error('💥 Complete error details:', {
        name: (error as Error).name,
        message: (error as Error).message,
        stack: (error as Error).stack
      });

      // Re-throw with appropriate message
      if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
        throw new Error('Cannot connect to DataGenie backend. Please ensure the server is running on http://localhost:8000');
      }

      throw error;
    }
  }
};

// Mock data fallback
const mockDataSources: DataSource[] = [
  { id: 'postgres_prod', name: 'Production Database', type: 'PostgreSQL', confidence: 0.92, status: 'available' },
  { id: 'sales_csv', name: 'Sales Data Export', type: 'CSV', confidence: 0.87, status: 'available' },
  { id: 'tableau_server', name: 'Tableau Server', type: 'Tableau', confidence: 0.78, status: 'available' },
  { id: 'api_crm', name: 'CRM API', type: 'REST API', confidence: 0.85, status: 'available' }
];

const mockAnalysisResults: AnalysisResults = {
  summary: "Sales trend analysis shows 23% growth in Q4 with strong performance in Enterprise segment",
  insights: [
    "📈 Enterprise sales increased 31% quarter-over-quarter",
    "🎯 Customer acquisition cost decreased by 15%",
    "⚠️ SMB segment showing 8% decline - requires attention",
    "🔥 Product A driving 45% of total growth"
  ],
  chartData: [
    { month: 'Jan', sales: 45000, target: 40000 },
    { month: 'Feb', sales: 52000, target: 45000 },
    { month: 'Mar', sales: 48000, target: 50000 },
    { month: 'Apr', sales: 61000, target: 55000 },
    { month: 'May', sales: 68000, target: 60000 },
    { month: 'Jun', sales: 71000, target: 65000 }
  ],
  confidence: 0.94,
  analysisType: 'trend_analysis'
};

// Main Component
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
  const [advancedOptions, setAdvancedOptions] = useState({
    includeStatisticalTests: false,
    detectOutliers: false,
    predictiveModeling: false,
    executiveSummary: true
  });

  // Auto-discover data sources on load
  useEffect(() => {
    const discoverDataSources = async () => {
      setIsDiscovering(true);
      try {
        const response = await apiService.discoverSources();
        if (response.status === 'success' && response.recommendations) {
          const sources = response.recommendations.map((rec: any) => ({
            id: rec.source_id,
            name: rec.source_id.replace(/_/g, ' ').replace(/\b\w/g, (l: string) => l.toUpperCase()),
            type: rec.context?.type || 'Unknown',
            confidence: rec.confidence,
            status: 'available'
          }));
          setDataSources(sources);
        } else {
          console.log('Using mock data sources');
          setDataSources(mockDataSources);
        }
      } catch (error) {
        console.error('Discovery failed, using mock data:', error);
        setDataSources(mockDataSources);
      } finally {
        setIsDiscovering(false);
      }
    };

    discoverDataSources();
  }, []);

  // Load saved analyses on mount
  useEffect(() => {
    try {
      const saved = JSON.parse(localStorage.getItem('dataGenie_analyses') || '[]');
      setSavedAnalyses(saved);
    } catch (error) {
      console.error('Failed to load saved analyses:', error);
    }
  }, []);

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

---
This report was generated by DataGenie Analytics Platform
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
      timestamp: new Date().toISOString()
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
      setUploadedFile(file);
      setSelectedSources([]);
      setError(null); // Clear any previous errors
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
      console.log('🔍 Starting analysis with:', {
        query,
        hasFile: !!uploadedFile,
        fileName: uploadedFile?.name,
        fileSize: uploadedFile?.size,
        fileType: uploadedFile?.type,
        selectedSources: selectedSources.length
      });

      const response = await apiService.analyzeData(query, uploadedFile, selectedSources);

      console.log('📡 Backend response:', response);

      if (response.status === 'success') {
        const transformedResults: AnalysisResults = {
          summary: response.analysis?.summary || 'Analysis completed successfully',
          insights: response.analysis?.insights || ['Analysis completed with no specific insights'],
          chartData: response.analysis?.data || [],
          confidence: response.system_info?.confidence || 0.8,
          analysisType: response.analysis?.type || 'general_analysis',
          chartSuggestions: response.chart_intelligence?.suggested_charts || [],
          performance: response.performance || {}
        };

        console.log('✅ Analysis successful, showing results');
        setAnalysisResults(transformedResults);
        setCurrentStep('results');

        // Clear error on success
        setError(null);
      } else {
        console.error('❌ Backend returned error:', response.error);
        throw new Error(response.error || response.message || 'Analysis failed');
      }
    } catch (error) {
      console.error('💥 Analysis error:', error);

      const errorMessage = (error as Error).message || 'Analysis failed';

      if (errorMessage.includes('Failed to fetch') || errorMessage.includes('NetworkError')) {
        setError('Cannot connect to DataGenie backend. Please check if the server is running on http://localhost:8000');
      } else if (errorMessage.includes('HTTP 422')) {
        setError('Invalid request format. Please check your file format and try again.');
      } else if (errorMessage.includes('HTTP 400')) {
        setError('Bad request. Please check your file format or query and try again.');
      } else if (errorMessage.includes('HTTP 500')) {
        setError('Server error. Please try again or contact support.');
      } else {
        setError(`Analysis failed: ${errorMessage}`);
      }

      // For development: Show file info if we have a file
      if (uploadedFile) {
        console.log('📄 File details:', {
          name: uploadedFile.name,
          size: uploadedFile.size,
          type: uploadedFile.type,
          lastModified: new Date(uploadedFile.lastModified).toISOString()
        });
      }

      setCurrentStep('analyze'); // Stay on analyze step to fix the issue
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
    "Show revenue trends",
    "Compare regional performance",
    "Identify top customers",
    "Analyze seasonal patterns"
  ];

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
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2 text-sm">
              <div className={`w-2 h-2 rounded-full ${
                dataSources.length > 0 ? 'bg-green-500' : 'bg-yellow-500'
              }`} />
              <span className="text-gray-600">
                {isDiscovering ? 'Discovering...' : `${dataSources.length} sources`}
              </span>
            </div>
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
                  <p className="text-red-700 text-sm">{error}</p>
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
                        <h4 className="font-semibold text-gray-900">{source.name}</h4>
                        <div className="flex items-center space-x-2">
                          <div className={`w-2 h-2 rounded-full ${
                            source.confidence > 0.8 ? 'bg-green-500' : 'bg-yellow-500'
                          }`} />
                          <span className="text-xs text-gray-500">{Math.round(source.confidence * 100)}%</span>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600">{source.type}</p>
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
                    <div className="flex-shrink-0">
                      <svg className="w-5 h-5 text-red-400 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                    </div>
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
                    disabled={!query.trim() || isAnalyzing}
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
                          {chart.type} chart
                        </span>
                      ))}
                    </div>
                    <div className="h-64 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl flex items-center justify-center">
                      <div className="text-center">
                        <BarChart3 className="w-16 h-16 text-blue-400 mx-auto mb-4" />
                        <p className="text-gray-600">Interactive chart would render here</p>
                        <p className="text-sm text-gray-500 mt-2">
                          Suggested: {analysisResults.chartSuggestions[0]?.type || 'Chart'} visualization
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
                        <span> • Processed {analysisResults.performance.data_stats.rows} rows</span>
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
                  <h3 className="font-semibold text-gray-900 mb-2">Download PDF</h3>
                  <p className="text-sm text-gray-600">Executive summary with charts and insights</p>
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
                      {new Date(analysis.timestamp).toLocaleDateString()} •
                      {analysis.file ? ` File: ${analysis.file}` : ` ${analysis.sources.length} source(s)`}
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
        </div>
      </div>
    </div>
  );
};

export default DataGenie;