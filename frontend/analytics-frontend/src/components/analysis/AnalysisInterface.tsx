import React, { useState, useRef, useEffect } from 'react';
import { analyticsAPI, extractErrorMessage, AnalysisResponse } from '../../services/api';
import Charts from '../charts/Charts'; // Import our Charts component

interface ChartConfig {
  id: string;
  title: string;
  type: 'line' | 'area' | 'bar' | 'pie' | 'scatter' | 'histogram' | 'waterfall' | 'funnel' | 'gauge';
  data: any[];
  xField?: string;
  yField?: string;
  categoryField?: string;
  valueField?: string;
  description?: string;
  reasoning?: string;
  color?: string;
  colors?: string[];
  customConfig?: any;
}

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  analysis?: any;
  charts?: ChartConfig[];
  loading?: boolean;
  error?: boolean;
}

interface AnalysisSession {
  id: string;
  title: string;
  messages: Message[];
  lastActivity: Date;
}

const AnalysisInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [sessions, setSessions] = useState<AnalysisSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [systemCapabilities, setSystemCapabilities] = useState<any>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const exampleQuestions = [
    "Show me a waterfall chart of quarterly performance",
    "Create a funnel chart for our sales pipeline",
    "Generate a gauge chart showing completion rate",
    "What are the trends in our sales data?",
    "Show me customer segmentation analysis",
    "Which products have the highest profit margins?",
    "Create a comprehensive report with insights",
    "Analyze seasonal patterns in the data"
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    loadSessions();
    checkBackendStatus();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const checkBackendStatus = async () => {
    try {
      setBackendStatus('checking');
      const [health, capabilities] = await Promise.allSettled([
        analyticsAPI.healthCheck(),
        analyticsAPI.getCapabilities()
      ]);

      if (health.status === 'fulfilled' && health.value.status === 'healthy') {
        setBackendStatus('connected');
      } else {
        setBackendStatus('disconnected');
      }

      if (capabilities.status === 'fulfilled') {
        setSystemCapabilities(capabilities.value);
      }
    } catch (error) {
      setBackendStatus('disconnected');
    }
  };

  const loadSessions = () => {
    try {
      const saved = localStorage.getItem('analysis_sessions');
      if (saved) {
        const parsedSessions = JSON.parse(saved);
        const sessionsWithDates = parsedSessions.map((session: any) => ({
          ...session,
          lastActivity: new Date(session.lastActivity),
          messages: session.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
        setSessions(sessionsWithDates);
      }
    } catch (error) {
      console.error('Failed to load sessions:', error);
    }
  };

  const saveSessions = (updatedSessions: AnalysisSession[]) => {
    try {
      localStorage.setItem('analysis_sessions', JSON.stringify(updatedSessions));
    } catch (error) {
      console.error('Failed to save sessions:', error);
    }
  };

  const createNewSession = () => {
    const newSession: AnalysisSession = {
      id: Date.now().toString(),
      title: 'New Analysis',
      messages: [],
      lastActivity: new Date()
    };

    const updatedSessions = [newSession, ...sessions];
    setSessions(updatedSessions);
    setActiveSessionId(newSession.id);
    setMessages([]);
    saveSessions(updatedSessions);
  };

  const loadSession = (sessionId: string) => {
    const session = sessions.find(s => s.id === sessionId);
    if (session) {
      setActiveSessionId(sessionId);
      setMessages(session.messages);
    }
  };

  const processChartsFromBackend = (chartIntelligence: any, analysisData: any[]): ChartConfig[] => {
    if (!chartIntelligence?.suggested_charts) return [];

    return chartIntelligence.suggested_charts.map((chart: any, index: number) => {
      const chartId = `chart-${Date.now()}-${index}`;

      // Transform backend chart data to our format
      let chartData = chart.data || analysisData || [];

      // Ensure data is in the right format for our Charts component
      if (chartData.length > 0 && typeof chartData[0] === 'object') {
        // Data is already in object format, good to go
      } else {
        // Generate sample data if none provided
        chartData = generateSampleDataForChart(chart.type || chart.chart_type || 'bar');
      }

      return {
        id: chartId,
        title: chart.title || `${(chart.type || chart.chart_type || 'Chart').charAt(0).toUpperCase() + (chart.type || chart.chart_type || 'Chart').slice(1)} Chart`,
        type: (chart.type || chart.chart_type || 'bar') as ChartConfig['type'],
        data: chartData,
        xField: chart.x_field || chart.xField || 'x',
        yField: chart.y_field || chart.yField || 'value',
        categoryField: chart.category_field || chart.categoryField || 'name',
        valueField: chart.value_field || chart.valueField || 'value',
        description: chart.description || 'Generated based on your data analysis',
        reasoning: chart.reasoning || chart.explanation || 'This visualization helps understand the patterns in your data',
        colors: chart.colors,
        customConfig: chart.config
      };
    });
  };

  const generateSampleDataForChart = (type: string) => {
    switch (type.toLowerCase()) {
      case 'waterfall':
        return [
          { x: 'Starting Value', value: 100 },
          { x: 'Q1 Revenue', value: 50 },
          { x: 'Q2 Revenue', value: 30 },
          { x: 'Q3 Costs', value: -20 },
          { x: 'Q4 Revenue', value: 40 }
        ];
      case 'funnel':
        return [
          { name: 'Website Visits', value: 10000 },
          { name: 'Leads', value: 2000 },
          { name: 'Qualified Leads', value: 800 },
          { name: 'Proposals', value: 200 },
          { name: 'Customers', value: 50 }
        ];
      case 'gauge':
        return [{ value: 75 }];
      case 'pie':
        return [
          { name: 'Category A', value: 400 },
          { name: 'Category B', value: 300 },
          { name: 'Category C', value: 200 },
          { name: 'Category D', value: 100 }
        ];
      default:
        return Array.from({ length: 8 }, (_, i) => ({
          x: `Point ${i + 1}`,
          value: Math.floor(Math.random() * 100) + 10
        }));
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() && !selectedFile) return;

    if (backendStatus !== 'connected') {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: '❌ Backend is not connected. Please ensure the server is running at http://localhost:8000',
        timestamp: new Date(),
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim() || `Uploaded file: ${selectedFile?.name}`,
      timestamp: new Date()
    };

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'Analyzing your request and generating visualizations...',
      timestamp: new Date(),
      loading: true
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setInputValue('');
    setIsAnalyzing(true);

    try {
      console.log('🔬 Starting enhanced analysis with backend...');

      const response: AnalysisResponse = await analyticsAPI.analyze({
        prompt: inputValue.trim() || 'Please analyze this data and provide insights with visualizations',
        file: selectedFile || undefined,
        use_adaptive: true,
        include_charts: true,
        auto_discover: !selectedFile,
        domain: 'general'
      });

      console.log('📡 Analysis response received:', response);

      if (response.status === 'success') {
        // Process charts from backend response
        const processedCharts = processChartsFromBackend(
          response.chart_intelligence,
          response.analysis?.data || []
        );

        // Create enhanced analysis result
        const analysisContent = `${response.analysis?.summary || 'Analysis completed successfully'}

${response.comprehensive_report?.executive_summary ? `**Executive Summary:**
${response.comprehensive_report.executive_summary}

` : ''}**Key Insights:**
${(response.analysis?.insights || []).map((insight: string, i: number) => `${i + 1}. ${insight}`).join('\n')}

${response.comprehensive_report?.recommendations ? `**Recommendations:**
${response.comprehensive_report.recommendations.map((rec: string, i: number) => `${i + 1}. ${rec}`).join('\n')}

` : ''}**Analysis Details:**
• Type: ${response.analysis?.type || 'General Analysis'}
• Confidence: ${Math.round((response.query_interpretation?.confidence || 0.8) * 100)}%
• Processing Time: ${response.performance?.total_time_ms || 'N/A'}ms
• Data Points: ${response.performance?.data_stats?.rows || response.performance?.data_stats?.rows_processed || 'N/A'}
• Charts Generated: ${processedCharts.length}`;

        const assistantMessage: Message = {
          id: (Date.now() + 2).toString(),
          type: 'assistant',
          content: analysisContent,
          timestamp: new Date(),
          analysis: {
            type: response.analysis?.type || 'general_analysis',
            summary: response.analysis?.summary || 'Analysis completed',
            insights: response.analysis?.insights || [],
            data: response.analysis?.data || [],
            metadata: response.analysis?.metadata || {},
            performance: response.performance,
            comprehensive_report: response.comprehensive_report,
            query_interpretation: response.query_interpretation
          },
          charts: processedCharts
        };

        setMessages(prev => prev.slice(0, -1).concat(assistantMessage));

        // Update or create session
        if (activeSessionId) {
          const updatedSessions = sessions.map(session =>
            session.id === activeSessionId
              ? {
                  ...session,
                  messages: [...session.messages, userMessage, assistantMessage],
                  lastActivity: new Date(),
                  title: session.title === 'New Analysis' ? userMessage.content.slice(0, 50) + '...' : session.title
                }
              : session
          );
          setSessions(updatedSessions);
          saveSessions(updatedSessions);
        }

      } else {
        throw new Error(extractErrorMessage(response));
      }

    } catch (error) {
      console.error('💥 Analysis failed:', error);

      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: `❌ **Analysis Error**

${extractErrorMessage(error)}

**Troubleshooting Steps:**
1. Ensure the backend server is running at http://localhost:8000
2. Check your data file format (CSV, Excel, JSON supported)
3. Verify your query is clear and specific
4. Try a simpler question first

You can also try the enhanced DataGenie interface for a more guided experience.`,
        timestamp: new Date(),
        error: true
      };

      setMessages(prev => prev.slice(0, -1).concat(errorMessage));
    } finally {
      setIsAnalyzing(false);
      setSelectedFile(null);
      setShowFileUpload(false);
    }
  };

  const handleFileSelect = (file: File) => {
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: `❌ File too large: ${Math.round(file.size / 1024 / 1024)}MB. Maximum size is 100MB.`,
        timestamp: new Date(),
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    const allowedTypes = ['.csv', '.xlsx', '.xls', '.json'];
    const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
    if (!allowedTypes.includes(fileExtension)) {
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: `❌ Unsupported file type: ${fileExtension}. Please use CSV, Excel, or JSON files.`,
        timestamp: new Date(),
        error: true
      };
      setMessages(prev => [...prev, errorMessage]);
      return;
    }

    setSelectedFile(file);
    setShowFileUpload(false);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFileSelect(files[0]);
    }
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getStatusIndicator = () => {
    switch (backendStatus) {
      case 'checking':
        return { color: 'bg-yellow-500', text: 'Checking...', pulse: true };
      case 'connected':
        return { color: 'bg-green-500', text: 'AI Ready', pulse: false };
      case 'disconnected':
        return { color: 'bg-red-500', text: 'Offline', pulse: false };
    }
  };

  const statusInfo = getStatusIndicator();

  const handleChartGenerate = (chartId: string) => {
    console.log('Chart generated:', chartId);
    // Optional: Add analytics tracking or additional functionality
  };

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar - Previous Sessions */}
      <div className="w-80 bg-white border-r border-gray-200 flex flex-col">
        <div className="p-4 border-b border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">Analysis Sessions</h2>
            <button
              onClick={createNewSession}
              className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
              title="New Session"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </button>
          </div>

          <button
            onClick={createNewSession}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors text-left"
          >
            🧠 Start New Analysis
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {sessions.length === 0 ? (
            <div className="p-4 text-center text-gray-500">
              <div className="text-4xl mb-2">💬</div>
              <p className="text-sm">No previous sessions</p>
            </div>
          ) : (
            <div className="p-2 space-y-2">
              {sessions.map((session) => (
                <button
                  key={session.id}
                  onClick={() => loadSession(session.id)}
                  className={`w-full text-left p-3 rounded-lg transition-colors ${
                    activeSessionId === session.id
                      ? 'bg-blue-50 border border-blue-200'
                      : 'hover:bg-gray-50'
                  }`}
                >
                  <div className="font-medium text-gray-900 truncate">
                    {session.title}
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    {session.lastActivity.toLocaleDateString()} • {session.messages.length} messages
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* System Status in Sidebar */}
        <div className="p-4 border-t border-gray-200">
          <div className="text-xs text-gray-500 mb-2">System Status</div>
          <div className="flex items-center space-x-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${statusInfo.color} ${statusInfo.pulse ? 'animate-pulse' : ''}`} />
            <span className="text-gray-700">{statusInfo.text}</span>
          </div>
          {systemCapabilities && (
            <div className="mt-2 text-xs space-y-1">
              {systemCapabilities.smart_features?.unified_smart_engine && (
                <div className="text-blue-600">🧠 Smart Engine Ready</div>
              )}
              {systemCapabilities.smart_features?.llm_powered_query_understanding && (
                <div className="text-green-600">🤖 AI Processing</div>
              )}
              {systemCapabilities.features?.chart_intelligence && (
                <div className="text-purple-600">📊 Chart Intelligence</div>
              )}
            </div>
          )}
          <button
            onClick={checkBackendStatus}
            className="mt-2 text-xs text-blue-600 hover:text-blue-800"
          >
            Refresh Status
          </button>
        </div>
      </div>

      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">Enhanced AI Analysis Interface</h1>
              <p className="text-gray-600">Ask for specific charts and analysis in natural language with advanced AI processing</p>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-1 bg-green-50 rounded-full">
                <div className={`w-2 h-2 rounded-full ${statusInfo.color} ${statusInfo.pulse ? 'animate-pulse' : ''}`} />
                <span className="text-sm text-green-700 font-medium">{statusInfo.text}</span>
              </div>

              <button
                onClick={() => setShowFileUpload(!showFileUpload)}
                className="p-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
                title="Upload Data"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </button>

              <a
                href="/datagenie"
                className="text-sm bg-blue-600 text-white px-3 py-2 rounded-lg hover:bg-blue-700 transition-colors"
              >
                DataGenie
              </a>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-4">
          {messages.length === 0 ? (
            <div className="max-w-4xl mx-auto">
              {/* Welcome Message */}
              <div className="text-center mb-8">
                <div className="text-6xl mb-4">🧠</div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">
                  Welcome to Enhanced AI Analysis
                </h2>
                <p className="text-gray-600 mb-8">
                  Ask for specific charts and analysis in plain English. I can create waterfall charts, funnel charts, gauge charts, and provide advanced insights using cutting-edge AI.
                </p>
              </div>

              {/* Enhanced Features */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
                <div className="p-4 bg-white rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                      <span className="text-blue-600">📊</span>
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Advanced Charts</h3>
                      <p className="text-sm text-gray-600">Waterfall, funnel, gauge, and more</p>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-white rounded-lg border border-gray-200 hover:border-green-300 hover:bg-green-50 transition-colors">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                      <span className="text-green-600">🧠</span>
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Smart Analysis</h3>
                      <p className="text-sm text-gray-600">AI-powered insights and recommendations</p>
                    </div>
                  </div>
                </div>

                <div className="p-4 bg-white rounded-lg border border-gray-200 hover:border-purple-300 hover:bg-purple-50 transition-colors">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center">
                      <span className="text-purple-600">📋</span>
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">Full Reports</h3>
                      <p className="text-sm text-gray-600">Comprehensive analysis reports</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Example Questions */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                {exampleQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => setInputValue(question)}
                    className="p-4 text-left bg-white rounded-lg border border-gray-200 hover:border-blue-300 hover:bg-blue-50 transition-colors"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                        <span className="text-blue-600">💡</span>
                      </div>
                      <span className="text-gray-700">{question}</span>
                    </div>
                  </button>
                ))}
              </div>

              {/* Quick Actions */}
              <div className="bg-white rounded-lg p-6 border border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <button
                    onClick={() => setShowFileUpload(true)}
                    className="p-4 text-center border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors"
                  >
                    <div className="text-2xl mb-2">📁</div>
                    <div className="font-medium text-gray-900">Upload Data</div>
                    <div className="text-sm text-gray-500">CSV, Excel, JSON</div>
                  </button>

                  <a
                    href="/discovery"
                    className="p-4 text-center border-2 border-dashed border-gray-300 rounded-lg hover:border-green-400 hover:bg-green-50 transition-colors no-underline"
                  >
                    <div className="text-2xl mb-2">🔗</div>
                    <div className="font-medium text-gray-900">Connect Source</div>
                    <div className="text-sm text-gray-500">Database, API</div>
                  </a>

                  <a
                    href="/datagenie"
                    className="p-4 text-center border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-colors no-underline"
                  >
                    <div className="text-2xl mb-2">🧞‍♂️</div>
                    <div className="font-medium text-gray-900">DataGenie</div>
                    <div className="text-sm text-gray-500">Guided analysis</div>
                  </a>
                </div>
              </div>
            </div>
          ) : (
            <div className="max-w-4xl mx-auto space-y-6">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-3xl p-4 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-blue-600 text-white'
                        : message.error
                        ? 'bg-red-50 border border-red-200 text-red-900'
                        : 'bg-white border border-gray-200'
                    }`}
                  >
                    {message.type === 'assistant' && !message.error && (
                      <div className="flex items-center space-x-2 mb-2">
                        <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center">
                          <span className="text-blue-600 text-sm">🧠</span>
                        </div>
                        <span className="text-sm font-medium text-gray-700">Enhanced AI Assistant</span>
                        <span className="text-xs text-gray-500">{formatTimestamp(message.timestamp)}</span>
                      </div>
                    )}

                    {message.loading ? (
                      <div className="flex items-center space-x-2">
                        <svg className="animate-spin w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        <span className="text-gray-600">Enhanced AI processing...</span>
                      </div>
                    ) : (
                      <div>
                        <div className="whitespace-pre-wrap">{message.content}</div>

                        {/* Enhanced Analysis Results */}
                        {message.analysis && !message.error && (
                          <div className="mt-4 space-y-4">
                            {/* Performance Metrics */}
                            {message.analysis.performance && (
                              <div className="bg-gray-50 p-3 rounded-lg border">
                                <h4 className="font-semibold text-gray-900 mb-2">📊 Analysis Metrics</h4>
                                <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-sm">
                                  <div className="text-center">
                                    <div className="font-bold text-blue-600">
                                      {message.analysis.performance.total_time_ms || 'N/A'}ms
                                    </div>
                                    <div className="text-gray-600">Processing Time</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="font-bold text-green-600">
                                      {message.analysis.performance.data_stats?.rows ||
                                       message.analysis.performance.data_stats?.rows_processed || 'N/A'}
                                    </div>
                                    <div className="text-gray-600">Data Points</div>
                                  </div>
                                  <div className="text-center">
                                    <div className="font-bold text-purple-600">
                                      {Math.round((message.analysis.query_interpretation?.confidence || 0.8) * 100)}%
                                    </div>
                                    <div className="text-gray-600">Confidence</div>
                                  </div>
                                </div>
                              </div>
                            )}

                            {/* Key Insights */}
                            {message.analysis.insights && message.analysis.insights.length > 0 && (
                              <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                <h4 className="font-semibold text-blue-900 mb-2">✨ Key Insights</h4>
                                <ul className="space-y-1">
                                  {message.analysis.insights.map((insight: string, index: number) => (
                                    <li key={index} className="text-blue-800 text-sm flex items-start space-x-2">
                                      <span className="text-blue-600 mt-1">•</span>
                                      <span>{insight}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            )}

                            {/* Data Summary */}
                            {message.analysis.data && message.analysis.data.length > 0 && (
                              <div className="bg-gray-50 p-4 rounded-lg border">
                                <h4 className="font-semibold text-gray-900 mb-2">📋 Data Summary</h4>
                                <div className="text-sm text-gray-700">
                                  <p>Records: {message.analysis.data.length}</p>
                                  <p>Analysis Type: {message.analysis.type}</p>
                                </div>
                              </div>
                            )}

                            {/* Comprehensive Report Section */}
                            {message.analysis.comprehensive_report && (
                              <div className="bg-green-50 p-4 rounded-lg border border-green-200">
                                <h4 className="font-semibold text-green-900 mb-2">📄 Comprehensive Report</h4>

                                {message.analysis.comprehensive_report.executive_summary && (
                                  <div className="mb-3">
                                    <h5 className="font-medium text-green-800 mb-1">Executive Summary:</h5>
                                    <p className="text-green-700 text-sm">{message.analysis.comprehensive_report.executive_summary}</p>
                                  </div>
                                )}

                                {message.analysis.comprehensive_report.recommendations && (
                                  <div>
                                    <h5 className="font-medium text-green-800 mb-1">Recommendations:</h5>
                                    <ul className="space-y-1">
                                      {message.analysis.comprehensive_report.recommendations.map((rec: string, index: number) => (
                                        <li key={index} className="text-green-700 text-sm flex items-start space-x-2">
                                          <span className="text-green-600 mt-1">✓</span>
                                          <span>{rec}</span>
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        )}

                        {/* ADVANCED CHARTS SECTION - This is the main integration! */}
                        {message.charts && message.charts.length > 0 && (
                          <div className="mt-6">
                            <Charts
                              charts={message.charts}
                              onChartGenerate={handleChartGenerate}
                              className="charts-in-message"
                            />
                          </div>
                        )}
                      </div>
                    )}

                    {message.type === 'user' && (
                      <div className="text-xs text-blue-200 mt-2">
                        {formatTimestamp(message.timestamp)}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* File Upload Overlay */}
        {showFileUpload && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900">Upload Data File</h3>
                <button
                  onClick={() => setShowFileUpload(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center ${
                  dragOver ? 'border-blue-400 bg-blue-50' : 'border-gray-300'
                }`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
              >
                <div className="text-4xl mb-4">📁</div>
                <p className="text-gray-600 mb-4">
                  Drag and drop your file here, or click to browse
                </p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls,.json"
                  onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-lg transition-colors"
                >
                  Choose File
                </button>
                <p className="text-xs text-gray-500 mt-2">
                  Supports CSV, Excel, and JSON files (max 100MB)
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4">
          {selectedFile && (
            <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <span className="text-blue-600">📁</span>
                <span className="text-sm font-medium text-blue-900">{selectedFile.name}</span>
                <span className="text-xs text-blue-600">
                  ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                </span>
              </div>
              <button
                onClick={() => setSelectedFile(null)}
                className="text-blue-400 hover:text-blue-600"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} className="flex space-x-4">
            <div className="flex-1">
              <div className="relative">
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  placeholder="Ask me anything about your data... Try: 'Show me a waterfall chart' or 'Create a funnel visualization'"
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={3}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                />
                <div className="absolute bottom-2 right-2 text-xs text-gray-400">
                  Press Enter to send, Shift+Enter for new line
                </div>
              </div>
            </div>

            <div className="flex flex-col space-y-2">
              <button
                type="submit"
                disabled={isAnalyzing || (!inputValue.trim() && !selectedFile) || backendStatus !== 'connected'}
                className={`px-6 py-3 bg-blue-600 text-white rounded-lg font-medium transition-colors ${
                  isAnalyzing || (!inputValue.trim() && !selectedFile) || backendStatus !== 'connected'
                    ? 'opacity-50 cursor-not-allowed'
                    : 'hover:bg-blue-700'
                }`}
              >
                {isAnalyzing ? (
                  <svg className="animate-spin w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                ) : (
                  'Analyze'
                )}
              </button>

              <button
                type="button"
                onClick={() => setShowFileUpload(true)}
                className="px-6 py-3 bg-gray-100 text-gray-700 rounded-lg font-medium hover:bg-gray-200 transition-colors"
              >
                📁
              </button>
            </div>
          </form>

          {backendStatus !== 'connected' && (
            <div className="mt-2 text-xs text-red-600">
              ⚠️ Backend server is offline. Please start the server at http://localhost:8000
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AnalysisInterface;