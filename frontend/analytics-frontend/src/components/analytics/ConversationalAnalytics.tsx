// Enhanced ConversationalAnalytics.tsx with proper backend integration
import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { analyticsAPI, extractErrorMessage, AnalysisResponse } from '../../services/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  analysis?: AnalysisResult;
  suggestions?: string[];
}

interface AnalysisResult {
  summary: string;
  insights: string[];
  charts: ChartData[];
  data: any[];
  metrics?: {
    [key: string]: number | string;
  };
}

interface ChartData {
  type: 'line' | 'bar' | 'pie';
  title: string;
  data: any[];
  xAxis?: string;
  yAxis?: string;
}

const ConversationalAnalytics: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [fileName, setFileName] = useState<string>('');
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [systemInfo, setSystemInfo] = useState<any>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Add welcome message and check backend
    setMessages([
      {
        id: '1',
        type: 'assistant',
        content: "👋 Hi! I'm your AI analytics assistant. Upload your data and ask me questions like:\n\n• What's the total sales by region in 2024?\n• Which month in 2024 had the highest sales?\n• Show me trends by category\n• Create a chart of sales by region\n\nI'll analyze your data using advanced AI!",
        timestamp: new Date(),
        suggestions: [
          "Upload a CSV file to get started",
          "total sales by region in 2024",
          "month in 2024 with highest sales",
          "show me sales trends over time"
        ]
      }
    ]);

    checkBackendStatus();
  }, []);

  const checkBackendStatus = async () => {
    try {
      console.log('🔍 Checking backend status...');
      setBackendStatus('checking');

      const [healthCheck, systemStatus, openaiTest] = await Promise.allSettled([
        analyticsAPI.healthCheck(),
        analyticsAPI.getSystemStatus(),
        analyticsAPI.testOpenAI()
      ]);

      let status: 'connected' | 'disconnected' = 'disconnected';
      let info: any = {};

      if (healthCheck.status === 'fulfilled' && healthCheck.value.status === 'healthy') {
        status = 'connected';
        console.log('✅ Backend is healthy');
      }

      if (systemStatus.status === 'fulfilled') {
        info.system = systemStatus.value;
      }

      if (openaiTest.status === 'fulfilled') {
        info.openai = openaiTest.value;
      }

      setBackendStatus(status);
      setSystemInfo(info);

    } catch (error) {
      console.error('❌ Backend health check failed:', error);
      setBackendStatus('disconnected');
    }
  };

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

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

    setFileName(file.name);
    setUploadedFile(file);
    setError(null);

    // Add file upload confirmation message
    const uploadMessage: Message = {
      id: Date.now().toString(),
      type: 'assistant',
      content: `✅ **File uploaded successfully!**\n\n📊 **${file.name}**\n• ${Math.round(file.size / 1024)} KB\n• Ready for analysis\n\nNow you can ask me questions about your data! Try:\n• "total sales by region in 2024"\n• "month in 2024 with highest sales"\n• "show me trends over time"`,
      timestamp: new Date(),
      suggestions: [
        "total sales by region in 2024",
        "month in 2024 with highest sales",
        "show me trends over time",
        "what are the top performing categories?"
      ]
    };

    setMessages(prev => [...prev, uploadMessage]);
  };

  const analyzeWithBackend = async (question: string, file: File): Promise<AnalysisResult> => {
    console.log('🚀 Analyzing with backend:', {
      question: question.substring(0, 100),
      fileName: file.name,
      fileSize: file.size
    });

    try {
      // Use the enhanced API method
      const result: AnalysisResponse = await analyticsAPI.analyze({
        prompt: question,
        file: file,
        use_adaptive: true,
        include_charts: true,
        auto_discover: false,
        domain: 'general'
      });

      console.log('✅ Backend analysis successful:', {
        status: result.status,
        hasAnalysis: !!result.analysis,
        insightsCount: result.analysis?.insights?.length || 0,
        chartsCount: result.chart_intelligence?.chart_count || 0
      });

      if (result.status !== 'success') {
        throw new Error(result.analysis?.metadata?.error || 'Analysis failed');
      }

      // Transform backend response to frontend format
      const transformedResult: AnalysisResult = {
        summary: result.analysis?.summary || "Analysis completed successfully",
        insights: result.analysis?.insights || [],
        charts: (result.chart_intelligence?.suggested_charts || []).map((chart: any, index: number) => ({
          type: (chart.chart_type || chart.type || 'bar') as 'line' | 'bar' | 'pie',
          title: chart.title || `Chart ${index + 1}`,
          data: chart.data || [],
          xAxis: chart.x_axis,
          yAxis: chart.y_axis
        })),
        data: result.analysis?.data || [],
        metrics: {
          'Analysis Type': result.analysis?.type || 'general',
          'Data Points': result.analysis?.data?.length || 0,
          'Processing Time': result.performance?.total_time_ms ? `${result.performance.total_time_ms}ms` : 'N/A',
          'Confidence': result.query_interpretation?.confidence ? `${Math.round(result.query_interpretation.confidence * 100)}%` : 'N/A',
          'Method': result.system_info?.method || 'enhanced_analytics'
        }
      };

      // Add additional metrics if available
      if (result.performance?.data_stats) {
        transformedResult.metrics!['Rows Processed'] = result.performance.data_stats.rows || result.performance.data_stats.rows_processed || 0;
        transformedResult.metrics!['Columns'] = result.performance.data_stats.columns || result.performance.data_stats.columns_analyzed || 0;
      }

      return transformedResult;

    } catch (error) {
      console.error('💥 Analysis failed:', error);
      throw error;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!currentQuestion.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: currentQuestion,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentQuestion('');
    setIsAnalyzing(true);
    setError(null);

    try {
      if (!uploadedFile) {
        throw new Error('Please upload a data file first before asking questions.');
      }

      if (backendStatus !== 'connected') {
        throw new Error('Backend is not available. Please ensure the server is running at http://localhost:8000');
      }

      const analysis = await analyzeWithBackend(currentQuestion, uploadedFile);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: analysis.summary,
        timestamp: new Date(),
        analysis,
        suggestions: [
          "Show me more details about this",
          "Break this down by time period",
          "What other patterns can you find?",
          "Create a different visualization"
        ]
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      console.error('💥 Analysis failed:', error);

      const errorMessage = extractErrorMessage(error);
      setError(errorMessage);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: `❌ **Analysis Error**\n\n${errorMessage}\n\n**Suggestions:**\n• Make sure the backend server is running\n• Check your data file format (CSV, Excel)\n• Try a simpler question first\n• Verify your data has the columns you're asking about`,
        timestamp: new Date(),
        suggestions: [
          "Check backend server status",
          "Try a simpler question",
          "Upload a different file",
          "Retry the same question"
        ]
      };

      setMessages(prev => [...prev, assistantMessage]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderChart = (chart: ChartData) => {
    const { type, title, data, xAxis, yAxis } = chart;

    if (!data || data.length === 0) {
      return (
        <div className="mb-6">
          <h4 className="text-lg font-semibold mb-3">{title}</h4>
          <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">No data available for visualization</p>
          </div>
        </div>
      );
    }

    switch (type) {
      case 'line':
        return (
          <div className="mb-6">
            <h4 className="text-lg font-semibold mb-3">{title}</h4>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey={xAxis || 'name'} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey={yAxis || 'value'} stroke="#8884d8" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        );

      case 'pie':
        return (
          <div className="mb-6">
            <h4 className="text-lg font-semibold mb-3">{title}</h4>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {data.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>
        );

      default: // bar
        return (
          <div className="mb-6">
            <h4 className="text-lg font-semibold mb-3">{title}</h4>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
    }
  };

  const quickQuestions = [
    "total sales by region in 2024",
    "month in 2024 with highest sales",
    "show me sales trends over time",
    "what are the top performing products?",
    "compare performance by category",
    "identify any unusual patterns"
  ];

  const handleBackToHome = () => {
    window.location.href = '/';
  };

  const getStatusIndicator = () => {
    switch (backendStatus) {
      case 'checking':
        return { color: 'bg-yellow-500', text: 'Checking...', pulse: true };
      case 'connected':
        return { color: 'bg-green-500', text: 'AI Ready', pulse: false };
      case 'disconnected':
        return { color: 'bg-red-500', text: 'AI Offline', pulse: false };
    }
  };

  const statusInfo = getStatusIndicator();

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={handleBackToHome}
                className="text-gray-600 hover:text-gray-900 transition-colors"
              >
                ← Back to Home
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900 flex items-center space-x-2">
                  <span>💬</span>
                  <span>Chat with Your Data</span>
                </h1>
                <p className="text-gray-600 text-sm">
                  Upload data and ask questions in natural language
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 ${statusInfo.color} rounded-full ${statusInfo.pulse ? 'animate-pulse' : ''}`}></div>
                <span className="text-sm text-gray-600">{statusInfo.text}</span>
              </div>

              {/* System Info */}
              {systemInfo && (
                <div className="text-xs text-gray-500">
                  {systemInfo.openai?.status === 'success' && (
                    <span className="text-green-600">🤖 OpenAI Ready</span>
                  )}
                  {systemInfo.system?.smart_engine?.available && (
                    <span className="ml-2 text-blue-600">⚡ Smart Engine</span>
                  )}
                </div>
              )}

              {backendStatus === 'disconnected' && (
                <button
                  onClick={checkBackendStatus}
                  className="text-xs bg-blue-500 text-white px-3 py-1 rounded hover:bg-blue-600"
                >
                  Retry
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto p-6">
        <div className="bg-white rounded-lg shadow-lg">
          {/* File Upload Section */}
          <div className="border-b border-gray-200 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900 mb-2">
                  Upload Your Data
                </h2>
                <p className="text-gray-600 text-sm">
                  Upload a CSV or Excel file to start analyzing your data with AI
                </p>
              </div>

              <div className="flex items-center space-x-4">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls,.json"
                  onChange={handleFileUpload}
                  className="block text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
                {fileName && (
                  <div className="flex items-center space-x-2 text-sm text-green-600">
                    <span>✅</span>
                    <span>{fileName}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Chat Messages */}
          <div className="h-96 overflow-y-auto p-6 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-3xl p-4 rounded-lg ${
                    message.type === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-100 text-gray-900'
                  }`}
                >
                  <div className="whitespace-pre-wrap">{message.content}</div>

                  {/* Analysis Results */}
                  {message.analysis && (
                    <div className="mt-4 space-y-4">
                      {/* Key Insights */}
                      {message.analysis.insights.length > 0 && (
                        <div className="bg-white p-4 rounded-lg border">
                          <h3 className="font-semibold text-gray-900 mb-2">✨ Key Insights</h3>
                          <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                            {message.analysis.insights.map((insight, index) => (
                              <li key={index}>{insight}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Metrics */}
                      {message.analysis.metrics && (
                        <div className="bg-white p-4 rounded-lg border">
                          <h3 className="font-semibold text-gray-900 mb-2">📊 Key Metrics</h3>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                            {Object.entries(message.analysis.metrics).map(([key, value]) => (
                              <div key={key} className="text-center">
                                <div className="text-lg font-bold text-blue-600">
                                  {typeof value === 'number' ? value.toLocaleString() : value}
                                </div>
                                <div className="text-sm text-gray-600">{key}</div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Charts */}
                      {message.analysis.charts.length > 0 && (
                        <div className="bg-white p-4 rounded-lg border">
                          <h3 className="font-semibold text-gray-900 mb-4">📈 Visualizations</h3>
                          {message.analysis.charts.map((chart, index) => (
                            <div key={index}>
                              {renderChart(chart)}
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Data Preview */}
                      {message.analysis.data && message.analysis.data.length > 0 && (
                        <div className="bg-white p-4 rounded-lg border">
                          <h3 className="font-semibold text-gray-900 mb-2">📋 Data Preview</h3>
                          <div className="text-sm text-gray-600 mb-2">
                            Showing first {Math.min(5, message.analysis.data.length)} rows of {message.analysis.data.length} total
                          </div>
                          <div className="overflow-x-auto">
                            <table className="min-w-full text-xs">
                              <thead>
                                <tr className="bg-gray-50">
                                  {Object.keys(message.analysis.data[0] || {}).map((key) => (
                                    <th key={key} className="px-2 py-1 text-left font-medium text-gray-700">
                                      {key}
                                    </th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                {message.analysis.data.slice(0, 5).map((row, index) => (
                                  <tr key={index} className="border-t">
                                    {Object.values(row).map((value, i) => (
                                      <td key={i} className="px-2 py-1 text-gray-900">
                                        {typeof value === 'number' ? value.toLocaleString() : String(value)}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Suggestions */}
                  {message.suggestions && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {message.suggestions.map((suggestion, index) => (
                        <button
                          key={index}
                          onClick={() => setCurrentQuestion(suggestion)}
                          className="px-3 py-1 text-xs bg-white bg-opacity-20 rounded-full hover:bg-opacity-30 transition-colors"
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}

                  <div className="text-xs opacity-75 mt-2">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}

            {isAnalyzing && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-900 p-4 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-500 border-t-transparent"></div>
                    <span>AI is analyzing your data...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Error Display */}
          {error && (
            <div className="border-t border-gray-200 p-4 bg-red-50">
              <div className="flex items-start space-x-2">
                <span className="text-red-500">⚠️</span>
                <div className="flex-1">
                  <h4 className="text-red-800 font-medium">Analysis Error</h4>
                  <p className="text-red-700 text-sm mt-1">{error}</p>
                </div>
                <button
                  onClick={() => setError(null)}
                  className="text-red-500 hover:text-red-700"
                >
                  ✕
                </button>
              </div>
            </div>
          )}

          {/* Quick Questions */}
          {uploadedFile && (
            <div className="border-t border-gray-200 p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">💡 Try These Questions</h3>
              <div className="flex flex-wrap gap-2">
                {quickQuestions.map((question, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentQuestion(question)}
                    className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded-full transition-colors"
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input Form */}
          <div className="border-t border-gray-200 p-6">
            <form onSubmit={handleSubmit} className="flex space-x-4">
              <input
                type="text"
                value={currentQuestion}
                onChange={(e) => setCurrentQuestion(e.target.value)}
                placeholder={uploadedFile ? "Ask a question about your data..." : "Upload data first, then ask questions..."}
                disabled={isAnalyzing || !uploadedFile || backendStatus !== 'connected'}
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={isAnalyzing || !currentQuestion.trim() || !uploadedFile || backendStatus !== 'connected'}
                className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 text-white px-6 py-3 rounded-lg transition-colors font-medium"
              >
                {isAnalyzing ? (
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                    <span>Analyzing...</span>
                  </div>
                ) : (
                  'Ask Question'
                )}
              </button>
            </form>

            {backendStatus === 'disconnected' && (
              <div className="mt-2 text-xs text-red-600">
                ⚠️ Backend server is offline. Please start the server at http://localhost:8000
              </div>
            )}

            {systemInfo?.openai?.status === 'error' && (
              <div className="mt-2 text-xs text-yellow-600">
                ⚠️ OpenAI not configured. Smart features limited.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConversationalAnalytics;