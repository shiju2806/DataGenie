import React, { useState, useRef, useEffect } from 'react';
import { analyticsAPI } from '../../services/api';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  analysis?: any;
  charts?: any[];
  loading?: boolean;
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

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const exampleQuestions = [
    "What are the trends in our sales data?",
    "Show me customer segmentation analysis",
    "Which products have the highest profit margins?",
    "What's the correlation between marketing spend and revenue?",
    "Identify outliers in the dataset",
    "Predict next quarter's performance"
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    loadSessions();
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadSessions = () => {
    try {
      const saved = localStorage.getItem('analysis_sessions');
      if (saved) {
        const parsedSessions = JSON.parse(saved);
        // Convert timestamp strings back to Date objects
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

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() && !selectedFile) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim() || `Uploaded file: ${selectedFile?.name}`,
      timestamp: new Date()
    };

    const loadingMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: 'Analyzing your request...',
      timestamp: new Date(),
      loading: true
    };

    setMessages(prev => [...prev, userMessage, loadingMessage]);
    setInputValue('');
    setIsAnalyzing(true);

    try {
      // Simulate analysis with mock response
      await new Promise(resolve => setTimeout(resolve, 2000 + Math.random() * 2000));

      // Generate mock analysis response
      const mockAnalysis = {
        type: 'trend_analysis',
        summary: `I've analyzed your request: "${userMessage.content}". Here are the key insights I discovered:`,
        insights: [
          'Data shows a positive upward trend over the last 6 months',
          'Peak performance occurs during Q2 and Q4 periods',
          'Customer retention rate has improved by 15% year-over-year',
          'Revenue growth is primarily driven by new customer acquisition',
          'Data quality is excellent with <2% missing values'
        ],
        data: Array.from({length: 100}, (_, i) => ({
          date: new Date(2024, 0, i).toISOString().split('T')[0],
          value: Math.random() * 1000 + 500,
          category: ['A', 'B', 'C'][i % 3]
        }))
      };

      const mockCharts = [
        {
          type: 'line',
          title: 'Trend Analysis',
          description: 'Time series showing performance over time'
        },
        {
          type: 'bar',
          title: 'Category Comparison',
          description: 'Comparative analysis across different segments'
        }
      ];

      const assistantMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: mockAnalysis.summary,
        timestamp: new Date(),
        analysis: mockAnalysis,
        charts: mockCharts
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

    } catch (error) {
      console.error('Analysis failed:', error);

      const errorMessage: Message = {
        id: (Date.now() + 2).toString(),
        type: 'assistant',
        content: 'Sorry, I encountered an error while analyzing your request. Please try again or check your data format.',
        timestamp: new Date()
      };

      setMessages(prev => prev.slice(0, -1).concat(errorMessage));
    } finally {
      setIsAnalyzing(false);
      setSelectedFile(null);
      setShowFileUpload(false);
    }
  };

  const handleFileSelect = (file: File) => {
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
            className="w-full btn-primary text-left"
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
      </div>

      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">AI Analysis Interface</h1>
              <p className="text-gray-600">Ask questions about your data in natural language</p>
            </div>

            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 px-3 py-1 bg-green-50 rounded-full">
                <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                <span className="text-sm text-green-700 font-medium">AI Ready</span>
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
                  Welcome to AI Analysis
                </h2>
                <p className="text-gray-600 mb-8">
                  Ask questions about your data in plain English. I can analyze trends, create visualizations, and provide insights.
                </p>
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

                  <button className="p-4 text-center border-2 border-dashed border-gray-300 rounded-lg hover:border-green-400 hover:bg-green-50 transition-colors">
                    <div className="text-2xl mb-2">🔗</div>
                    <div className="font-medium text-gray-900">Connect Source</div>
                    <div className="text-sm text-gray-500">Database, API</div>
                  </button>

                  <button className="p-4 text-center border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-colors">
                    <div className="text-2xl mb-2">📊</div>
                    <div className="font-medium text-gray-900">View Sample</div>
                    <div className="text-sm text-gray-500">Demo data</div>
                  </button>
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
                        : 'bg-white border border-gray-200'
                    }`}
                  >
                    {message.type === 'assistant' && (
                      <div className="flex items-center space-x-2 mb-2">
                        <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center">
                          <span className="text-blue-600 text-sm">🧠</span>
                        </div>
                        <span className="text-sm font-medium text-gray-700">AI Assistant</span>
                        <span className="text-xs text-gray-500">{formatTimestamp(message.timestamp)}</span>
                      </div>
                    )}

                    {message.loading ? (
                      <div className="flex items-center space-x-2">
                        <svg className="animate-spin w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                        </svg>
                        <span className="text-gray-600">Analyzing...</span>
                      </div>
                    ) : (
                      <div>
                        <p className="whitespace-pre-wrap">{message.content}</p>

                        {/* Analysis Results */}
                        {message.analysis && (
                          <div className="mt-4 space-y-4">
                            {message.analysis.insights && message.analysis.insights.length > 0 && (
                              <div className="bg-blue-50 p-4 rounded-lg">
                                <h4 className="font-semibold text-blue-900 mb-2">Key Insights</h4>
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

                            {message.analysis.data && (
                              <div className="bg-gray-50 p-4 rounded-lg">
                                <h4 className="font-semibold text-gray-900 mb-2">Data Summary</h4>
                                <div className="text-sm text-gray-700">
                                  <p>Records: {message.analysis.data.length}</p>
                                  <p>Type: {message.analysis.type}</p>
                                </div>
                              </div>
                            )}
                          </div>
                        )}

                        {/* Chart Suggestions */}
                        {message.charts && message.charts.length > 0 && (
                          <div className="mt-4">
                            <h4 className="font-semibold text-gray-900 mb-3">Suggested Visualizations</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              {message.charts.map((chart: any, index: number) => (
                                <div key={index} className="p-3 bg-gray-50 rounded-lg border">
                                  <div className="flex items-center space-x-2 mb-2">
                                    <span className="text-lg">📈</span>
                                    <span className="font-medium text-gray-900">{chart.title || 'Chart'}</span>
                                  </div>
                                  <p className="text-sm text-gray-600">{chart.description || 'Visualization ready'}</p>
                                  <button className="mt-2 text-sm text-blue-600 hover:text-blue-800 font-medium">
                                    Generate Chart
                                  </button>
                                </div>
                              ))}
                            </div>
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
                  className="btn-primary"
                >
                  Choose File
                </button>
                <p className="text-xs text-gray-500 mt-2">
                  Supports CSV, Excel, and JSON files
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
                  placeholder="Ask me anything about your data..."
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
                disabled={isAnalyzing || (!inputValue.trim() && !selectedFile)}
                className={`px-6 py-3 bg-blue-600 text-white rounded-lg font-medium transition-colors ${
                  isAnalyzing || (!inputValue.trim() && !selectedFile)
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
        </div>
      </div>
    </div>
  );
};

export default AnalysisInterface;