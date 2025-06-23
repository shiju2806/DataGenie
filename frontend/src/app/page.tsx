'use client'

import React, { useState, useRef, useEffect } from 'react'
import {
  MessageCircle, Upload, BarChart3, TrendingUp, Users, Calculator,
  Download, Activity, Brain, Database, Settings, FileText, Zap,
  CheckCircle, AlertCircle, Info, Sparkles, Target, PieChart,
  LineChart, Globe, Cpu, Building, Briefcase
} from 'lucide-react'

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  data?: AnalysisResult | UploadResult | ReportResult | null
  timestamp: Date
}

interface SystemStatus {
  is_initialized: boolean
  components: Record<string, boolean>
  datasets: {
    count: number
    names: string[]
    total_records: number
  }
  learning_profile: {
    available: boolean
    primary_domain: string
    metrics_discovered: number
    dimensions_discovered: number
  }
  enhanced_analytics_part4b?: {
    available: boolean
    comprehensive_reporting: boolean
  }
}

interface MathematicalAnalysis {
  method_used: string
  results: Record<string, number | string>
  interpretation: string
  confidence: number
  assumptions_met?: Record<string, boolean>
  warnings?: string[]
  recommendations?: string[]
}

interface ConfidenceAssessment {
  overall_confidence: number
  confidence_factors?: Record<string, number>
  recommendation: string
  explanation?: string
}

interface AnalysisResult {
  analysis_type: string
  summary: string
  data: Record<string, unknown>[]
  insights: string[]
  metadata?: {
    processing_time_ms?: number
    interpretation?: Record<string, unknown>
    [key: string]: unknown
  }
  concept_explanations?: Record<string, string>
  method_justification?: string
  mathematical_analysis?: MathematicalAnalysis
  confidence_assessment?: ConfidenceAssessment
}

interface UploadResult {
  status: string
  message: string
  profile?: {
    filename: string
    rows: number
    columns: number
    suggested_queries?: string[]
    numeric_columns?: string[]
    suggested_domain?: string
  }
}

interface ReportResult {
  status: string
  report?: {
    title: string
    executive_summary: string
    key_findings: string[]
    recommendations: string[]
    confidence_score: number
    sections?: Array<{
      title: string
      content: string | Record<string, unknown>
    }>
  }
  type?: string
}

export default function DataGenieAnalytics(): React.JSX.Element {
  const [activeTab, setActiveTab] = useState<string>('chat')
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState<string>('')
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [uploadedFiles, setUploadedFiles] = useState<UploadResult[]>([])
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [selectedDomain, setSelectedDomain] = useState<string>('auto')
  const [queryHistory, setQueryHistory] = useState<string[]>([])
  const [suggestions, setSuggestions] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Initialize system
  useEffect(() => {
    initializeSystem()
    loadQuerySuggestions()
  }, [])

  const initializeSystem = async (): Promise<void> => {
    try {
      const response = await fetch('http://localhost:8000/api/enhanced-system-status')
      const status = await response.json()
      setSystemStatus(status)

      // Initialize welcome message based on system capabilities
      const welcomeMessage: Message = {
        id: '1',
        type: 'assistant',
        content: generateWelcomeMessage(status),
        timestamp: new Date()
      }
      setMessages([welcomeMessage])
    } catch (error) {
      console.error('System initialization failed:', error)
      setMessages([{
        id: '1',
        type: 'assistant',
        content: "Welcome to DataGenie! I'm your analytics assistant. Upload your data or try asking questions.",
        timestamp: new Date()
      }])
    }
  }

  const generateWelcomeMessage = (status: SystemStatus): string => {
    const capabilities: string[] = []
    if (status.components?.mathematical_engine) capabilities.push("Mathematical Analysis")
    if (status.components?.knowledge_framework) capabilities.push("Domain Knowledge")
    if (status.components?.adaptive_interpreter) capabilities.push("Natural Language Processing")
    if (status.enhanced_analytics_part4b?.comprehensive_reporting) capabilities.push("Comprehensive Reporting")

    const domains: string = status.learning_profile?.primary_domain || "Multiple domains"
    const datasets: string = status.datasets?.names?.join(", ") || "sample datasets"

    return `🚀 Welcome to DataGenie Analytics Platform!

🧠 **Active Capabilities:** ${capabilities.join(", ")}
📊 **Available Data:** ${datasets}
🎯 **Primary Domain:** ${domains}
📈 **Metrics Discovered:** ${status.learning_profile?.metrics_discovered || 0}

I can analyze any industry data using natural language. Try asking:
• "Give me a comprehensive analysis of the data"
• "Show me trends and patterns"
• "Analyze performance by segment"
• "Generate an executive report"

Upload your own data to get started with domain-specific insights!`
  }

  const loadQuerySuggestions = async (): Promise<void> => {
    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: "suggest queries" })
      })
      const result = await response.json()
      if (result.metadata?.interpretation?.suggested_queries) {
        setSuggestions(result.metadata.interpretation.suggested_queries)
      }
    } catch (error) {
      console.error('Failed to load suggestions:', error)
    }
  }

  const handleSendMessage = async (): Promise<void> => {
    if (!inputValue.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setQueryHistory(prev => [inputValue, ...prev.slice(0, 9)]) // Keep last 10
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: inputValue,
          domain: selectedDomain === 'auto' ? null : selectedDomain,
          use_uploaded_data: uploadedFiles.length > 0
        })
      })

      const result: AnalysisResult = await response.json()

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: result.summary,
        data: result,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "I encountered an error processing your request. Please check the system status and try again.",
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    }

    setIsLoading(false)
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
    const file = event.target.files?.[0]
    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData
      })

      const result: UploadResult = await response.json()
      setUploadedFiles(prev => [...prev, result])

      const uploadMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: `✅ Successfully uploaded ${result.profile?.filename}!

📊 **Dataset Profile:**
• ${result.profile?.rows} rows × ${result.profile?.columns} columns
• Domain: ${result.profile?.suggested_domain || 'Auto-detected'}
• Key metrics: ${result.profile?.numeric_columns?.slice(0, 3).join(", ") || "None detected"}

🎯 **Suggested queries:**
${result.profile?.suggested_queries?.slice(0, 3).map((q: string) => `• ${q}`).join('\n') || '• Analyze trends\n• Show summary statistics'}

You can now ask specific questions about your data!`,
        data: result,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, uploadMessage])

      // Refresh system status and suggestions
      setTimeout(() => {
        initializeSystem()
        loadQuerySuggestions()
      }, 1000)

    } catch (error) {
      console.error('Upload error:', error)
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: "Failed to upload file. Please check the file format (CSV/Excel) and try again.",
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    }
    setIsLoading(false)
  }

  const generateComprehensiveReport = async (query: string): Promise<void> => {
    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/comprehensive-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query || "Generate comprehensive analysis report",
          domain: selectedDomain === 'auto' ? null : selectedDomain
        })
      })

      const result: { status: string; report?: ReportResult['report']; message?: string } = await response.json()

      if (result.status === 'success' && result.report) {
        // Create download link
        const reportText = formatReportForDownload(result.report)
        const element = document.createElement('a')
        const file = new Blob([reportText], {type: 'text/plain'})
        element.href = URL.createObjectURL(file)
        element.download = `DataGenie_Report_${new Date().toISOString().split('T')[0]}.txt`
        document.body.appendChild(element)
        element.click()
        document.body.removeChild(element)

        const reportMessage: Message = {
          id: Date.now().toString(),
          type: 'assistant',
          content: `📄 **Comprehensive Report Generated**

${result.report.executive_summary}

**Key Findings (${result.report.key_findings?.length || 0}):**
${result.report.key_findings?.slice(0, 3).map((f: string) => `• ${f}`).join('\n') || 'None'}

**Report downloaded!** Check your downloads folder for the complete analysis.`,
          data: { report: result.report, type: 'comprehensive_report' } as ReportResult,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, reportMessage])
      } else {
        throw new Error(result.message || 'Report generation failed')
      }
    } catch (error) {
      console.error('Report generation error:', error)
      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: "Failed to generate comprehensive report. This feature may not be available in your current setup.",
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    }
    setIsLoading(false)
  }

  const formatReportForDownload = (report: NonNullable<ReportResult['report']>): string => {
    let content = `DATAGENIE COMPREHENSIVE ANALYSIS REPORT
Generated: ${new Date().toISOString()}

TITLE: ${report.title}

EXECUTIVE SUMMARY:
${report.executive_summary}

KEY FINDINGS:
${report.key_findings?.map((f: string, i: number) => `${i + 1}. ${f}`).join('\n') || 'None'}

RECOMMENDATIONS:
${report.recommendations?.map((r: string, i: number) => `${i + 1}. ${r}`).join('\n') || 'None'}

CONFIDENCE SCORE: ${(report.confidence_score * 100).toFixed(1)}%

DETAILED SECTIONS:
`

    report.sections?.forEach((section, i: number) => {
      content += `
${i + 1}. ${section.title.toUpperCase()}
${'-'.repeat(50)}
${typeof section.content === 'string' ? section.content : JSON.stringify(section.content, null, 2)}
`
    })

    return content
  }

  const renderVisualization = (data: AnalysisResult): React.ReactNode => {
    if (!data.data || data.data.length === 0) return null

    // Intelligent visualization based on analysis type
    if (data.analysis_type.includes('error')) {
      return (
        <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
            <span className="text-red-800 font-medium">Analysis Error</span>
          </div>
          <p className="text-red-700 mt-2">{data.summary}</p>
        </div>
      )
    }

    // Summary/Dashboard view
    if (data.analysis_type.includes('summary') || data.analysis_type.includes('dashboard')) {
      return (
        <div className="mt-4 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border">
          <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
            <PieChart className="h-5 w-5 mr-2 text-blue-600" />
            Data Overview
          </h4>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {data.data.slice(0, 8).map((item, index) => {
              const stringKey = Object.keys(item).find(key => typeof item[key] === 'string')
              const numberEntry = Object.entries(item).find(([, value]) => typeof value === 'number')

              return (
                <div key={index} className="bg-white p-3 rounded-lg shadow-sm">
                  <div className="text-sm text-gray-600 truncate">
                    {stringKey ? String(item[stringKey]) : `Item ${index + 1}`}
                  </div>
                  <div className="text-lg font-bold text-gray-900">
                    {numberEntry ? Number(numberEntry[1]).toLocaleString() : 'N/A'}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )
    }

    // Mathematical analysis results
    if (data.mathematical_analysis) {
      return (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
          <h4 className="font-semibold text-green-900 mb-3 flex items-center">
            <Calculator className="h-5 w-5 mr-2" />
            Mathematical Analysis: {data.mathematical_analysis.method_used?.replace('_', ' ')}
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-white p-3 rounded">
              <span className="text-sm text-gray-600">Confidence</span>
              <div className="text-xl font-bold text-green-600">
                {(data.mathematical_analysis.confidence * 100).toFixed(1)}%
              </div>
            </div>
            {Object.entries(data.mathematical_analysis.results || {}).map(([key, value]) => (
              <div key={key} className="bg-white p-3 rounded">
                <span className="text-sm text-gray-600 capitalize">{key.replace('_', ' ')}</span>
                <div className="text-lg font-bold text-gray-900">
                  {typeof value === 'number' ? value.toFixed(3) : String(value)}
                </div>
              </div>
            ))}
          </div>
          {data.mathematical_analysis.interpretation && (
            <div className="mt-3 p-3 bg-white rounded border-l-4 border-green-400">
              <p className="text-sm text-gray-700">{data.mathematical_analysis.interpretation}</p>
            </div>
          )}
        </div>
      )
    }

    // Generic data table for other analysis types
    const firstItem = data.data[0]
    if (!firstItem) return null

    const columns = Object.keys(firstItem).slice(0, 5)

    return (
      <div className="mt-4 p-4 bg-gray-50 rounded-lg">
        <h4 className="font-semibold text-gray-900 mb-3 flex items-center">
          <BarChart3 className="h-5 w-5 mr-2 text-blue-600" />
          Analysis Results ({data.data.length} records)
        </h4>
        <div className="overflow-x-auto">
          <table className="min-w-full bg-white rounded border">
            <thead className="bg-gray-100">
              <tr>
                {columns.map(key => (
                  <th key={key} className="px-4 py-2 text-left text-sm font-medium text-gray-700 capitalize">
                    {key.replace('_', ' ')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.data.slice(0, 10).map((row, index) => (
                <tr key={index} className="border-t">
                  {columns.map((col, i) => {
                    const value = row[col]
                    return (
                      <td key={i} className="px-4 py-2 text-sm text-gray-900">
                        {typeof value === 'number' ? value.toLocaleString() : String(value)}
                      </td>
                    )
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          {data.data.length > 10 && (
            <p className="text-sm text-gray-500 mt-2">
              Showing 10 of {data.data.length} results
            </p>
          )}
        </div>
      </div>
    )
  }

  const getDomainIcon = (domain: string): React.ReactNode => {
    switch (domain.toLowerCase()) {
      case 'insurance': return <Building className="h-4 w-4" />
      case 'banking': return <Briefcase className="h-4 w-4" />
      case 'technology': return <Cpu className="h-4 w-4" />
      default: return <Globe className="h-4 w-4" />
    }
  }

  const smartSuggestions: string[] = [
    "Give me a comprehensive analysis of all available data",
    "Show me key performance indicators and trends",
    "Analyze correlation patterns in the dataset",
    "Generate executive summary with actionable insights",
    "Compare performance across different segments",
    "Identify outliers and anomalies in the data",
    "What are the main drivers of performance?",
    "Show me predictive insights and forecasts"
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg mr-3">
                <Sparkles className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  DataGenie Analytics
                </h1>
                <p className="text-sm text-gray-500">Industry-Agnostic Intelligence Platform</p>
              </div>
            </div>

            {/* Status Indicators */}
            <div className="flex items-center space-x-4">
              {systemStatus && (
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    systemStatus.is_initialized ? 'bg-green-500' : 'bg-red-500'
                  }`}></div>
                  <span className="text-sm text-gray-600">
                    {systemStatus.datasets?.count || 0} datasets
                  </span>
                </div>
              )}

              {/* Domain Selector */}
              <select
                value={selectedDomain}
                onChange={(e) => setSelectedDomain(e.target.value)}
                className="text-sm border border-gray-300 rounded-lg px-3 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="auto">Auto-detect</option>
                <option value="insurance">Insurance</option>
                <option value="banking">Banking</option>
                <option value="technology">Technology</option>
                <option value="general">General</option>
              </select>

              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="flex items-center px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all transform hover:scale-105 shadow-lg"
              >
                <Upload className="h-4 w-4 mr-2" />
                Upload Data
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* System Status Card */}
            {systemStatus && (
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <Activity className="h-5 w-5 mr-2 text-green-600" />
                  System Status
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Core Engine</span>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">AI Processing</span>
                    {systemStatus.components?.adaptive_interpreter ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-yellow-500" />
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Advanced Reports</span>
                    {systemStatus.enhanced_analytics_part4b?.comprehensive_reporting ? (
                      <CheckCircle className="h-4 w-4 text-green-500" />
                    ) : (
                      <AlertCircle className="h-4 w-4 text-yellow-500" />
                    )}
                  </div>

                  {systemStatus.learning_profile?.available && (
                    <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                      <div className="flex items-center text-blue-800 mb-2">
                        {getDomainIcon(systemStatus.learning_profile.primary_domain)}
                        <span className="ml-2 font-medium capitalize">
                          {systemStatus.learning_profile.primary_domain} Domain
                        </span>
                      </div>
                      <div className="text-sm text-blue-600">
                        {systemStatus.learning_profile.metrics_discovered} metrics • {systemStatus.learning_profile.dimensions_discovered} dimensions
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Smart Suggestions */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Brain className="h-5 w-5 mr-2 text-purple-600" />
                Smart Suggestions
              </h3>
              <div className="space-y-2">
                {(suggestions.length > 0 ? suggestions : smartSuggestions).slice(0, 6).map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => setInputValue(suggestion)}
                    className="w-full text-left p-3 text-sm bg-gray-50 hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 rounded-lg transition-all hover:shadow-md"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>

            {/* Quick Actions */}
            <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                <Zap className="h-5 w-5 mr-2 text-yellow-600" />
                Quick Actions
              </h3>
              <div className="space-y-3">
                <button
                  onClick={() => generateComprehensiveReport("")}
                  className="w-full flex items-center p-3 bg-gradient-to-r from-green-50 to-green-100 hover:from-green-100 hover:to-green-200 rounded-lg transition-all"
                >
                  <FileText className="h-4 w-4 mr-2 text-green-600" />
                  <span className="text-green-800 font-medium">Generate Report</span>
                </button>

                <button
                  onClick={() => setInputValue("Show me dashboard overview")}
                  className="w-full flex items-center p-3 bg-gradient-to-r from-blue-50 to-blue-100 hover:from-blue-100 hover:to-blue-200 rounded-lg transition-all"
                >
                  <TrendingUp className="h-4 w-4 mr-2 text-blue-600" />
                  <span className="text-blue-800 font-medium">Dashboard View</span>
                </button>
              </div>
            </div>

            {/* Recent Uploads */}
            {uploadedFiles.length > 0 && (
              <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-100">
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <Database className="h-5 w-5 mr-2 text-indigo-600" />
                  Uploaded Data
                </h3>
                <div className="space-y-2">
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="p-3 bg-indigo-50 rounded-lg">
                      <div className="font-medium text-indigo-900 text-sm">
                        {file.profile?.filename}
                      </div>
                      <div className="text-xs text-indigo-600">
                        {file.profile?.rows} rows • {file.profile?.columns} columns
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Main Chat Interface */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-xl shadow-lg h-[700px] flex flex-col border border-gray-100">
              {/* Chat Header */}
              <div className="p-6 border-b border-gray-100">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <MessageCircle className="h-6 w-6 text-blue-600 mr-3" />
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">Analytics Assistant</h2>
                      <p className="text-sm text-gray-500">Ask anything about your data in natural language</p>
                    </div>
                  </div>
                  <button
                    onClick={() => generateComprehensiveReport("")}
                    disabled={isLoading}
                    className="flex items-center px-4 py-2 bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-lg hover:from-purple-700 hover:to-pink-700 transition-all disabled:opacity-50"
                  >
                    <Download className="h-4 w-4 mr-2" />
                    Export Report
                  </button>
                </div>
              </div>

              {/* Messages Area */}
              <div className="flex-1 overflow-y-auto p-6 space-y-6">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-xl p-4 shadow-lg ${
                        message.type === 'user'
                          ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                          : 'bg-white text-gray-900 border border-gray-100'
                      }`}
                    >
                      <div className="whitespace-pre-wrap">{message.content}</div>

                      {/* Render Concept Explanations */}
                      {message.data && 'concept_explanations' in message.data && message.data.concept_explanations && Object.keys(message.data.concept_explanations).length > 0 && (
                        <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <h5 className="font-medium text-blue-900 mb-2 flex items-center">
                            <Info className="h-4 w-4 mr-2" />
                            Concept Explanations
                          </h5>
                          <div className="space-y-2">
                            {Object.entries(message.data.concept_explanations).map(([term, explanation]) => (
                              <div key={term} className="text-sm">
                                <span className="font-medium text-blue-800">{term}:</span>
                                <span className="text-blue-700 ml-2">{String(explanation)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Render Method Justification */}
                      {message.data && 'method_justification' in message.data && message.data.method_justification && (
                        <div className="mt-3 p-3 bg-purple-50 rounded-lg border border-purple-200">
                          <h5 className="font-medium text-purple-900 mb-1 flex items-center">
                            <Target className="h-4 w-4 mr-2" />
                            Method Selection
                          </h5>
                          <p className="text-sm text-purple-700">{String(message.data.method_justification)}</p>
                        </div>
                      )}

                      {/* Render Advanced Insights */}
                      {message.data && 'insights' in message.data && message.data.insights && message.data.insights.length > 0 && (
                        <div className="mt-4 space-y-2">
                          <h5 className="font-medium text-gray-900 flex items-center">
                            <Sparkles className="h-4 w-4 mr-2 text-yellow-500" />
                            AI Insights ({message.data.insights.length})
                          </h5>
                          {message.data.insights.slice(0, 8).map((insight: string, index: number) => (
                            <div key={index} className={`text-sm rounded-lg p-3 border-l-4 ${
                              insight.includes('⚠️') || insight.includes('🚨') ? 'bg-red-50 border-red-400 text-red-800' :
                              insight.includes('✅') || insight.includes('💡') ? 'bg-green-50 border-green-400 text-green-800' :
                              insight.includes('📊') || insight.includes('🔍') ? 'bg-blue-50 border-blue-400 text-blue-800' :
                              'bg-gray-50 border-gray-400 text-gray-800'
                            }`}>
                              <div className="flex items-start">
                                <span className="flex-shrink-0 text-lg mr-2">
                                  {insight.match(/[🔍📊💡⚠️🚨✅📈📉🎯💼🔧]/)?.[0] || '💡'}
                                </span>
                                <span className="font-medium">{insight.replace(/[🔍📊💡⚠️🚨✅📈📉🎯💼🔧]/g, '').trim()}</span>
                              </div>
                            </div>
                          ))}
                          {message.data.insights.length > 8 && (
                            <div className="text-sm text-gray-500 italic">
                              ... and {message.data.insights.length - 8} more insights
                            </div>
                          )}
                        </div>
                      )}

                      {/* Render Visualization */}
                      {message.data && 'data' in message.data && renderVisualization(message.data as AnalysisResult)}

                      {/* Confidence Assessment */}
                      {message.data && 'confidence_assessment' in message.data && message.data.confidence_assessment && (
                        <div className="mt-3 p-3 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg border">
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-gray-700">Analysis Confidence</span>
                            <div className="flex items-center">
                              <div className={`w-2 h-2 rounded-full mr-2 ${
                                message.data.confidence_assessment.overall_confidence > 0.8 ? 'bg-green-500' :
                                message.data.confidence_assessment.overall_confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                              }`}></div>
                              <span className="text-sm font-bold">
                                {(message.data.confidence_assessment.overall_confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>
                          <div className="mt-1">
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  message.data.confidence_assessment.overall_confidence > 0.8 ? 'bg-green-500' :
                                  message.data.confidence_assessment.overall_confidence > 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                                style={{ width: `${message.data.confidence_assessment.overall_confidence * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      )}

                      <div className="text-xs opacity-70 mt-3 flex items-center justify-between">
                        <span>{message.timestamp.toLocaleTimeString()}</span>
                        {message.data && 'metadata' in message.data && message.data.metadata?.processing_time_ms && (
                          <span className="flex items-center">
                            <Cpu className="h-3 w-3 mr-1" />
                            {Number(message.data.metadata.processing_time_ms).toFixed(0)}ms
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="flex justify-start">
                    <div className="bg-white rounded-xl p-4 shadow-lg border border-gray-100 max-w-md">
                      <div className="flex items-center space-x-3">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-purple-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                          <div className="w-2 h-2 bg-pink-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                        </div>
                        <span className="text-gray-600 font-medium">Analyzing your data with AI...</span>
                      </div>
                      <div className="mt-2 text-xs text-gray-500">
                        Processing through mathematical engine, knowledge framework, and domain expertise...
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Input Area */}
              <div className="border-t border-gray-100 p-4 bg-gray-50">
                {/* Query History */}
                {queryHistory.length > 0 && (
                  <div className="mb-3">
                    <div className="flex items-center mb-2">
                      <span className="text-xs text-gray-500 mr-2">Recent:</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {queryHistory.slice(0, 3).map((query, index) => (
                        <button
                          key={index}
                          onClick={() => setInputValue(query)}
                          className="text-xs px-2 py-1 bg-white border border-gray-200 rounded-full hover:bg-gray-50 transition-colors truncate max-w-40"
                        >
                          {query}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div className="flex space-x-4">
                  <div className="flex-1 relative">
                    <input
                      type="text"
                      value={inputValue}
                      onChange={(e) => setInputValue(e.target.value)}
                      onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
                      placeholder="Ask me anything about your data... e.g., 'Show me trends', 'Analyze performance', 'Generate insights'"
                      className="w-full border border-gray-300 rounded-xl px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                      disabled={isLoading}
                    />
                    <div className="absolute right-3 top-3 text-xs text-gray-400">
                      {selectedDomain !== 'auto' && (
                        <span className="capitalize">{selectedDomain}</span>
                      )}
                    </div>
                  </div>

                  <button
                    onClick={handleSendMessage}
                    disabled={isLoading || !inputValue.trim()}
                    className="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105 flex items-center shadow-lg"
                  >
                    {isLoading ? (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    ) : (
                      <>
                        <MessageCircle className="h-4 w-4 mr-2" />
                        Analyze
                      </>
                    )}
                  </button>
                </div>

                {/* Feature indicators */}
                <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
                  <div className="flex items-center space-x-4">
                    <span className="flex items-center">
                      <Brain className="h-3 w-3 mr-1" />
                      AI-Powered
                    </span>
                    <span className="flex items-center">
                      <Calculator className="h-3 w-3 mr-1" />
                      Mathematical Analysis
                    </span>
                    <span className="flex items-center">
                      <Globe className="h-3 w-3 mr-1" />
                      Multi-Domain
                    </span>
                  </div>
                  {systemStatus?.enhanced_analytics_part4b?.comprehensive_reporting && (
                    <span className="flex items-center text-green-600">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      Advanced Reporting Available
                    </span>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}