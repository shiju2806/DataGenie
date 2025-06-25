import React, { useState, useEffect, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import { analyticsAPI } from '../../services/api';

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
  const [uploadedData, setUploadedData] = useState<any[]>([]);
  const [fileName, setFileName] = useState<string>('');
  const [useBackend, setUseBackend] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    // Add welcome message
    setMessages([
      {
        id: '1',
        type: 'assistant',
        content: "👋 Hi! I'm your AI analytics assistant. Upload your data and ask me questions like:\n\n• What's the total sales in 2024?\n• Show me trends by month\n• Which region performs best?\n• Create a chart of sales by category\n\nI'll analyze your data and we can have a conversation about your insights!",
        timestamp: new Date(),
        suggestions: [
          "Upload a CSV file to get started",
          "Ask about trends and patterns",
          "Request specific metrics",
          "Generate visualizations"
        ]
      }
    ]);
  }, []);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);

    try {
      if (useBackend) {
        // Try backend upload first
        const formData = new FormData();
        formData.append('file', file);
        formData.append('prompt', 'File uploaded - ready for analysis');

        const response = await analyticsAPI.analyzeData(formData);

        if (response.status === 'success') {
          // Extract data from backend response
          const backendData = response.analysis?.data || [];
          setUploadedData(backendData);

          const uploadMessage: Message = {
            id: Date.now().toString(),
            type: 'assistant',
            content: `✅ **File uploaded successfully via backend!**\n\n📊 **${file.name}**\n• ${backendData.length} rows processed\n• Backend analysis ready\n\nNow you can ask me questions about your data!`,
            timestamp: new Date(),
            suggestions: [
              "What's the total sales?",
              "Show me trends over time",
              "Which category performs best?",
              "Create a chart of the data"
            ]
          };

          setMessages(prev => [...prev, uploadMessage]);
          return;
        }
      }

      // Fallback to frontend parsing
      const text = await file.text();
      const lines = text.split('\n');
      const headers = lines[0].split(',').map(h => h.trim());

      const data = lines.slice(1)
        .filter(line => line.trim())
        .map((line, index) => {
          const values = line.split(',');
          const row: any = { id: index };
          headers.forEach((header, i) => {
            const value = values[i]?.trim() || '';
            // Try to parse as number
            const numValue = parseFloat(value);
            row[header] = isNaN(numValue) ? value : numValue;
          });
          return row;
        });

      setUploadedData(data);

      // Add file upload message
      const uploadMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: `✅ **File uploaded successfully!**\n\n📊 **${file.name}**\n• ${data.length} rows\n• ${headers.length} columns\n• Columns: ${headers.join(', ')}\n\nNow you can ask me questions about your data!`,
        timestamp: new Date(),
        suggestions: [
          "What's the total sales?",
          "Show me trends over time",
          "Which category performs best?",
          "Create a chart of the data"
        ]
      };

      setMessages(prev => [...prev, uploadMessage]);
    } catch (error) {
      console.error('Error uploading file:', error);

      const errorMessage: Message = {
        id: Date.now().toString(),
        type: 'assistant',
        content: `❌ **Error uploading file**: ${error}\n\nPlease try uploading a CSV file with proper formatting.`,
        timestamp: new Date(),
        suggestions: ["Try a different CSV file", "Check file format"]
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  };

  const analyzeWithBackend = async (question: string): Promise<AnalysisResult | null> => {
    try {
      if (!useBackend) return null;

      const formData = new FormData();
      formData.append('prompt', question);

      // If we have uploaded data, we need to re-upload it or store it
      if (uploadedData.length > 0 && fileInputRef.current?.files?.[0]) {
        formData.append('file', fileInputRef.current.files[0]);
      }

      const response = await analyticsAPI.analyzeData(formData);

      if (response.status === 'success' && response.analysis) {
        return {
          summary: response.analysis.summary || "Analysis completed",
          insights: response.analysis.insights || [],
          charts: (response.chart_intelligence?.suggested_charts || []).map((chart: any) => ({
            type: chart.chart_type || 'bar',
            title: chart.title || 'Chart',
            data: chart.data || [],
            xAxis: chart.x_axis,
            yAxis: chart.y_axis
          })),
          data: response.analysis.data || [],
          metrics: response.analysis.metadata
        };
      }

      return null;
    } catch (error) {
      console.error('Backend analysis failed:', error);
      return null;
    }
  };

  const analyzeQuestion = async (question: string): Promise<AnalysisResult> => {
    // Try backend first if available
    const backendResult = await analyzeWithBackend(question);
    if (backendResult) {
      return backendResult;
    }

    // Fallback to frontend analysis
    if (uploadedData.length === 0) {
      return {
        summary: "I don't have any data to analyze yet. Please upload a CSV file first!",
        insights: ["Upload a CSV file to get started with analysis"],
        charts: [],
        data: []
      };
    }

    const questionLower = question.toLowerCase();
    const headers = Object.keys(uploadedData[0] || {});
    const numericColumns = headers.filter(header =>
      uploadedData.some(row => typeof row[header] === 'number')
    );

    // Detect question intent and generate appropriate response
    if (questionLower.includes('total') || questionLower.includes('sum')) {
      return analyzeTotalQuestion(question, questionLower, numericColumns);
    } else if (questionLower.includes('trend') || questionLower.includes('over time')) {
      return analyzeTrendQuestion(question, questionLower);
    } else if (questionLower.includes('chart') || questionLower.includes('graph') || questionLower.includes('visualiz')) {
      return generateChartAnalysis(question, questionLower);
    } else if (questionLower.includes('best') || questionLower.includes('top') || questionLower.includes('highest')) {
      return analyzeTopPerformers(question, questionLower);
    } else if (questionLower.includes('average') || questionLower.includes('mean')) {
      return analyzeAverages(question, questionLower, numericColumns);
    } else {
      return generateGeneralAnalysis(question);
    }
  };

  const analyzeTotalQuestion = (question: string, questionLower: string, numericColumns: string[]): AnalysisResult => {
    // Find the column that matches the question
    const targetColumn = numericColumns.find(col =>
      questionLower.includes(col.toLowerCase()) ||
      col.toLowerCase().includes('sales') ||
      col.toLowerCase().includes('revenue') ||
      col.toLowerCase().includes('amount')
    ) || numericColumns[0];

    if (!targetColumn) {
      return {
        summary: "I couldn't find any numeric columns to sum up.",
        insights: ["Your data appears to have no numeric columns for calculation"],
        charts: [],
        data: uploadedData.slice(0, 5)
      };
    }

    const total = uploadedData.reduce((sum, row) => sum + (row[targetColumn] || 0), 0);
    const count = uploadedData.length;
    const average = total / count;

    // Check for year filter
    let filteredData = uploadedData;
    let yearFilter = '';
    if (questionLower.includes('2024')) {
      yearFilter = '2024';
      filteredData = uploadedData.filter(row => {
        const dateFields = Object.keys(row).filter(key =>
          key.toLowerCase().includes('date') ||
          key.toLowerCase().includes('year') ||
          String(row[key]).includes('2024')
        );
        return dateFields.some(field => String(row[field]).includes('2024'));
      });
    }

    const filteredTotal = filteredData.reduce((sum, row) => sum + (row[targetColumn] || 0), 0);

    return {
      summary: `📊 **${targetColumn} Analysis${yearFilter ? ` for ${yearFilter}` : ''}**\n\n💰 **Total ${targetColumn}**: ${filteredTotal.toLocaleString()}\n📈 **Records**: ${filteredData.length}\n📊 **Average**: ${(filteredTotal / filteredData.length).toLocaleString()}`,
      insights: [
        `Total ${targetColumn} is ${filteredTotal.toLocaleString()}`,
        `Based on ${filteredData.length} records${yearFilter ? ` from ${yearFilter}` : ''}`,
        `Average ${targetColumn} per record: ${(filteredTotal / filteredData.length).toFixed(2)}`,
        filteredData.length !== uploadedData.length ? `Filtered from ${uploadedData.length} total records` : ''
      ].filter(Boolean),
      charts: [
        {
          type: 'bar',
          title: `${targetColumn} Summary`,
          data: [
            { name: 'Total', value: filteredTotal },
            { name: 'Average', value: filteredTotal / filteredData.length },
            { name: 'Count', value: filteredData.length }
          ]
        }
      ],
      data: filteredData.slice(0, 10),
      metrics: {
        [`Total ${targetColumn}`]: filteredTotal,
        'Records': filteredData.length,
        'Average': Math.round(filteredTotal / filteredData.length)
      }
    };
  };

  const analyzeTrendQuestion = (question: string, questionLower: string): AnalysisResult => {
    const dateColumn = Object.keys(uploadedData[0] || {}).find(col =>
      col.toLowerCase().includes('date') ||
      col.toLowerCase().includes('time') ||
      col.toLowerCase().includes('month') ||
      col.toLowerCase().includes('year')
    );

    const valueColumn = Object.keys(uploadedData[0] || {}).find(col =>
      typeof uploadedData[0][col] === 'number' &&
      (col.toLowerCase().includes('sales') ||
       col.toLowerCase().includes('revenue') ||
       col.toLowerCase().includes('amount'))
    );

    if (!dateColumn || !valueColumn) {
      return {
        summary: "I need date and numeric columns to show trends over time.",
        insights: ["Please ensure your data has date/time and numeric columns"],
        charts: [],
        data: uploadedData.slice(0, 5)
      };
    }

    // Group by date and sum values
    const trendData = uploadedData.reduce((acc: any, row) => {
      const date = String(row[dateColumn]);
      const value = row[valueColumn] || 0;

      if (!acc[date]) {
        acc[date] = { date, total: 0, count: 0 };
      }
      acc[date].total += value;
      acc[date].count += 1;

      return acc;
    }, {});

    const chartData = Object.values(trendData).sort((a: any, b: any) =>
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    return {
      summary: `📈 **Trend Analysis: ${valueColumn} over ${dateColumn}**\n\n📊 **Time periods**: ${chartData.length}\n📈 **Total ${valueColumn}**: ${chartData.reduce((sum: number, item: any) => sum + item.total, 0).toLocaleString()}`,
      insights: [
        `Analyzed ${valueColumn} trends across ${chartData.length} time periods`,
        `Highest peak: ${Math.max(...chartData.map((item: any) => item.total)).toLocaleString()}`,
        `Lowest point: ${Math.min(...chartData.map((item: any) => item.total)).toLocaleString()}`,
        `Average per period: ${(chartData.reduce((sum: number, item: any) => sum + item.total, 0) / chartData.length).toFixed(2)}`
      ],
      charts: [
        {
          type: 'line',
          title: `${valueColumn} Trend Over Time`,
          data: chartData,
          xAxis: 'date',
          yAxis: 'total'
        }
      ],
      data: chartData
    };
  };

  const generateChartAnalysis = (question: string, questionLower: string): AnalysisResult => {
    const numericColumns = Object.keys(uploadedData[0] || {}).filter(header =>
      uploadedData.some(row => typeof row[header] === 'number')
    );

    if (numericColumns.length === 0) {
      return {
        summary: "No numeric data found for visualization.",
        insights: ["Your data needs numeric columns to create charts"],
        charts: [],
        data: uploadedData.slice(0, 5)
      };
    }

    const primaryColumn = numericColumns[0];
    const categoryColumn = Object.keys(uploadedData[0] || {}).find(col =>
      typeof uploadedData[0][col] === 'string'
    );

    let chartData;
    let chartType: 'bar' | 'pie' | 'line' = 'bar';

    if (categoryColumn) {
      // Group by category
      const grouped = uploadedData.reduce((acc: any, row) => {
        const category = row[categoryColumn];
        const value = row[primaryColumn] || 0;

        if (!acc[category]) {
          acc[category] = { name: category, value: 0 };
        }
        acc[category].value += value;

        return acc;
      }, {});

      chartData = Object.values(grouped);
      chartType = questionLower.includes('pie') ? 'pie' : 'bar';
    } else {
      // Use raw data
      chartData = uploadedData.slice(0, 10).map((row, index) => ({
        name: `Item ${index + 1}`,
        value: row[primaryColumn] || 0
      }));
    }

    return {
      summary: `📊 **Chart: ${primaryColumn}${categoryColumn ? ` by ${categoryColumn}` : ''}**\n\n📈 **Chart type**: ${chartType.toUpperCase()}\n📊 **Categories**: ${chartData.length}`,
      insights: [
        `Created ${chartType} chart for ${primaryColumn}`,
        `Showing ${chartData.length} categories`,
        `Highest value: ${Math.max(...chartData.map((item: any) => item.value)).toLocaleString()}`,
        `Total sum: ${chartData.reduce((sum: number, item: any) => sum + item.value, 0).toLocaleString()}`
      ],
      charts: [
        {
          type: chartType,
          title: `${primaryColumn}${categoryColumn ? ` by ${categoryColumn}` : ''}`,
          data: chartData
        }
      ],
      data: chartData
    };
  };

  const analyzeTopPerformers = (question: string, questionLower: string): AnalysisResult => {
    const numericColumns = Object.keys(uploadedData[0] || {}).filter(header =>
      uploadedData.some(row => typeof row[header] === 'number')
    );

    const targetColumn = numericColumns.find(col =>
      questionLower.includes(col.toLowerCase())
    ) || numericColumns[0];

    if (!targetColumn) {
      return {
        summary: "No numeric columns found for performance analysis.",
        insights: ["Need numeric data to identify top performers"],
        charts: [],
        data: uploadedData.slice(0, 5)
      };
    }

    const sorted = [...uploadedData].sort((a, b) => (b[targetColumn] || 0) - (a[targetColumn] || 0));
    const top5 = sorted.slice(0, 5);

    return {
      summary: `🏆 **Top Performers by ${targetColumn}**\n\n🥇 **#1**: ${top5[0]?.[targetColumn]?.toLocaleString() || 'N/A'}\n🥈 **#2**: ${top5[1]?.[targetColumn]?.toLocaleString() || 'N/A'}\n🥉 **#3**: ${top5[2]?.[targetColumn]?.toLocaleString() || 'N/A'}`,
      insights: [
        `Top performer has ${targetColumn} of ${top5[0]?.[targetColumn]?.toLocaleString()}`,
        `Top 5 combined total: ${top5.reduce((sum, item) => sum + (item[targetColumn] || 0), 0).toLocaleString()}`,
        `Performance gap: ${((top5[0]?.[targetColumn] || 0) - (top5[4]?.[targetColumn] || 0)).toLocaleString()} difference between #1 and #5`
      ],
      charts: [
        {
          type: 'bar',
          title: `Top 5 by ${targetColumn}`,
          data: top5.map((item, index) => ({
            name: `#${index + 1}`,
            value: item[targetColumn] || 0
          }))
        }
      ],
      data: top5
    };
  };

  const analyzeAverages = (question: string, questionLower: string, numericColumns: string[]): AnalysisResult => {
    const targetColumn = numericColumns.find(col =>
      questionLower.includes(col.toLowerCase())
    ) || numericColumns[0];

    if (!targetColumn) {
      return {
        summary: "No numeric columns found for average calculation.",
        insights: ["Need numeric data to calculate averages"],
        charts: [],
        data: uploadedData.slice(0, 5)
      };
    }

    const values = uploadedData.map(row => row[targetColumn] || 0).filter(val => val > 0);
    const average = values.reduce((sum, val) => sum + val, 0) / values.length;
    const median = [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
    const max = Math.max(...values);
    const min = Math.min(...values);

    return {
      summary: `📊 **Average Analysis for ${targetColumn}**\n\n📈 **Average**: ${average.toLocaleString()}\n📊 **Median**: ${median.toLocaleString()}\n📈 **Range**: ${min.toLocaleString()} - ${max.toLocaleString()}`,
      insights: [
        `Average ${targetColumn}: ${average.toFixed(2)}`,
        `Median value: ${median.toFixed(2)}`,
        `Highest value: ${max.toLocaleString()}`,
        `Lowest value: ${min.toLocaleString()}`,
        `Standard deviation: ${Math.sqrt(values.reduce((acc, val) => acc + Math.pow(val - average, 2), 0) / values.length).toFixed(2)}`
      ],
      charts: [
        {
          type: 'bar',
          title: `${targetColumn} Statistics`,
          data: [
            { name: 'Average', value: average },
            { name: 'Median', value: median },
            { name: 'Max', value: max },
            { name: 'Min', value: min }
          ]
        }
      ],
      data: uploadedData.slice(0, 10),
      metrics: {
        'Average': Math.round(average),
        'Median': Math.round(median),
        'Max': max,
        'Min': min
      }
    };
  };

  const generateGeneralAnalysis = (question: string): AnalysisResult => {
    const headers = Object.keys(uploadedData[0] || {});
    const numericColumns = headers.filter(header =>
      uploadedData.some(row => typeof row[header] === 'number')
    );

    return {
      summary: `🔍 **General Data Overview**\n\n📊 **Records**: ${uploadedData.length}\n📋 **Columns**: ${headers.length}\n📈 **Numeric fields**: ${numericColumns.length}`,
      insights: [
        `Your dataset contains ${uploadedData.length} records`,
        `${numericColumns.length} numeric columns available for analysis`,
        `Columns: ${headers.join(', ')}`,
        "Try asking more specific questions like 'What's the total sales?' or 'Show me trends over time'"
      ],
      charts: [],
      data: uploadedData.slice(0, 10)
    };
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

    try {
      const analysis = await analyzeQuestion(currentQuestion);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: analysis.summary,
        timestamp: new Date(),
        analysis,
        suggestions: [
          "Tell me more about this data",
          "Show me different visualizations",
          "What other insights can you find?",
          "Compare this with other metrics"
        ]
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Analysis failed:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: "Sorry, I encountered an error analyzing your question. Please try again!",
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderChart = (chart: ChartData) => {
    const { type, title, data, xAxis, yAxis } = chart;

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
    "What's the total sales?",
    "Show me trends over time",
    "Which category performs best?",
    "What's the average revenue?",
    "Create a pie chart of data",
    "Show me top 5 performers"
  ];

  const handleBackToHome = () => {
    window.location.href = '/';
  };

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
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={useBackend}
                  onChange={(e) => setUseBackend(e.target.checked)}
                  className="rounded"
                />
                <span>Use Backend AI</span>
              </label>
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
                  Upload a CSV file to start analyzing your data
                </p>
              </div>

              <div className="flex items-center space-x-4">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="block text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
                {fileName && (
                  <div className="flex items-center space-x-2 text-sm text-green-600">
                    <span>✅</span>
                    <span>{fileName} ({uploadedData.length} rows)</span>
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
                    <span>Analyzing your question...</span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Quick Questions */}
          {uploadedData.length > 0 && (
            <div className="border-t border-gray-200 p-4">
              <h3 className="text-sm font-medium text-gray-700 mb-2">💡 Quick Questions</h3>
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
                placeholder={uploadedData.length > 0 ? "Ask a question about your data..." : "Upload data first, then ask questions..."}
                disabled={isAnalyzing || uploadedData.length === 0}
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={isAnalyzing || !currentQuestion.trim() || uploadedData.length === 0}
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
          </div>
        </div>
      </div>
    </div>
  );
};

export default ConversationalAnalytics;