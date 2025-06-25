import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import {TrendingUp, BarChart3, PieChart as PieChartIcon, Scatter as ScatterIcon, Download, Maximize2, RefreshCw, Eye, EyeOff} from 'lucide-react';

interface ChartData {
  [key: string]: any;
}

interface ChartConfig {
  id: string;
  title: string;
  type: 'line' | 'area' | 'bar' | 'pie' | 'scatter' | 'histogram';
  data: ChartData[];
  xField?: string;
  yField?: string;
  categoryField?: string;
  valueField?: string;
  description?: string;
  reasoning?: string;
  color?: string;
  colors?: string[];
}

interface ChartsProps {
  charts: ChartConfig[];
  data?: ChartData[];
  onChartGenerate?: (chartId: string) => void;
  className?: string;
}

// Color palettes for charts
const COLOR_PALETTES = {
  default: ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#8dd1e1', '#d084d0'],
  blue: ['#1e40af', '#3b82f6', '#60a5fa', '#93c5fd', '#dbeafe'],
  green: ['#065f46', '#047857', '#059669', '#10b981', '#34d399'],
  purple: ['#581c87', '#7c3aed', '#8b5cf6', '#a78bfa', '#c4b5fd'],
  rainbow: ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6']
};

// Chart type icons
const CHART_ICONS = {
  line: TrendingUp,
  area: TrendingUp,
  bar: BarChart3,
  pie: PieChartIcon,
  scatter: ScatterIcon,
  histogram: BarChart3
};

const ChartComponent: React.FC<{
  config: ChartConfig;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}> = ({ config, isExpanded = false, onToggleExpand }) => {
  const [isVisible, setIsVisible] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  const colors = config.colors || COLOR_PALETTES.default;
  const IconComponent = CHART_ICONS[config.type];

  const containerHeight = isExpanded ? 500 : 300;

  const handleRefresh = () => {
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 1000); // Simulate refresh
  };

  const renderChart = () => {
    if (!config.data || config.data.length === 0) {
      return (
        <div className="flex items-center justify-center h-64 text-gray-500">
          <div className="text-center">
            <div className="text-4xl mb-2">📊</div>
            <p>No data available for visualization</p>
            <button
              onClick={handleRefresh}
              className="mt-2 text-sm text-blue-600 hover:text-blue-800 flex items-center space-x-1 mx-auto"
            >
              <RefreshCw className="w-3 h-3" />
              <span>Retry</span>
            </button>
          </div>
        </div>
      );
    }

    const commonProps = {
      data: config.data,
      margin: { top: 20, right: 30, left: 20, bottom: 20 }
    };

    switch (config.type) {
      case 'line':
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={config.xField || 'x'} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey={config.yField || 'value'}
              stroke={colors[0]}
              strokeWidth={2}
              dot={{ r: 4 }}
            />
          </LineChart>
        );

      case 'area':
        return (
          <AreaChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={config.xField || 'x'} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area
              type="monotone"
              dataKey={config.yField || 'value'}
              stroke={colors[0]}
              fill={colors[0]}
              fillOpacity={0.6}
            />
          </AreaChart>
        );

      case 'bar':
      case 'histogram':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={config.xField || 'x'} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey={config.yField || 'value'} fill={colors[0]} />
          </BarChart>
        );

      case 'pie':
        return (
          <PieChart {...commonProps}>
            <Pie
              data={config.data}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey={config.valueField || 'value'}
            >
              {config.data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        );

      case 'scatter':
        return (
          <ScatterChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={config.xField || 'x'} type="number" />
            <YAxis dataKey={config.yField || 'y'} type="number" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter data={config.data} fill={colors[0]} />
          </ScatterChart>
        );

      default:
        return (
          <div className="flex items-center justify-center h-64 text-gray-500">
            <div className="text-center">
              <div className="text-4xl mb-2">❓</div>
              <p>Unsupported chart type: {config.type}</p>
            </div>
          </div>
        );
    }
  };

  if (!isVisible) {
    return (
      <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 text-gray-600">
            <IconComponent className="w-4 h-4" />
            <span className="text-sm font-medium">{config.title}</span>
            <span className="text-xs text-gray-500">(Hidden)</span>
          </div>
          <button
            onClick={() => setIsVisible(true)}
            className="text-gray-400 hover:text-gray-600 p-1"
            title="Show Chart"
          >
            <Eye className="w-4 h-4" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`border border-gray-200 rounded-lg bg-white ${isExpanded ? 'fixed inset-4 z-50 shadow-2xl' : ''}`}>
      {/* Chart Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200">
        <div className="flex items-center space-x-2">
          <IconComponent className="w-5 h-5 text-blue-600" />
          <div>
            <h3 className="font-semibold text-gray-900">{config.title}</h3>
            {config.description && (
              <p className="text-sm text-gray-600">{config.description}</p>
            )}
          </div>
        </div>

        <div className="flex items-center space-x-2">
          {isLoading && (
            <RefreshCw className="w-4 h-4 text-blue-600 animate-spin" />
          )}

          <button
            onClick={() => setIsVisible(false)}
            className="text-gray-400 hover:text-gray-600 p-1"
            title="Hide Chart"
          >
            <EyeOff className="w-4 h-4" />
          </button>

          <button
            onClick={handleRefresh}
            className="text-gray-400 hover:text-gray-600 p-1"
            title="Refresh Chart"
          >
            <RefreshCw className="w-4 h-4" />
          </button>

          <button
            onClick={onToggleExpand}
            className="text-gray-400 hover:text-gray-600 p-1"
            title={isExpanded ? "Minimize" : "Expand"}
          >
            <Maximize2 className="w-4 h-4" />
          </button>

          <button
            className="text-gray-400 hover:text-gray-600 p-1"
            title="Download Chart"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Chart Content */}
      <div className="p-4">
        <ResponsiveContainer width="100%" height={containerHeight}>
          {renderChart()}
        </ResponsiveContainer>

        {/* Chart Info */}
        {config.reasoning && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-800">
              <strong>Why this chart:</strong> {config.reasoning}
            </p>
          </div>
        )}
      </div>

      {/* Expanded overlay close button */}
      {isExpanded && (
        <button
          onClick={onToggleExpand}
          className="absolute top-4 right-4 bg-white border border-gray-300 rounded-lg p-2 shadow-lg hover:bg-gray-50"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      )}
    </div>
  );
};

const Charts: React.FC<ChartsProps> = ({
  charts,
  data,
  onChartGenerate,
  className = ""
}) => {
  const [expandedChart, setExpandedChart] = useState<string | null>(null);
  const [generatedCharts, setGeneratedCharts] = useState<ChartConfig[]>([]);

  useEffect(() => {
    if (charts) {
      setGeneratedCharts(charts);
    }
  }, [charts]);

  const handleToggleExpand = (chartId: string) => {
    setExpandedChart(expandedChart === chartId ? null : chartId);
  };

  const generateSampleChart = (type: ChartConfig['type']) => {
    const sampleData = generateSampleData(type);
    const newChart: ChartConfig = {
      id: `sample-${Date.now()}`,
      title: `Sample ${type.charAt(0).toUpperCase() + type.slice(1)} Chart`,
      type,
      data: sampleData,
      xField: 'x',
      yField: 'value',
      description: `A sample ${type} chart with mock data`,
      reasoning: `This ${type} chart demonstrates the visualization capability.`
    };

    setGeneratedCharts(prev => [...prev, newChart]);
    onChartGenerate?.(newChart.id);
  };

  const generateSampleData = (type: ChartConfig['type']) => {
    switch (type) {
      case 'pie':
        return [
          { name: 'Category A', value: 400 },
          { name: 'Category B', value: 300 },
          { name: 'Category C', value: 300 },
          { name: 'Category D', value: 200 }
        ];
      case 'scatter':
        return Array.from({ length: 20 }, (_, i) => ({
          x: Math.random() * 100,
          y: Math.random() * 100,
          value: Math.random() * 50
        }));
      default:
        return Array.from({ length: 10 }, (_, i) => ({
          x: `Point ${i + 1}`,
          value: Math.floor(Math.random() * 100) + 10
        }));
    }
  };

  if (!generatedCharts || generatedCharts.length === 0) {
    return (
      <div className={`space-y-6 ${className}`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <BarChart3 className="w-6 h-6 text-blue-600" />
            <h2 className="text-xl font-bold text-gray-900">Visualizations</h2>
          </div>
        </div>

        <div className="bg-white border border-gray-200 rounded-lg p-8 text-center">
          <div className="text-6xl mb-4">📊</div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">No Charts Available</h3>
          <p className="text-gray-600 mb-6">
            Upload data and ask for analysis to generate intelligent visualizations, or create sample charts below.
          </p>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            {(['line', 'bar', 'area', 'pie', 'scatter'] as const).map((type) => {
              const IconComponent = CHART_ICONS[type];
              return (
                <button
                  key={type}
                  onClick={() => generateSampleChart(type)}
                  className="p-4 border border-gray-200 rounded-lg hover:border-blue-300 hover:bg-blue-50 transition-colors"
                >
                  <IconComponent className="w-6 h-6 text-blue-600 mx-auto mb-2" />
                  <div className="text-sm font-medium text-gray-900 capitalize">{type}</div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-6 h-6 text-blue-600" />
          <h2 className="text-xl font-bold text-gray-900">Visualizations</h2>
          <span className="text-sm text-gray-500">({generatedCharts.length})</span>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setGeneratedCharts([])}
            className="text-sm text-gray-600 hover:text-gray-800"
          >
            Clear All
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {generatedCharts.map((chart) => (
          <ChartComponent
            key={chart.id}
            config={chart}
            isExpanded={expandedChart === chart.id}
            onToggleExpand={() => handleToggleExpand(chart.id)}
          />
        ))}
      </div>

      {/* Quick Chart Generation */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h3 className="text-sm font-medium text-gray-900 mb-3">Quick Chart Generation</h3>
        <div className="flex flex-wrap gap-2">
          {(['line', 'bar', 'area', 'pie', 'scatter'] as const).map((type) => (
            <button
              key={type}
              onClick={() => generateSampleChart(type)}
              className="text-xs bg-white border border-gray-200 rounded px-3 py-1 hover:border-blue-300 hover:bg-blue-50 transition-colors capitalize"
            >
              + {type}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default Charts;