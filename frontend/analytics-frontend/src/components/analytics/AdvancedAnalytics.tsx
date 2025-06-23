import React, { useState, useEffect, useRef } from 'react';
import { analyticsAPI } from '../../services/api';

interface AnalyticsModel {
  id: string;
  name: string;
  type: 'predictive' | 'classification' | 'clustering' | 'anomaly' | 'correlation';
  description: string;
  accuracy?: number;
  status: 'training' | 'ready' | 'error';
  lastTrained?: string;
  metrics?: {
    mse?: number;
    r2_score?: number;
    precision?: number;
    recall?: number;
    f1_score?: number;
  };
}

interface PredictionResult {
  model_id: string;
  predictions: Array<{
    timestamp: string;
    predicted_value: number;
    confidence_interval: [number, number];
    confidence_score: number;
  }>;
  feature_importance: Array<{
    feature: string;
    importance: number;
  }>;
  model_performance: {
    accuracy: number;
    training_duration: number;
    data_points_used: number;
  };
}

interface AnomalyDetectionResult {
  anomalies: Array<{
    timestamp: string;
    value: number;
    anomaly_score: number;
    severity: 'low' | 'medium' | 'high' | 'critical';
    explanation: string;
  }>;
  patterns: Array<{
    type: string;
    description: string;
    frequency: string;
    impact: number;
  }>;
  recommendations: string[];
}

interface CorrelationAnalysis {
  correlations: Array<{
    variable1: string;
    variable2: string;
    correlation: number;
    p_value: number;
    significance: 'weak' | 'moderate' | 'strong';
    relationship_type: 'positive' | 'negative' | 'no_correlation';
  }>;
  causal_insights: Array<{
    cause: string;
    effect: string;
    strength: number;
    confidence: number;
    explanation: string;
  }>;
}

interface StatisticalInsights {
  distribution_analysis: {
    data_type: string;
    distribution: string;
    parameters: any;
    goodness_of_fit: number;
  };
  trend_analysis: {
    trend_direction: 'increasing' | 'decreasing' | 'stable' | 'cyclical';
    trend_strength: number;
    seasonality: boolean;
    change_points: Array<{
      timestamp: string;
      significance: number;
    }>;
  };
  outlier_analysis: {
    outlier_count: number;
    outlier_percentage: number;
    outlier_impact: 'low' | 'moderate' | 'high';
    method_used: string;
  };
}

const AdvancedAnalytics: React.FC = () => {
  const [activeTab, setActiveTab] = useState('predictive');
  const [models, setModels] = useState<AnalyticsModel[]>([]);
  const [selectedModel, setSelectedModel] = useState<AnalyticsModel | null>(null);
  const [predictionResults, setPredictionResults] = useState<PredictionResult | null>(null);
  const [anomalyResults, setAnomalyResults] = useState<AnomalyDetectionResult | null>(null);
  const [correlationResults, setCorrelationResults] = useState<CorrelationAnalysis | null>(null);
  const [statisticalInsights, setStatisticalInsights] = useState<StatisticalInsights | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState<{ [key: string]: boolean }>({});
  const [trainingProgress, setTrainingProgress] = useState<{ [key: string]: number }>({});

  const fileInputRef = useRef<HTMLInputElement>(null);

  const analyticsTypes = [
    {
      id: 'predictive',
      name: 'Predictive Analytics',
      icon: '🔮',
      description: 'Forecast future trends and outcomes'
    },
    {
      id: 'anomaly',
      name: 'Anomaly Detection',
      icon: '🎯',
      description: 'Identify unusual patterns and outliers'
    },
    {
      id: 'correlation',
      name: 'Correlation Analysis',
      icon: '🔗',
      description: 'Find relationships between variables'
    },
    {
      id: 'statistical',
      name: 'Statistical Insights',
      icon: '📈',
      description: 'Advanced statistical analysis'
    },
    {
      id: 'clustering',
      name: 'Clustering',
      icon: '🎪',
      description: 'Group similar data points'
    }
  ];

  const modelTemplates = [
    {
      type: 'predictive',
      name: 'Time Series Forecasting',
      description: 'ARIMA/LSTM models for time series prediction',
      algorithms: ['ARIMA', 'LSTM', 'Prophet', 'Linear Regression']
    },
    {
      type: 'classification',
      name: 'Classification Model',
      description: 'Classify data into categories',
      algorithms: ['Random Forest', 'SVM', 'Neural Network', 'Logistic Regression']
    },
    {
      type: 'clustering',
      name: 'Customer Segmentation',
      description: 'Group customers by behavior patterns',
      algorithms: ['K-Means', 'DBSCAN', 'Hierarchical', 'Gaussian Mixture']
    },
    {
      type: 'anomaly',
      name: 'Anomaly Detection',
      description: 'Detect unusual patterns and outliers',
      algorithms: ['Isolation Forest', 'One-Class SVM', 'Autoencoder', 'Local Outlier Factor']
    }
  ];

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = () => {
    // Load saved models or initialize with mock data
    const savedModels = localStorage.getItem('analytics_models');
    if (savedModels) {
      setModels(JSON.parse(savedModels));
    } else {
      const mockModels: AnalyticsModel[] = [
        {
          id: '1',
          name: 'Sales Forecasting Model',
          type: 'predictive',
          description: 'Predicts monthly sales based on historical data',
          accuracy: 0.94,
          status: 'ready',
          lastTrained: '2024-01-15',
          metrics: { mse: 0.06, r2_score: 0.94 }
        },
        {
          id: '2',
          name: 'Customer Churn Detection',
          type: 'classification',
          description: 'Identifies customers likely to churn',
          accuracy: 0.87,
          status: 'ready',
          lastTrained: '2024-01-10',
          metrics: { precision: 0.89, recall: 0.85, f1_score: 0.87 }
        }
      ];
      setModels(mockModels);
      localStorage.setItem('analytics_models', JSON.stringify(mockModels));
    }
  };

  const setLoadingState = (key: string, isLoading: boolean) => {
    setLoading(prev => ({ ...prev, [key]: isLoading }));
  };

  const createModel = async (template: typeof modelTemplates[0], algorithm: string) => {
    const modelId = Date.now().toString();
    const newModel: AnalyticsModel = {
      id: modelId,
      name: `${template.name} (${algorithm})`,
      type: template.type as any,
      description: template.description,
      status: 'training'
    };

    const updatedModels = [...models, newModel];
    setModels(updatedModels);
    localStorage.setItem('analytics_models', JSON.stringify(updatedModels));

    // Simulate training progress
    setTrainingProgress(prev => ({ ...prev, [modelId]: 0 }));

    const progressInterval = setInterval(() => {
      setTrainingProgress(prev => {
        const currentProgress = prev[modelId] || 0;
        const newProgress = currentProgress + Math.random() * 15;

        if (newProgress >= 100) {
          clearInterval(progressInterval);

          // Mark model as ready
          const finalModels = updatedModels.map(m =>
            m.id === modelId
              ? {
                  ...m,
                  status: 'ready' as const,
                  accuracy: 0.8 + Math.random() * 0.15,
                  lastTrained: new Date().toISOString().split('T')[0],
                  metrics: {
                    mse: Math.random() * 0.1,
                    r2_score: 0.8 + Math.random() * 0.15,
                    precision: 0.8 + Math.random() * 0.15,
                    recall: 0.8 + Math.random() * 0.15,
                    f1_score: 0.8 + Math.random() * 0.15
                  }
                }
              : m
          );
          setModels(finalModels);
          localStorage.setItem('analytics_models', JSON.stringify(finalModels));

          return { ...prev, [modelId]: 100 };
        }

        return { ...prev, [modelId]: newProgress };
      });
    }, 500);
  };

  const runPredictiveAnalysis = async () => {
    if (!selectedFile || !selectedModel) return;

    setLoadingState('prediction', true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 3000));

      const mockResults: PredictionResult = {
        model_id: selectedModel.id,
        predictions: Array.from({ length: 12 }, (_, i) => ({
          timestamp: new Date(2024, i, 1).toISOString().split('T')[0],
          predicted_value: 1000 + Math.random() * 500 + i * 50,
          confidence_interval: [900 + i * 45, 1100 + i * 55] as [number, number],
          confidence_score: 0.8 + Math.random() * 0.15
        })),
        feature_importance: [
          { feature: 'historical_sales', importance: 0.35 },
          { feature: 'seasonality', importance: 0.28 },
          { feature: 'marketing_spend', importance: 0.22 },
          { feature: 'economic_indicators', importance: 0.15 }
        ],
        model_performance: {
          accuracy: selectedModel.accuracy || 0.9,
          training_duration: 245,
          data_points_used: 10000
        }
      };

      setPredictionResults(mockResults);
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setLoadingState('prediction', false);
    }
  };

  const runAnomalyDetection = async () => {
    if (!selectedFile) return;

    setLoadingState('anomaly', true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2500));

      const mockResults: AnomalyDetectionResult = {
        anomalies: [
          {
            timestamp: '2024-01-15',
            value: 2500,
            anomaly_score: 0.95,
            severity: 'critical',
            explanation: 'Unusual spike in sales, 300% above normal range'
          },
          {
            timestamp: '2024-01-22',
            value: 120,
            anomaly_score: 0.78,
            severity: 'high',
            explanation: 'Significant drop in customer engagement'
          },
          {
            timestamp: '2024-01-28',
            value: 1850,
            anomaly_score: 0.65,
            severity: 'medium',
            explanation: 'Moderate increase in processing time'
          }
        ],
        patterns: [
          {
            type: 'Weekly Cyclical',
            description: 'Regular pattern with peaks on Fridays',
            frequency: 'Weekly',
            impact: 0.8
          },
          {
            type: 'Monthly Trend',
            description: 'Gradual increase throughout the month',
            frequency: 'Monthly',
            impact: 0.6
          }
        ],
        recommendations: [
          'Investigate the cause of the January 15th spike',
          'Monitor customer engagement metrics more closely',
          'Consider capacity planning for Friday peaks',
          'Set up automated alerts for values >200% of normal range'
        ]
      };

      setAnomalyResults(mockResults);
    } catch (error) {
      console.error('Anomaly detection failed:', error);
    } finally {
      setLoadingState('anomaly', false);
    }
  };

  const runCorrelationAnalysis = async () => {
    if (!selectedFile) return;

    setLoadingState('correlation', true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2000));

      const mockResults: CorrelationAnalysis = {
        correlations: [
          {
            variable1: 'marketing_spend',
            variable2: 'sales_revenue',
            correlation: 0.84,
            p_value: 0.001,
            significance: 'strong',
            relationship_type: 'positive'
          },
          {
            variable1: 'customer_satisfaction',
            variable2: 'retention_rate',
            correlation: 0.72,
            p_value: 0.005,
            significance: 'strong',
            relationship_type: 'positive'
          },
          {
            variable1: 'processing_time',
            variable2: 'error_rate',
            correlation: 0.65,
            p_value: 0.01,
            significance: 'moderate',
            relationship_type: 'positive'
          }
        ],
        causal_insights: [
          {
            cause: 'marketing_spend',
            effect: 'sales_revenue',
            strength: 0.78,
            confidence: 0.92,
            explanation: 'Increased marketing spend leads to higher sales with 2-week lag'
          },
          {
            cause: 'customer_satisfaction',
            effect: 'retention_rate',
            strength: 0.69,
            confidence: 0.85,
            explanation: 'Higher satisfaction scores predict better retention rates'
          }
        ]
      };

      setCorrelationResults(mockResults);
    } catch (error) {
      console.error('Correlation analysis failed:', error);
    } finally {
      setLoadingState('correlation', false);
    }
  };

  const runStatisticalAnalysis = async () => {
    if (!selectedFile) return;

    setLoadingState('statistical', true);
    try {
      await new Promise(resolve => setTimeout(resolve, 2200));

      const mockResults: StatisticalInsights = {
        distribution_analysis: {
          data_type: 'continuous',
          distribution: 'normal',
          parameters: { mean: 1250.5, std: 245.8 },
          goodness_of_fit: 0.94
        },
        trend_analysis: {
          trend_direction: 'increasing',
          trend_strength: 0.76,
          seasonality: true,
          change_points: [
            { timestamp: '2024-01-15', significance: 0.89 },
            { timestamp: '2024-02-01', significance: 0.72 }
          ]
        },
        outlier_analysis: {
          outlier_count: 12,
          outlier_percentage: 2.3,
          outlier_impact: 'moderate',
          method_used: 'IQR + Z-Score'
        }
      };

      setStatisticalInsights(mockResults);
    } catch (error) {
      console.error('Statistical analysis failed:', error);
    } finally {
      setLoadingState('statistical', false);
    }
  };

  const getModelStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'text-green-600 bg-green-100';
      case 'training': return 'text-blue-600 bg-blue-100';
      case 'error': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'medium': return 'text-yellow-600 bg-yellow-100';
      case 'low': return 'text-blue-600 bg-blue-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getCorrelationColor = (correlation: number) => {
    const abs = Math.abs(correlation);
    if (abs >= 0.8) return 'text-red-600 bg-red-100';
    if (abs >= 0.6) return 'text-orange-600 bg-orange-100';
    if (abs >= 0.4) return 'text-yellow-600 bg-yellow-100';
    return 'text-gray-600 bg-gray-100';
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Advanced Analytics</h1>
        <p className="text-gray-600">
          Machine learning models, predictive analytics, and statistical insights
        </p>
      </div>

      {/* Analytics Type Tabs */}
      <div className="border-b border-gray-200 mb-8">
        <nav className="-mb-px flex space-x-8">
          {analyticsTypes.map((type) => (
            <button
              key={type.id}
              onClick={() => setActiveTab(type.id)}
              className={`py-3 px-1 border-b-2 font-medium text-sm whitespace-nowrap ${
                activeTab === type.id
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {type.icon} {type.name}
            </button>
          ))}
        </nav>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Main Content */}
        <div className="lg:col-span-3">
          {/* File Upload Section */}
          <div className="card mb-8">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Data Input</h2>
            <div className="flex items-center space-x-4">
              <div className="flex-1">
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx,.xls,.json"
                  onChange={(e) => e.target.files?.[0] && setSelectedFile(e.target.files[0])}
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-center"
                >
                  <div className="text-2xl mb-2">📁</div>
                  <p className="text-gray-600">
                    {selectedFile ? selectedFile.name : 'Upload dataset for analysis'}
                  </p>
                </button>
              </div>

              {selectedFile && (
                <div className="flex flex-col space-y-2">
                  <span className="text-sm text-gray-600">
                    Size: {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                  </span>
                  <button
                    onClick={() => setSelectedFile(null)}
                    className="text-red-600 hover:text-red-800 text-sm"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Predictive Analytics Tab */}
          {activeTab === 'predictive' && (
            <div className="space-y-8">
              {/* Model Selection */}
              <div className="card">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Select Prediction Model</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
                  {models.filter(m => m.type === 'predictive' || m.type === 'classification').map((model) => (
                    <button
                      key={model.id}
                      onClick={() => setSelectedModel(model)}
                      className={`p-4 border rounded-lg text-left transition-colors ${
                        selectedModel?.id === model.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold text-gray-900">{model.name}</h3>
                        <span className={`px-2 py-1 text-xs rounded-full ${getModelStatusColor(model.status)}`}>
                          {model.status}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{model.description}</p>
                      {model.accuracy && (
                        <div className="text-sm text-gray-500">
                          Accuracy: {(model.accuracy * 100).toFixed(1)}%
                        </div>
                      )}
                      {trainingProgress[model.id] !== undefined && model.status === 'training' && (
                        <div className="mt-2">
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                              style={{ width: `${trainingProgress[model.id]}%` }}
                            />
                          </div>
                          <div className="text-xs text-gray-500 mt-1">
                            Training: {Math.round(trainingProgress[model.id])}%
                          </div>
                        </div>
                      )}
                    </button>
                  ))}
                </div>

                <button
                  onClick={runPredictiveAnalysis}
                  disabled={!selectedFile || !selectedModel || loading.prediction}
                  className={`btn-primary ${
                    (!selectedFile || !selectedModel || loading.prediction) ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {loading.prediction ? (
                    <>
                      <svg className="animate-spin w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Running Prediction...
                    </>
                  ) : (
                    'Run Prediction Analysis'
                  )}
                </button>
              </div>

              {/* Prediction Results */}
              {predictionResults && (
                <div className="card">
                  <h2 className="text-xl font-semibold text-gray-900 mb-6">Prediction Results</h2>

                  {/* Performance Metrics */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="text-center p-4 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">
                        {(predictionResults.model_performance.accuracy * 100).toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">Model Accuracy</div>
                    </div>
                    <div className="text-center p-4 bg-green-50 rounded-lg">
                      <div className="text-2xl font-bold text-green-600">
                        {predictionResults.model_performance.data_points_used.toLocaleString()}
                      </div>
                      <div className="text-sm text-gray-600">Data Points Used</div>
                    </div>
                    <div className="text-center p-4 bg-purple-50 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">
                        {predictionResults.model_performance.training_duration}s
                      </div>
                      <div className="text-sm text-gray-600">Training Duration</div>
                    </div>
                  </div>

                  {/* Feature Importance */}
                  <div className="mb-6">
                    <h3 className="font-semibold text-gray-900 mb-3">Feature Importance</h3>
                    <div className="space-y-2">
                      {predictionResults.feature_importance.map((feature, index) => (
                        <div key={index} className="flex items-center justify-between">
                          <span className="text-sm font-medium text-gray-700 capitalize">
                            {feature.feature.replace('_', ' ')}
                          </span>
                          <div className="flex items-center space-x-3">
                            <div className="w-32 bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-blue-600 h-2 rounded-full"
                                style={{ width: `${feature.importance * 100}%` }}
                              />
                            </div>
                            <span className="text-sm text-gray-600 w-12">
                              {(feature.importance * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Predictions Preview */}
                  <div>
                    <h3 className="font-semibold text-gray-900 mb-3">Next 6 Months Forecast</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {predictionResults.predictions.slice(0, 6).map((prediction, index) => (
                        <div key={index} className="p-3 bg-gray-50 rounded-lg">
                          <div className="text-sm text-gray-600">
                            {new Date(prediction.timestamp).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })}
                          </div>
                          <div className="text-lg font-semibold text-gray-900">
                            {prediction.predicted_value.toFixed(0)}
                          </div>
                          <div className="text-xs text-gray-500">
                            Range: {prediction.confidence_interval[0].toFixed(0)} - {prediction.confidence_interval[1].toFixed(0)}
                          </div>
                          <div className="text-xs text-blue-600">
                            Confidence: {(prediction.confidence_score * 100).toFixed(0)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Anomaly Detection Tab */}
          {activeTab === 'anomaly' && (
            <div className="space-y-8">
              <div className="card">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Anomaly Detection</h2>
                <p className="text-gray-600 mb-6">
                  Identify unusual patterns, outliers, and anomalies in your data using advanced machine learning algorithms.
                </p>

                <button
                  onClick={runAnomalyDetection}
                  disabled={!selectedFile || loading.anomaly}
                  className={`btn-primary ${
                    (!selectedFile || loading.anomaly) ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {loading.anomaly ? (
                    <>
                      <svg className="animate-spin w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Detecting Anomalies...
                    </>
                  ) : (
                    'Run Anomaly Detection'
                  )}
                </button>
              </div>

              {anomalyResults && (
                <div className="space-y-6">
                  {/* Detected Anomalies */}
                  <div className="card">
                    <h2 className="text-xl font-semibold text-gray-900 mb-6">Detected Anomalies</h2>
                    <div className="space-y-4">
                      {anomalyResults.anomalies.map((anomaly, index) => (
                        <div key={index} className="p-4 border border-gray-200 rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center space-x-3">
                              <span className={`px-2 py-1 text-xs rounded-full ${getSeverityColor(anomaly.severity)}`}>
                                {anomaly.severity.toUpperCase()}
                              </span>
                              <span className="font-medium text-gray-900">{anomaly.timestamp}</span>
                            </div>
                            <div className="text-right">
                              <div className="font-semibold text-gray-900">{anomaly.value}</div>
                              <div className="text-sm text-gray-500">Score: {anomaly.anomaly_score.toFixed(2)}</div>
                            </div>
                          </div>
                          <p className="text-sm text-gray-600">{anomaly.explanation}</p>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Patterns and Recommendations */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="card">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Detected Patterns</h3>
                      <div className="space-y-3">
                        {anomalyResults.patterns.map((pattern, index) => (
                          <div key={index} className="p-3 bg-gray-50 rounded-lg">
                            <div className="flex items-center justify-between mb-1">
                              <h4 className="font-medium text-gray-900">{pattern.type}</h4>
                              <span className="text-sm text-blue-600">{pattern.frequency}</span>
                            </div>
                            <p className="text-sm text-gray-600 mb-2">{pattern.description}</p>
                            <div className="flex items-center">
                              <span className="text-xs text-gray-500 mr-2">Impact:</span>
                              <div className="w-16 bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-orange-500 h-2 rounded-full"
                                  style={{ width: `${pattern.impact * 100}%` }}
                                />
                              </div>
                              <span className="text-xs text-gray-500 ml-2">{(pattern.impact * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="card">
                      <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommendations</h3>
                      <div className="space-y-2">
                        {anomalyResults.recommendations.map((rec, index) => (
                          <div key={index} className="flex items-start space-x-2">
                            <div className="w-1.5 h-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                            <p className="text-sm text-gray-700">{rec}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Correlation Analysis Tab */}
          {activeTab === 'correlation' && (
            <div className="space-y-8">
              <div className="card">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Correlation Analysis</h2>
                <p className="text-gray-600 mb-6">
                  Discover relationships and causal connections between different variables in your dataset.
                </p>

                <button
                  onClick={runCorrelationAnalysis}
                  disabled={!selectedFile || loading.correlation}
                  className={`btn-primary ${
                    (!selectedFile || loading.correlation) ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {loading.correlation ? (
                    <>
                      <svg className="animate-spin w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Analyzing Correlations...
                    </>
                  ) : (
                    'Run Correlation Analysis'
                  )}
                </button>
              </div>

              {correlationResults && (
                <div className="space-y-6">
                  {/* Correlation Matrix */}
                  <div className="card">
                    <h2 className="text-xl font-semibold text-gray-900 mb-6">Correlation Matrix</h2>
                    <div className="space-y-4">
                      {correlationResults.correlations.map((corr, index) => (
                        <div key={index} className="p-4 border border-gray-200 rounded-lg">
                          <div className="flex items-center justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <span className="font-medium text-gray-900 capitalize">
                                  {corr.variable1.replace('_', ' ')}
                                </span>
                                <span className="text-gray-500">↔</span>
                                <span className="font-medium text-gray-900 capitalize">
                                  {corr.variable2.replace('_', ' ')}
                                </span>
                              </div>
                              <div className="flex items-center space-x-4">
                                <span className={`px-2 py-1 text-xs rounded-full ${getCorrelationColor(corr.correlation)}`}>
                                  {corr.significance}
                                </span>
                                <span className="text-sm text-gray-600">
                                  p-value: {corr.p_value.toFixed(3)}
                                </span>
                                <span className={`text-sm ${corr.relationship_type === 'positive' ? 'text-green-600' : 'text-red-600'}`}>
                                  {corr.relationship_type}
                                </span>
                              </div>
                            </div>
                            <div className="text-right">
                              <div className="text-2xl font-bold text-gray-900">
                                {corr.correlation.toFixed(2)}
                              </div>
                              <div className="w-16 bg-gray-200 rounded-full h-2 mt-1">
                                <div
                                  className={`h-2 rounded-full ${
                                    Math.abs(corr.correlation) >= 0.8 ? 'bg-red-500' :
                                    Math.abs(corr.correlation) >= 0.6 ? 'bg-orange-500' :
                                    Math.abs(corr.correlation) >= 0.4 ? 'bg-yellow-500' : 'bg-gray-400'
                                  }`}
                                  style={{ width: `${Math.abs(corr.correlation) * 100}%` }}
                                />
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Causal Insights */}
                  <div className="card">
                    <h2 className="text-xl font-semibold text-gray-900 mb-6">Causal Insights</h2>
                    <div className="space-y-4">
                      {correlationResults.causal_insights.map((insight, index) => (
                        <div key={index} className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                          <div className="flex items-center justify-between mb-3">
                            <div className="flex items-center space-x-2">
                              <span className="font-medium text-blue-900 capitalize">
                                {insight.cause.replace('_', ' ')}
                              </span>
                              <span className="text-blue-600">→</span>
                              <span className="font-medium text-blue-900 capitalize">
                                {insight.effect.replace('_', ' ')}
                              </span>
                            </div>
                            <div className="text-right">
                              <div className="text-sm text-blue-700">
                                Confidence: {(insight.confidence * 100).toFixed(0)}%
                              </div>
                              <div className="text-sm text-blue-600">
                                Strength: {insight.strength.toFixed(2)}
                              </div>
                            </div>
                          </div>
                          <p className="text-sm text-blue-800">{insight.explanation}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Statistical Insights Tab */}
          {activeTab === 'statistical' && (
            <div className="space-y-8">
              <div className="card">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Statistical Analysis</h2>
                <p className="text-gray-600 mb-6">
                  Comprehensive statistical analysis including distribution analysis, trend detection, and outlier identification.
                </p>

                <button
                  onClick={runStatisticalAnalysis}
                  disabled={!selectedFile || loading.statistical}
                  className={`btn-primary ${
                    (!selectedFile || loading.statistical) ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {loading.statistical ? (
                    <>
                      <svg className="animate-spin w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                      </svg>
                      Running Statistical Analysis...
                    </>
                  ) : (
                    'Run Statistical Analysis'
                  )}
                </button>
              </div>

              {statisticalInsights && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Distribution Analysis */}
                  <div className="card">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Distribution Analysis</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Data Type:</span>
                        <span className="font-medium text-gray-900 capitalize">
                          {statisticalInsights.distribution_analysis.data_type}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Distribution:</span>
                        <span className="font-medium text-gray-900 capitalize">
                          {statisticalInsights.distribution_analysis.distribution}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Goodness of Fit:</span>
                        <span className="font-medium text-green-600">
                          {(statisticalInsights.distribution_analysis.goodness_of_fit * 100).toFixed(1)}%
                        </span>
                      </div>
                      {statisticalInsights.distribution_analysis.parameters && (
                        <div className="pt-2 border-t border-gray-200">
                          <h4 className="text-sm font-medium text-gray-700 mb-2">Parameters:</h4>
                          <div className="space-y-1">
                            {Object.entries(statisticalInsights.distribution_analysis.parameters).map(([key, value]) => (
                              <div key={key} className="flex justify-between text-sm">
                                <span className="text-gray-600 capitalize">{key}:</span>
                                <span className="font-medium text-gray-900">{(value as number).toFixed(2)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Trend Analysis */}
                  <div className="card">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Trend Analysis</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Trend Direction:</span>
                        <span className={`font-medium capitalize ${
                          statisticalInsights.trend_analysis.trend_direction === 'increasing' ? 'text-green-600' :
                          statisticalInsights.trend_analysis.trend_direction === 'decreasing' ? 'text-red-600' :
                          'text-gray-600'
                        }`}>
                          {statisticalInsights.trend_analysis.trend_direction}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Trend Strength:</span>
                        <span className="font-medium text-gray-900">
                          {statisticalInsights.trend_analysis.trend_strength.toFixed(2)}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Seasonality:</span>
                        <span className={`font-medium ${
                          statisticalInsights.trend_analysis.seasonality ? 'text-blue-600' : 'text-gray-600'
                        }`}>
                          {statisticalInsights.trend_analysis.seasonality ? 'Detected' : 'Not detected'}
                        </span>
                      </div>

                      {statisticalInsights.trend_analysis.change_points.length > 0 && (
                        <div className="pt-2 border-t border-gray-200">
                          <h4 className="text-sm font-medium text-gray-700 mb-2">Change Points:</h4>
                          <div className="space-y-1">
                            {statisticalInsights.trend_analysis.change_points.map((point, index) => (
                              <div key={index} className="flex justify-between text-sm">
                                <span className="text-gray-600">{point.timestamp}</span>
                                <span className="font-medium text-gray-900">
                                  {(point.significance * 100).toFixed(0)}%
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Outlier Analysis */}
                  <div className="card lg:col-span-2">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Outlier Analysis</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="text-center">
                        <div className="text-2xl font-bold text-orange-600">
                          {statisticalInsights.outlier_analysis.outlier_count}
                        </div>
                        <div className="text-sm text-gray-600">Outliers Found</div>
                      </div>
                      <div className="text-center">
                        <div className="text-2xl font-bold text-purple-600">
                          {statisticalInsights.outlier_analysis.outlier_percentage.toFixed(1)}%
                        </div>
                        <div className="text-sm text-gray-600">Of Total Data</div>
                      </div>
                      <div className="text-center">
                        <div className={`text-2xl font-bold capitalize ${
                          statisticalInsights.outlier_analysis.outlier_impact === 'high' ? 'text-red-600' :
                          statisticalInsights.outlier_analysis.outlier_impact === 'moderate' ? 'text-yellow-600' :
                          'text-green-600'
                        }`}>
                          {statisticalInsights.outlier_analysis.outlier_impact}
                        </div>
                        <div className="text-sm text-gray-600">Impact Level</div>
                      </div>
                      <div className="text-center">
                        <div className="text-lg font-bold text-gray-600">
                          {statisticalInsights.outlier_analysis.method_used}
                        </div>
                        <div className="text-sm text-gray-600">Detection Method</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Clustering Tab */}
          {activeTab === 'clustering' && (
            <div className="card">
              <h2 className="text-xl font-semibold text-gray-900 mb-4">Clustering Analysis</h2>
              <p className="text-gray-600 mb-6">
                Group similar data points together to discover hidden patterns and segments in your data.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 hover:bg-blue-50 transition-colors text-center cursor-pointer">
                  <div className="text-2xl mb-2">🎯</div>
                  <h3 className="font-semibold text-gray-900">K-Means</h3>
                  <p className="text-sm text-gray-600">Partition data into k clusters</p>
                </div>

                <div className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-green-400 hover:bg-green-50 transition-colors text-center cursor-pointer">
                  <div className="text-2xl mb-2">🌐</div>
                  <h3 className="font-semibold text-gray-900">DBSCAN</h3>
                  <p className="text-sm text-gray-600">Density-based clustering</p>
                </div>

                <div className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-purple-400 hover:bg-purple-50 transition-colors text-center cursor-pointer">
                  <div className="text-2xl mb-2">🌳</div>
                  <h3 className="font-semibold text-gray-900">Hierarchical</h3>
                  <p className="text-sm text-gray-600">Tree-based clustering</p>
                </div>

                <div className="p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-orange-400 hover:bg-orange-50 transition-colors text-center cursor-pointer">
                  <div className="text-2xl mb-2">📊</div>
                  <h3 className="font-semibold text-gray-900">Gaussian Mixture</h3>
                  <p className="text-sm text-gray-600">Probabilistic clustering</p>
                </div>
              </div>

              <button
                disabled={!selectedFile}
                className={`btn-primary ${!selectedFile ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                Run Clustering Analysis
              </button>
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Available Models */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Available Models</h3>
            <div className="space-y-3">
              {models.map((model) => (
                <div key={model.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div>
                    <h4 className="font-medium text-gray-900 text-sm">{model.name}</h4>
                    <p className="text-xs text-gray-600">{model.type}</p>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${getModelStatusColor(model.status)}`}>
                    {model.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Create New Model */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Create New Model</h3>
            <div className="space-y-3">
              {modelTemplates.map((template, index) => (
                <div key={index} className="p-3 border border-gray-200 rounded-lg">
                  <h4 className="font-medium text-gray-900 text-sm mb-1">{template.name}</h4>
                  <p className="text-xs text-gray-600 mb-2">{template.description}</p>
                  <div className="flex flex-wrap gap-1">
                    {template.algorithms.map((algorithm) => (
                      <button
                        key={algorithm}
                        onClick={() => createModel(template, algorithm)}
                        className="text-xs bg-blue-100 text-blue-700 px-2 py-1 rounded hover:bg-blue-200 transition-colors"
                      >
                        {algorithm}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Stats</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Models:</span>
                <span className="font-medium text-gray-900">{models.length}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Ready Models:</span>
                <span className="font-medium text-green-600">
                  {models.filter(m => m.status === 'ready').length}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Training:</span>
                <span className="font-medium text-blue-600">
                  {models.filter(m => m.status === 'training').length}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedAnalytics;