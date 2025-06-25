import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import type { InternalAxiosRequestConfig } from 'axios';

declare module 'axios' {
  export interface InternalAxiosRequestConfig {
    metadata?: {
      startTime: number;
    };
  }
}

// ====================
// TYPE DEFINITIONS
// ====================

export interface DiscoverySession {
  id: string;
  timestamp: string;
  sources_found: number;
  scan_mode: string;
  duration_ms: number;
  success_rate: number;
}

export interface DiscoverySettings {
  auto_discovery: boolean;
  scan_frequency: 'hourly' | 'daily' | 'weekly';
  confidence_threshold: number;
  include_cloud_sources: boolean;
  scan_locations: string[];
}

export interface DiscoveryInsights {
  recommendations: Array<{
    type: string;
    title: string;
    description: string;
    priority: 'high' | 'medium' | 'low';
    estimated_value: string;
  }>;
  patterns: Array<{
    pattern_type: string;
    description: string;
    frequency: number;
  }>;
}

export interface DataSourceValidation {
  status: 'valid' | 'invalid' | 'warning';
  issues: Array<{
    type: 'connection' | 'schema' | 'data_quality';
    severity: 'low' | 'medium' | 'high';
    message: string;
    suggestion?: string;
  }>;
  health_score: number;
}

export interface AnalysisRequest {
  prompt: string;
  file?: File;
  data_source_id?: string;
  domain?: string;
  use_adaptive?: boolean;
  include_charts?: boolean;
  auto_discover?: boolean;
  max_rows?: number;
}

export interface EnhancedAnalysisRequest extends AnalysisRequest {
  enable_predictive?: boolean;
  enable_anomaly_detection?: boolean;
  enable_correlation_analysis?: boolean;
  enable_statistical_insights?: boolean;
  enable_clustering?: boolean;
  forecast_periods?: number;
  correlation_threshold?: number;
  anomaly_sensitivity?: number;
  clustering_algorithm?: string;
}

export interface AnalysisResponse {
  status: string;
  analysis: {
    type: string;
    summary: string;
    data: any[];
    insights: string[];
    metadata: any;
  };
  query_interpretation: {
    intent: string;
    entities: any[];
    confidence: number;
  };
  comprehensive_report: {
    executive_summary: string;
    detailed_findings: string[];
    recommendations: string[];
    next_steps: string[];
  };
  chart_intelligence: {
    suggested_charts: ChartSuggestion[];
    intent_metadata: any;
    chart_count: number;
  };
  data_discovery: {
    patterns: string[];
    anomalies: string[];
    correlations: any[];
  };
  performance: {
    total_time_ms: number;
    breakdown: {
      data_processing_ms: number;
      analysis_ms: number;
      chart_generation_ms: number;
    };
    data_stats: {
      rows_processed: number;
      columns_analyzed: number;
      data_quality_score: number;
    };
  };
  system_info: {
    version: string;
    capabilities: string[];
    model_info: any;
  };
  timestamp: string;
}

export interface EnhancedAnalysisResponse extends AnalysisResponse {
  advanced_analytics?: {
    predictive_analysis?: any;
    anomaly_detection?: any;
    correlation_analysis?: any;
    statistical_insights?: any;
    clustering_analysis?: any;
  };
  advanced_options_used?: {
    predictive_analysis: boolean;
    anomaly_detection: boolean;
    correlation_analysis: boolean;
    statistical_insights: boolean;
    clustering_analysis: boolean;
  };
  enhanced_performance?: {
    total_enhanced_time_ms: number;
    advanced_modules_run: number;
  };
}

export interface ConversationMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  analysis_results?: any;
  suggestions?: string[];
  requires_clarification?: boolean;
  query_classification?: any;
  attachments?: any[];
}

export interface ConversationStartRequest {
  context?: any;
  user_preferences?: any;
}

export interface ConversationStartResponse {
  status: string;
  session_id: string;
  message: string;
  suggestions?: string[];
  error?: string;
}

export interface ConversationMessageRequest {
  prompt: string;
  file?: File;
  context?: any;
}

export interface ConversationMessageResponse {
  status: string;
  response: ConversationMessage;
  context?: any;
  error?: string;
}

export interface ChartSuggestion {
  id: string;
  type: 'line' | 'bar' | 'pie' | 'scatter' | 'area' | 'heatmap' | 'box';
  title: string;
  description: string;
  data_columns: string[];
  config: any;
  priority: number;
  reasoning: string;
}

export interface DataSource {
  id: string;
  source_id: string;
  type: string;
  confidence: number;
  reasoning: string;
  context: {
    host?: string;
    port?: number;
    database?: string;
    schema?: string;
    table_count?: number;
    size?: string;
    last_modified?: string;
    connection_string?: string;
    [key: string]: any;
  };
  status?: 'discovered' | 'connecting' | 'connected' | 'error' | 'testing';
  connection_params?: any;
  capabilities?: string[];
  metadata?: any;
}

export interface DiscoveryResponse {
  status: string;
  user_id: string;
  discovered_sources: number;
  recommendations: DataSource[];
  metadata: {
    scan_duration_ms: number;
    scan_mode: string;
    environment_scan: boolean;
    total_locations_scanned: number;
    confidence_threshold: number;
  };
  performance_stats: {
    total_time_ms: number;
    sources_per_second: number;
    success_rate: number;
  };
  timestamp: string;
}

export interface SystemStatus {
  status: 'healthy' | 'warning' | 'error';
  timestamp: string;
  version: string;
  uptime_seconds: number;
  components: {
    database: ComponentStatus;
    ai_engine: ComponentStatus;
    smart_defaults: ComponentStatus;
    data_processing: ComponentStatus;
    cache: ComponentStatus;
  };
  capabilities: {
    analysis_types: string[];
    supported_formats: string[];
    max_file_size_mb: number;
    concurrent_analyses: number;
  };
  performance_metrics: {
    avg_analysis_time_ms: number;
    total_analyses_today: number;
    success_rate_24h: number;
    active_connections: number;
  };
  resource_usage: {
    cpu_percent: number;
    memory_percent: number;
    disk_percent: number;
  };
}

export interface ComponentStatus {
  status: 'healthy' | 'warning' | 'error';
  last_check: string;
  response_time_ms?: number;
  error_message?: string;
  version?: string;
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  role: string;
  preferences: {
    default_analysis_mode: string;
    auto_charts: boolean;
    email_notifications: boolean;
    data_retention_days: number;
  };
  usage_stats: {
    total_analyses: number;
    data_sources_connected: number;
    reports_generated: number;
    last_active: string;
  };
  permissions: string[];
}

export interface RecommendationResponse {
  status: string;
  mode: string;
  recommendations: {
    data_sources: DataSourceRecommendation[];
    analysis_suggestions: AnalysisSuggestion[];
    optimization_tips: OptimizationTip[];
  };
  personalization: {
    confidence: number;
    reasoning: string;
    user_pattern: string;
  };
  metadata: any;
}

export interface DataSourceRecommendation {
  source_id: string;
  type: string;
  priority: number;
  reasoning: string;
  estimated_value: string;
  effort_required: string;
  connection_guide: string[];
}

export interface AnalysisSuggestion {
  id: string;
  title: string;
  description: string;
  complexity: 'low' | 'medium' | 'high';
  estimated_time: string;
  required_data: string[];
  expected_insights: string[];
}

export interface OptimizationTip {
  category: string;
  title: string;
  description: string;
  impact: 'low' | 'medium' | 'high';
  effort: 'low' | 'medium' | 'high';
  implementation_steps: string[];
}

export interface APIError {
  error: string;
  message: string;
  code: number;
  details?: any;
  timestamp: string;
}

// ====================
// API CONFIGURATION
// ====================

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
const API_TIMEOUT = parseInt(process.env.REACT_APP_API_TIMEOUT || '30000');
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// ====================
// ANALYTICS API CLASS
// ====================

class AnalyticsAPI {
  private api: AxiosInstance;
  private userId: string;
  private authToken: string | null = null;

  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });

    // Generate or retrieve user ID
    this.userId = this.getOrCreateUserId();

    // Setup interceptors
    this.setupRequestInterceptor();
    this.setupResponseInterceptor();

    console.log(`🔌 Analytics API initialized - Base URL: ${API_BASE_URL}`);
  }

  private getOrCreateUserId(): string {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
      // Server-side or Node.js environment - generate a temporary ID
      return `temp_user_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
    }

    let userId = localStorage.getItem('analytics_user_id');
    if (!userId) {
      userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
      localStorage.setItem('analytics_user_id', userId);
    }
    return userId;
  }

  private setupRequestInterceptor(): void {
    this.api.interceptors.request.use(
      (config) => {
        // Add authentication token (only in browser environment)
        if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
          const token = this.authToken || localStorage.getItem('auth_token') || 'demo-token';
          if (token) {
            config.headers.Authorization = `Bearer ${token}`;
          }
        } else {
          // Fallback token for server-side
          config.headers.Authorization = `Bearer demo-token`;
        }

        // Add user ID to requests (but not for FormData)
        if (config.method === 'post' || config.method === 'put') {
          if (config.data && typeof config.data === 'object' && !(config.data instanceof FormData)) {
            config.data.user_id = this.userId;
          }
        } else {
          config.params = { ...config.params, user_id: this.userId };
        }

        // Add request timestamp
        config.metadata = { startTime: Date.now() };

        console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ Request interceptor error:', error);
        return Promise.reject(error);
      }
    );
  }

  private setupResponseInterceptor(): void {
    this.api.interceptors.response.use(
      (response) => {
        // Calculate request duration
        const startTime = response.config.metadata?.startTime ?? Date.now();
        const duration = Date.now() - startTime;
        console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url} (${duration}ms)`);

        return response;
      },
      async (error: AxiosError) => {
        const duration = Date.now() - (error.config?.metadata?.startTime || Date.now());
        console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} (${duration}ms)`, {
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data
        });

        // Handle authentication errors
        if (error.response?.status === 401) {
          this.handleAuthError();
        }

        // Handle network errors with retry logic
        if (this.shouldRetry(error)) {
          return this.retryRequest(error);
        }

        return Promise.reject(this.formatError(error));
      }
    );
  }

  private shouldRetry(error: AxiosError): boolean {
    const retryableStatuses = [408, 429, 500, 502, 503, 504];
    const isRetryable =
      !error.response ||
      retryableStatuses.includes(error.response.status) ||
      error.code === 'ECONNABORTED' ||
      error.code === 'NETWORK_ERROR';

    const retryCount = (error.config as any)?._retryCount || 0;
    return isRetryable && retryCount < MAX_RETRIES;
  }

  private async retryRequest(error: AxiosError): Promise<any> {
    const config = error.config as any;
    config._retryCount = (config._retryCount || 0) + 1;

    console.log(`🔄 Retrying request (${config._retryCount}/${MAX_RETRIES}): ${config.method?.toUpperCase()} ${config.url}`);

    await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * config._retryCount));
    return this.api.request(config);
  }

  private handleAuthError(): void {
    console.warn('🔐 Authentication error - clearing tokens');
    this.authToken = null;
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  }

  private formatError(error: AxiosError): APIError {
    const apiError: APIError = {
      error: error.name || 'APIError',
      message: error.message || 'An unexpected error occurred',
      code: error.response?.status || 0,
      timestamp: new Date().toISOString()
    };

    if (error.response?.data) {
      apiError.details = error.response.data;
      if (typeof error.response.data === 'object' && 'message' in error.response.data) {
        apiError.message = (error.response.data as any).message;
      }
    }

    return apiError;
  }

  // ====================
  // AUTHENTICATION METHODS
  // ====================

  setAuthToken(token: string): void {
    this.authToken = token;
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      localStorage.setItem('auth_token', token);
    }
  }

  clearAuth(): void {
    this.authToken = null;
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  }

  getUserId(): string {
    return this.userId;
  }

  // ====================
  // HEALTH & STATUS METHODS
  // ====================

  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    try {
      const response = await this.api.get('/health/');
      return response.data;
    } catch (error) {
      console.error('Health check failed:', error);
      return {
        status: 'error',
        timestamp: new Date().toISOString()
      };
    }
  }

  async getSystemStatus(): Promise<SystemStatus> {
    try {
      const response = await this.api.get('/status/');
      return response.data;
    } catch (error) {
      console.error('Failed to get system status:', error);
      throw error;
    }
  }

  async getCapabilities(): Promise<any> {
    try {
      const response = await this.api.get('/capabilities/');
      return response.data;
    } catch (error) {
      console.error('Failed to get capabilities:', error);
      throw error;
    }
  }

  // ====================
  // CONVERSATION METHODS
  // ====================

  async startConversation(request: ConversationStartRequest = {}): Promise<ConversationStartResponse> {
    try {
      console.log('🚀 Starting conversation...');
      const response = await this.api.post('/conversation/start/', {
        context: request.context || {},
        user_preferences: request.user_preferences || {},
        user_id: this.userId
      });

      console.log('✅ Conversation started successfully:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Failed to start conversation:', error);

      return {
        status: 'error',
        session_id: `fallback_${Date.now()}`,
        message: 'Conversation service unavailable. You can still use the analyze features.',
        error: (error as Error).message,
        suggestions: [
          "Try uploading a file for analysis",
          "Ask about data analysis concepts",
          "Switch to wizard mode",
          "Check if backend is running"
        ]
      };
    }
  }

  async sendConversationMessage(sessionId: string, request: ConversationMessageRequest): Promise<ConversationMessageResponse> {
    try {
      console.log('📤 Sending conversation message...', {
        sessionId,
        prompt: request.prompt.substring(0, 100) + (request.prompt.length > 100 ? '...' : ''),
        hasFile: !!request.file,
        fileName: request.file?.name,
        fileSize: request.file?.size
      });

      const formData = new FormData();
      formData.append('prompt', request.prompt);

      if (request.file) {
        console.log('📎 Adding file to conversation:', {
          name: request.file.name,
          size: request.file.size,
          type: request.file.type
        });
        formData.append('file', request.file);
      }

      if (request.context) {
        formData.append('context', JSON.stringify(request.context));
      }

      formData.append('user_id', this.userId);

      const response = await this.api.post(`/conversation/${sessionId}/message/`, formData, {
        headers: {
          // Don't set Content-Type - let browser set it with boundary for FormData
        },
        timeout: 120000,
      });

      console.log('✅ Conversation message sent successfully:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Failed to send conversation message:', error);

      console.log('🔄 Attempting fallback to analyze endpoint...');
      try {
        const fallbackResponse = await this.analyzeWithFallback(request.prompt, request.file);

        const conversationResponse: ConversationMessageResponse = {
          status: 'success',
          response: {
            id: Date.now().toString(),
            role: 'assistant',
            content: fallbackResponse.analysis?.summary || 'Analysis completed successfully',
            timestamp: new Date().toISOString(),
            analysis_results: fallbackResponse,
            suggestions: [
              "Would you like to explore this further?",
              "Should I break this down by segments?",
              "Any specific time periods to focus on?"
            ]
          }
        };

        console.log('✅ Fallback analyze successful');
        return conversationResponse;
      } catch (fallbackError) {
        console.error('❌ Fallback analyze also failed:', fallbackError);
        throw error;
      }
    }
  }

  private async analyzeWithFallback(prompt: string, file?: File): Promise<AnalysisResponse> {
    const endpoints = ['/analyze/', '/quick-analyze/'];

    for (const endpoint of endpoints) {
      try {
        console.log(`🔄 Trying fallback endpoint: ${endpoint}`);

        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('use_adaptive', 'true');
        formData.append('include_charts', 'true');
        formData.append('auto_discover', 'true');

        if (file) {
          formData.append('file', file);
        }

        const response = await this.api.post(endpoint, formData, {
          headers: {
            // Don't set Content-Type for FormData
          },
          timeout: 60000,
        });

        console.log(`✅ Fallback endpoint ${endpoint} successful`);
        return response.data;
      } catch (error) {
        console.warn(`⚠️ Fallback endpoint ${endpoint} failed:`, error);
        continue;
      }
    }

    throw new Error('All analyze endpoints failed');
  }

  // ====================
  // DATA DISCOVERY METHODS
  // ====================

  async discoverDataSources(options: {
    mode?: 'fast' | 'balanced' | 'thorough';
    include_environment_scan?: boolean;
    max_recommendations?: number;
    confidence_threshold?: number;
    scan_locations?: string[];
  } = {}): Promise<DiscoveryResponse> {
    try {
      const params = new URLSearchParams({
        mode: options.mode || 'balanced',
        include_environment_scan: String(options.include_environment_scan ?? true),
        max_recommendations: String(options.max_recommendations || 10),
        confidence_threshold: String(options.confidence_threshold || 0.5),
      });

      if (options.scan_locations?.length) {
        params.append('scan_locations', options.scan_locations.join(','));
      }

      const response = await this.api.get(`/discover-sources/?${params}`);
      return response.data;
    } catch (error) {
      console.error('Data source discovery failed:', error);

      return {
        status: 'success',
        user_id: this.userId,
        discovered_sources: 3,
        recommendations: [
          {
            id: 'mock_csv_1',
            source_id: 'sample_sales_data',
            type: 'CSV File',
            confidence: 0.85,
            reasoning: 'Mock data source for development',
            context: { size: '2.3MB', rows: 1500 }
          },
          {
            id: 'mock_db_1',
            source_id: 'local_database',
            type: 'SQLite',
            confidence: 0.70,
            reasoning: 'Development database',
            context: { host: 'localhost', database: 'analytics' }
          },
          {
            id: 'mock_api_1',
            source_id: 'rest_api_endpoint',
            type: 'REST API',
            confidence: 0.60,
            reasoning: 'Sample API endpoint',
            context: { endpoint: '/api/data' }
          }
        ],
        metadata: {
          scan_duration_ms: 1200,
          scan_mode: options.mode || 'balanced',
          environment_scan: true,
          total_locations_scanned: 5,
          confidence_threshold: 0.5
        },
        performance_stats: {
          total_time_ms: 1200,
          sources_per_second: 2.5,
          success_rate: 1.0
        },
        timestamp: new Date().toISOString()
      };
    }
  }

  // ====================
  // ANALYSIS METHODS
  // ====================

  async analyze(request: AnalysisRequest): Promise<AnalysisResponse> {
    try {
      console.log('🔬 Starting analysis...', {
        prompt: request.prompt.substring(0, 100) + (request.prompt.length > 100 ? '...' : ''),
        hasFile: !!request.file,
        fileName: request.file?.name,
        dataSourceId: request.data_source_id
      });

      const formData = new FormData();
      formData.append('prompt', request.prompt);

      if (request.file) {
        console.log('📎 Adding file to analysis:', {
          name: request.file.name,
          size: request.file.size,
          type: request.file.type,
          lastModified: new Date(request.file.lastModified).toISOString()
        });
        formData.append('file', request.file);
      }

      const optionalParams = {
        data_source_id: request.data_source_id,
        domain: request.domain,
        use_adaptive: request.use_adaptive ?? true,
        include_charts: request.include_charts ?? true,
        auto_discover: request.auto_discover ?? true,
        max_rows: request.max_rows || 10000,
      };

      // Use Object.entries with proper iteration
      const paramEntries = Object.entries(optionalParams);
      for (let i = 0; i < paramEntries.length; i++) {
        const [key, value] = paramEntries[i];
        if (value !== undefined) {
          formData.append(key, String(value));
        }
      }

      console.log('📦 Analysis FormData contents:');
      // Fix the FormData iteration issue
      const formDataEntries: [string, FormDataEntryValue][] = [];
      formData.forEach((value, key) => {
        formDataEntries.push([key, value]);
      });

      for (let i = 0; i < formDataEntries.length; i++) {
        const [key, value] = formDataEntries[i];
        if (value instanceof File) {
          console.log(`  ${key}: File(${value.name}, ${value.size} bytes, ${value.type})`);
        } else {
          console.log(`  ${key}: ${value}`);
        }
      }

      const response = await this.api.post('/analyze/', formData, {
        headers: {
          // Don't set Content-Type for FormData - browser handles this
        },
        timeout: 60000,
      });

      console.log('✅ Analysis completed successfully');
      return response.data;
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      throw error;
    }
  }

  // ====================
  // NEW: FORMDATA ANALYZE METHOD FOR CONVERSATIONAL ANALYTICS
  // ====================

  async analyzeData(formData: FormData): Promise<AnalysisResponse> {
    try {
      console.log('📡 FormData analyzeData called...');

      // Extract data from FormData
      const prompt = formData.get('prompt') as string;
      const file = formData.get('file') as File | null;

      if (!prompt) {
        throw new Error('Prompt is required');
      }

      console.log('📊 Extracted from FormData:', {
        prompt: prompt.substring(0, 50) + (prompt.length > 50 ? '...' : ''),
        hasFile: !!file,
        fileName: file?.name,
        fileSize: file ? formatFileSize(file.size) : 'N/A'
      });

      // Convert to our existing analyze method format
      const analysisRequest: AnalysisRequest = {
        prompt: prompt,
        file: file || undefined,
        use_adaptive: true,
        include_charts: true,
        auto_discover: true,
        max_rows: 10000
      };

      // Use the existing analyze method
      const response = await this.analyze(analysisRequest);

      console.log('✅ FormData analyzeData successful');
      return response;

    } catch (error) {
      console.error('❌ FormData analyzeData failed:', error);

      // Return a properly structured error response
      return {
        status: 'error',
        analysis: {
          type: 'error',
          summary: 'Analysis failed - backend unavailable',
          data: [],
          insights: [
            'Backend server is not responding',
            'Check if the server is running on http://localhost:8000',
            'Verify file format is supported (CSV, Excel, JSON)',
            'Try with a smaller file or simpler query'
          ],
          metadata: {
            error: extractErrorMessage(error),
            timestamp: new Date().toISOString()
          }
        },
        query_interpretation: {
          intent: 'error_fallback',
          entities: [],
          confidence: 0.0
        },
        comprehensive_report: {
          executive_summary: 'Analysis could not be completed due to backend connectivity issues',
          detailed_findings: ['Backend service unavailable'],
          recommendations: [
            'Check backend server status',
            'Verify network connectivity',
            'Try again in a few moments'
          ],
          next_steps: [
            'Restart the backend server',
            'Check server logs for errors',
            'Verify API endpoints are working'
          ]
        },
        chart_intelligence: {
          suggested_charts: [],
          intent_metadata: {},
          chart_count: 0
        },
        data_discovery: {
          patterns: [],
          anomalies: ['Backend connectivity error'],
          correlations: []
        },
        performance: {
          total_time_ms: 0,
          breakdown: {
            data_processing_ms: 0,
            analysis_ms: 0,
            chart_generation_ms: 0
          },
          data_stats: {
            rows_processed: 0,
            columns_analyzed: 0,
            data_quality_score: 0.0
          }
        },
        system_info: {
          version: 'unknown',
          capabilities: [],
          model_info: {}
        },
        timestamp: new Date().toISOString()
      };
    }
  }

  async getAnalyticsDashboard(): Promise<{
    status: string;
    dashboard_data: {
      total_analyses: number;
      success_rate: number;
      avg_processing_time: number;
      top_data_sources: Array<{ name: string; usage_count: number }>;
      recent_activities: Array<{
        id: string;
        type: string;
        description: string;
        timestamp: string;
        status: string;
      }>;
      performance_metrics: {
        analyses_this_week: number;
        data_processed_gb: number;
        insights_generated: number;
        charts_created: number;
      };
    };
    timestamp: string;
  }> {
    try {
      const response = await this.api.get('/analytics-dashboard/');
      return response.data;
    } catch (error) {
      console.error('Failed to get analytics dashboard:', error);
      throw error;
    }
  }

  async generateReport(reportType: string, options: any): Promise<any> {
    try {
      const response = await this.api.post('/generate-report/', {
        report_type: reportType,
        options: options
      });
      return response.data;
    } catch (error) {
      console.error('Failed to generate report:', error);
      throw error;
    }
  }

  // ====================
  // DEBUG AND TESTING METHODS
  // ====================

  async testFileUpload(file: File): Promise<{
    status: 'success' | 'error';
    message: string;
    details?: any;
  }> {
    try {
      console.log('🧪 Testing file upload...', {
        name: file.name,
        size: file.size,
        type: file.type
      });

      const formData = new FormData();
      formData.append('prompt', 'Test file upload - please analyze this data');
      formData.append('file', file);
      formData.append('max_rows', '100');

      const response = await this.api.post('/quick-analyze/', formData, {
        headers: {
          // Don't set Content-Type for FormData
        },
        timeout: 30000,
      });

      console.log('✅ File upload test successful');
      return {
        status: 'success',
        message: 'File upload and processing successful',
        details: response.data
      };
    } catch (error) {
      console.error('❌ File upload test failed:', error);
      return {
        status: 'error',
        message: `File upload failed: ${(error as Error).message}`,
        details: error
      };
    }
  }

  async getDiagnostics(): Promise<{
    backend_health: any;
    connectivity_test: any;
    endpoints_available: Record<string, boolean>;
    file_upload_support: boolean;
    conversation_support: boolean;
  }> {
    const diagnostics = {
      backend_health: null as any,
      connectivity_test: null as any,
      endpoints_available: {} as Record<string, boolean>,
      file_upload_support: false,
      conversation_support: false
    };

    try {
      diagnostics.backend_health = await this.healthCheck();
    } catch (error) {
      diagnostics.backend_health = { status: 'error', error: (error as Error).message };
    }

    try {
      diagnostics.connectivity_test = await this.ping();
    } catch (error) {
      diagnostics.connectivity_test = { status: 'error', error: (error as Error).message };
    }

    const endpointsToTest = [
      '/health/',
      '/analyze/',
      '/quick-analyze/',
      '/conversation/start/',
      '/discover-sources/',
      '/capabilities/'
    ];

    for (let i = 0; i < endpointsToTest.length; i++) {
      const endpoint = endpointsToTest[i];
      try {
        if (endpoint === '/conversation/start/') {
          await this.startConversation();
          diagnostics.endpoints_available[endpoint] = true;
          diagnostics.conversation_support = true;
        } else {
          await this.api.get(endpoint);
          diagnostics.endpoints_available[endpoint] = true;
        }
      } catch (error) {
        diagnostics.endpoints_available[endpoint] = false;
      }
    }

    diagnostics.file_upload_support =
      diagnostics.endpoints_available['/analyze/'] ||
      diagnostics.endpoints_available['/quick-analyze/'];

    return diagnostics;
  }

  // ====================
  // DATA SOURCE MANAGEMENT METHODS
  // ====================

  async getConnectedSources(): Promise<DataSource[]> {
    try {
      const response = await this.api.get('/data-sources/connected/');
      return response.data.sources || [];
    } catch (error) {
      console.error('Failed to get connected sources:', error);
      return [];
    }
  }

  async getDiscoveryHistory(): Promise<{ discovery_sessions: DiscoverySession[] }> {
    try {
      const response = await this.api.get('/discovery/history/');
      return response.data;
    } catch (error) {
      console.error('Failed to get discovery history:', error);
      return { discovery_sessions: [] };
    }
  }

  async getDiscoverySettings(): Promise<{ settings: DiscoverySettings }> {
    try {
      const response = await this.api.get('/discovery/settings/');
      return response.data;
    } catch (error) {
      console.error('Failed to get discovery settings:', error);
      return {
        settings: {
          auto_discovery: true,
          scan_frequency: 'daily',
          confidence_threshold: 0.7,
          include_cloud_sources: false,
          scan_locations: []
        }
      };
    }
  }

  async saveDiscoverySettings(settings: DiscoverySettings): Promise<void> {
    try {
      await this.api.post('/discovery/settings/', { settings });
    } catch (error) {
      console.error('Failed to save discovery settings:', error);
      throw error;
    }
  }

  async getDiscoveryInsights(): Promise<{ insights: DiscoveryInsights }> {
    try {
      const response = await this.api.get('/discovery/insights/');
      return response.data;
    } catch (error) {
      console.error('Failed to get discovery insights:', error);
      return {
        insights: {
          recommendations: [],
          patterns: []
        }
      };
    }
  }

  async testConnection(sourceId: string, connectionParams: any): Promise<{ status: string; message?: string; details?: any }> {
    try {
      const response = await this.api.post(`/data-sources/${sourceId}/test-connection/`, {
        connection_params: connectionParams
      });
      return response.data;
    } catch (error) {
      console.error('Connection test failed:', error);
      return {
        status: 'error',
        message: 'Connection test failed',
        details: error
      };
    }
  }

  async validateDataSource(sourceId: string): Promise<DataSourceValidation> {
    try {
      const response = await this.api.get(`/data-sources/${sourceId}/validate/`);
      return response.data;
    } catch (error) {
      console.error('Data source validation failed:', error);
      return {
        status: 'invalid',
        issues: [{
          type: 'connection',
          severity: 'high',
          message: 'Validation failed',
          suggestion: 'Check connection parameters'
        }],
        health_score: 0.0
      };
    }
  }

  async connectDataSource(sourceId: string, connectionParams: any): Promise<void> {
    try {
      await this.api.post(`/data-sources/${sourceId}/connect/`, {
        connection_params: connectionParams
      });
    } catch (error) {
      console.error('Failed to connect data source:', error);
      throw error;
    }
  }

  async disconnectDataSource(sourceId: string): Promise<void> {
    try {
      await this.api.post(`/data-sources/${sourceId}/disconnect/`);
    } catch (error) {
      console.error('Failed to disconnect data source:', error);
      throw error;
    }
  }

  async refreshDataSource(sourceId: string): Promise<void> {
    try {
      await this.api.post(`/data-sources/${sourceId}/refresh/`);
    } catch (error) {
      console.error('Failed to refresh data source:', error);
      throw error;
    }
  }

  // ====================
  // USER PREFERENCES METHODS
  // ====================

  async updateUserPreferences(preferences: any): Promise<void> {
    try {
      await this.api.post('/user/preferences/', preferences);
    } catch (error) {
      console.error('Failed to update user preferences:', error);
      throw error;
    }
  }

  async getRecommendations(): Promise<any> {
    try {
      const response = await this.api.get('/recommendations/');
      return response.data;
    } catch (error) {
      console.error('Failed to get recommendations:', error);
      return { recommendations: [] };
    }
  }

  // ====================
  // UTILITY METHODS
  // ====================

  async ping(): Promise<{ status: string; response_time: number }> {
    const startTime = Date.now();
    try {
      await this.api.get('/ping/');
      return {
        status: 'success',
        response_time: Date.now() - startTime
      };
    } catch (error) {
      return {
        status: 'error',
        response_time: Date.now() - startTime
      };
    }
  }

  getBaseURL(): string {
    return API_BASE_URL;
  }

  isConnected(): boolean {
    if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
      return this.authToken !== null;
    }
    return this.authToken !== null || localStorage.getItem('auth_token') !== null;
  }

  async withRetry<T>(
    operation: () => Promise<T>,
    maxRetries: number = 3,
    delay: number = 1000
  ): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        return await operation();
      } catch (error) {
        lastError = error as Error;

        if (attempt === maxRetries) {
          throw lastError;
        }

        console.log(`🔄 Retry attempt ${attempt}/${maxRetries} failed, waiting ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay * attempt));
      }
    }

    throw lastError!;
  }
}

// ====================
// SINGLETON INSTANCE
// ====================

export const analyticsAPI = new AnalyticsAPI();

// ====================
// CONVERSATION API WRAPPER
// ====================

export const conversationAPI = {
  async startConversation(): Promise<ConversationStartResponse> {
    return analyticsAPI.startConversation();
  },

  async sendMessage(sessionId: string, message: string, file?: File): Promise<ConversationMessageResponse> {
    return analyticsAPI.sendConversationMessage(sessionId, {
      prompt: message,
      file: file,
      context: {}
    });
  },

  async sendMessageFallback(message: string, file?: File): Promise<ConversationMessageResponse> {
    try {
      const result = await analyticsAPI.analyze({
        prompt: message,
        file: file,
        use_adaptive: true,
        include_charts: true,
        auto_discover: true
      });

      return {
        status: 'success',
        response: {
          id: Date.now().toString(),
          role: 'assistant',
          content: result.analysis?.summary || 'Analysis completed successfully',
          timestamp: new Date().toISOString(),
          analysis_results: result,
          suggestions: [
            "What would you like to explore further?",
            "Should I break this down differently?",
            "Any specific aspects to focus on?"
          ]
        }
      };
    } catch (error) {
      throw error;
    }
  }
};

// ====================
// LEGACY COMPATIBILITY API SERVICE
// ====================

export const apiService = {
  async discoverSources() {
    try {
      const response = await analyticsAPI.discoverDataSources({
        mode: 'balanced',
        include_environment_scan: true,
        max_recommendations: 10
      });

      return {
        status: 'success',
        discovered_sources: response.discovered_sources,
        recommendations: response.recommendations
      };
    } catch (error) {
      console.error('Error discovering sources:', error);
      return {
        status: 'error',
        discovered_sources: 0,
        recommendations: []
      };
    }
  },

  async analyzeData(query: string, file: File | null = null, selectedSources: string[] = []) {
    try {
      console.log('📡 Legacy analyzeData called with:', {
        query: query.substring(0, 50) + (query.length > 50 ? '...' : ''),
        hasFile: !!file,
        fileName: file?.name,
        selectedSources: selectedSources.length
      });

      const response = await analyticsAPI.analyze({
        prompt: query,
        file: file || undefined,
        data_source_id: selectedSources[0],
        use_adaptive: true,
        include_charts: true,
        auto_discover: selectedSources.length === 0
      });

      console.log('✅ Legacy analyzeData successful:', response);
      return response;

    } catch (error) {
      console.error('💥 Legacy analyzeData error:', error);
      throw error;
    }
  }
};

export default analyticsAPI;

// ====================
// UTILITY FUNCTIONS
// ====================

export const formatFileSize = (bytes: number): string => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
};

export const isValidFileType = (file: File, allowedTypes: string[]): boolean => {
  const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
  const allowedExtensions = ['.csv', '.xls', '.xlsx', '.json'];

  return allowedTypes.includes(file.type) || allowedExtensions.includes(fileExtension);
};

export const createDownloadLink = (blob: Blob, filename: string): void => {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    console.warn('Download functionality not available in this environment');
    return;
  }

  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};

export const handleAPIError = (error: APIError, fallbackMessage: string = 'An error occurred'): string => {
  if (error.details && typeof error.details === 'object' && error.details.message) {
    return error.details.message;
  }
  return error.message || fallbackMessage;
};

export const retryOperation = async <T>(
  operation: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  return analyticsAPI.withRetry(operation, maxRetries, delay);
};

export const formatTimestamp = (timestamp: string | Date): string => {
  const date = typeof timestamp === 'string' ? new Date(timestamp) : timestamp;
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
  return `${Math.floor(diffMins / 1440)}d ago`;
};

export const validateFile = (file: File): { isValid: boolean; error?: string } => {
  const maxSize = 100 * 1024 * 1024; // 100MB
  if (file.size > maxSize) {
    return {
      isValid: false,
      error: `File too large: ${formatFileSize(file.size)}. Maximum size is ${formatFileSize(maxSize)}.`
    };
  }

  const allowedTypes = [
    'text/csv',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/json'
  ];

  const allowedExtensions = ['.csv', '.xls', '.xlsx', '.json'];
  const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));

  if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
    return {
      isValid: false,
      error: `Unsupported file type: ${file.type || fileExtension}. Please use CSV, Excel, or JSON files.`
    };
  }

  if (file.name.length > 255) {
    return {
      isValid: false,
      error: 'File name too long. Please use a shorter filename.'
    };
  }

  return { isValid: true };
};

export const getFileInfo = (file: File) => {
  return {
    name: file.name,
    size: file.size,
    sizeFormatted: formatFileSize(file.size),
    type: file.type,
    extension: file.name.toLowerCase().substring(file.name.lastIndexOf('.')),
    lastModified: new Date(file.lastModified).toISOString(),
    isValid: validateFile(file).isValid
  };
};

export const formatConversationMessage = (content: string, role: 'user' | 'assistant'): ConversationMessage => {
  return {
    id: Date.now().toString() + '_' + Math.random().toString(36).substr(2, 8),
    role,
    content,
    timestamp: new Date().toISOString()
  };
};

export const extractErrorMessage = (error: any): string => {
  if (typeof error === 'string') return error;

  if (error?.response?.data?.message) return error.response.data.message;
  if (error?.response?.data?.error) return error.response.data.error;
  if (error?.message) return error.message;
  if (error?.error) return error.error;

  return 'An unexpected error occurred';
};

export const createErrorResponse = (error: any, context?: string): ConversationMessage => {
  const errorMessage = extractErrorMessage(error);
  const contextualMessage = context
    ? `${context}: ${errorMessage}`
    : errorMessage;

  return {
    id: Date.now().toString() + '_error',
    role: 'assistant',
    content: `I encountered an error: ${contextualMessage}`,
    timestamp: new Date().toISOString(),
    suggestions: [
      "Try rephrasing your question",
      "Check your file format",
      "Try again with a smaller file",
      "Switch to wizard mode"
    ]
  };
};

export const measurePerformance = async <T>(
  operation: () => Promise<T>,
  operationName: string
): Promise<{ result: T; duration: number }> => {
  const startTime = performance.now();

  try {
    const result = await operation();
    const duration = performance.now() - startTime;

    console.log(`📊 Performance: ${operationName} completed in ${duration.toFixed(2)}ms`);

    return { result, duration };
  } catch (error) {
    const duration = performance.now() - startTime;
    console.error(`📊 Performance: ${operationName} failed after ${duration.toFixed(2)}ms`, error);
    throw error;
  }
};

// ====================
// CACHE MANAGEMENT
// ====================

class APICache {
  private cache = new Map<string, { data: any; expiry: number }>();
  private defaultTTL = 5 * 60 * 1000; // 5 minutes

  set(key: string, data: any, ttl = this.defaultTTL): void {
    const expiry = Date.now() + ttl;
    this.cache.set(key, { data, expiry });
  }

  get<T>(key: string): T | null {
    const item = this.cache.get(key);

    if (!item) return null;

    if (Date.now() > item.expiry) {
      this.cache.delete(key);
      return null;
    }

    return item.data as T;
  }

  has(key: string): boolean {
    return this.get(key) !== null;
  }

  delete(key: string): void {
    this.cache.delete(key);
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    const now = Date.now();
    // Clean expired items first
    const expiredKeys: string[] = [];
    this.cache.forEach((item, key) => {
      if (now > item.expiry) {
        expiredKeys.push(key);
      }
    });

    for (let i = 0; i < expiredKeys.length; i++) {
      this.cache.delete(expiredKeys[i]);
    }

    return this.cache.size;
  }
}

export const apiCache = new APICache();

export const withCache = async <T>(
  key: string,
  operation: () => Promise<T>,
  ttl?: number
): Promise<T> => {
  const cached = apiCache.get<T>(key);
  if (cached !== null) {
    console.log(`📦 Cache hit: ${key}`);
    return cached;
  }

  console.log(`🔄 Cache miss: ${key}, executing operation...`);
  const result = await operation();
  apiCache.set(key, result, ttl);

  return result;
};

// ====================
// ENVIRONMENT DETECTION
// ====================

export const getEnvironmentInfo = () => {
  const isDevelopment = process.env.NODE_ENV === 'development';
  const isProduction = process.env.NODE_ENV === 'production';
  const isBrowser = typeof window !== 'undefined';
  const hasLocalStorage = isBrowser && typeof localStorage !== 'undefined';

  return {
    isDevelopment,
    isProduction,
    isBrowser,
    hasLocalStorage,
    apiBaseUrl: API_BASE_URL,
    userAgent: isBrowser ? window.navigator.userAgent : 'Server',
    timestamp: new Date().toISOString()
  };
};

export const logEvent = (event: string, data?: any) => {
  const environment = getEnvironmentInfo();

  const logEntry = {
    event,
    data,
    timestamp: new Date().toISOString(),
    environment: environment.isDevelopment ? 'development' : 'production',
    userAgent: environment.userAgent
  };

  if (environment.isDevelopment) {
    console.log(`📊 Event: ${event}`, logEntry);
  }
};

export const getFeatureFlags = () => {
  return {
    enableConversationMode: true,
    enableAdvancedAnalytics: true,
    enableFileUpload: true,
    enableDataSourceDiscovery: true,
    enableCaching: true,
    enableDebugMode: getEnvironmentInfo().isDevelopment,
    maxFileSize: 100 * 1024 * 1024, // 100MB
    supportedFileTypes: ['.csv', '.xlsx', '.xls', '.json'],
    defaultTimeout: API_TIMEOUT,
    maxRetries: MAX_RETRIES
  };
};

export const initializeAPI = async (): Promise<{
  status: 'success' | 'partial' | 'failed';
  capabilities: any;
  diagnostics: any;
}> => {
  try {
    console.log('🚀 Initializing DataGenie API...');

    const diagnostics = await analyticsAPI.getDiagnostics();

    let capabilities = {};
    try {
      capabilities = await analyticsAPI.getCapabilities();
    } catch (error) {
      console.warn('⚠️ Could not fetch capabilities:', error);
      capabilities = getFeatureFlags();
    }

    const status = diagnostics.backend_health?.status === 'error' ? 'failed' :
                  (!diagnostics.conversation_support || !diagnostics.file_upload_support) ? 'partial' : 'success';

    console.log(`✅ API initialization ${status}:`, { capabilities, diagnostics });

    logEvent('api_initialization', { status, capabilities, diagnostics });

    return { status, capabilities, diagnostics };
  } catch (error) {
    console.error('❌ API initialization failed:', error);
    logEvent('api_initialization_failed', { error: extractErrorMessage(error) });

    return {
      status: 'failed',
      capabilities: getFeatureFlags(),
      diagnostics: { error: extractErrorMessage(error) }
    };
  }
};

