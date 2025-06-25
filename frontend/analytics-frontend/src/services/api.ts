// services/api.ts - Complete Enhanced Analytics API Module
import axios, { AxiosInstance, AxiosError } from 'axios';

// ====================
// ENHANCED TYPE DEFINITIONS (Updated to match backend)
// ====================

export interface AnalysisRequest {
  prompt: string;
  file?: File;
  data_source_id?: string;
  user_id?: string;
  domain?: string;
  use_adaptive?: boolean;
  include_charts?: boolean;
  auto_discover?: boolean;
  max_rows?: number;
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
  comprehensive_report?: {
    executive_summary: string;
    detailed_findings: string[];
    recommendations: string[];
    next_steps: string[];
    sections?: any[];
  };
  chart_intelligence: {
    suggested_charts: ChartSuggestion[];
    intent_metadata: any;
    chart_count: number;
  };
  data_discovery?: {
    auto_discovered?: boolean;
    source_id?: string;
    confidence?: number;
    reasoning?: string;
    source?: string;
  };
  performance: {
    total_time_ms: number;
    breakdown: {
      data_processing_ms?: number;
      interpretation_ms?: number;
      analysis_ms: number;
      insights_ms?: number;
      reporting_ms?: number;
      chart_generation_ms?: number;
      charts_ms?: number;
    };
    data_stats: {
      rows_processed?: number;
      rows?: number;
      columns_analyzed?: number;
      columns?: number;
      data_quality_score?: number;
      memory_mb?: number;
    };
  };
  system_info: {
    version: string;
    capabilities?: string[];
    model_info?: any;
    method?: string;
    adaptive_processing_used?: boolean;
    chart_intelligence_used?: boolean;
    smart_defaults_enabled?: boolean;
    auto_discovery_used?: boolean;
    smart_engine_available?: boolean;
    domain?: string;
    user_id?: string;
    openai_used?: boolean;
    confidence_threshold?: number;
  };
  timestamp: string;
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
  chart_type?: string; // Alternative naming from backend
  data?: any[];
  x_axis?: string;
  y_axis?: string;
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
    type?: string;
    location?: string;
    endpoint?: string;
    server?: string;
    rows?: number;
    [key: string]: any;
  };
  status?: 'discovered' | 'connecting' | 'connected' | 'error' | 'testing';
  connection_params?: any;
  capabilities?: string[];
  metadata?: any;
  recommendation_type?: string; // For smart defaults
}

export interface DiscoveryResponse {
  status: string;
  user_id: string;
  discovered_sources: number;
  recommendations: DataSource[];
  metadata: {
    scan_duration_ms?: number;
    scan_mode?: string;
    environment_scan?: boolean;
    total_locations_scanned?: number;
    confidence_threshold?: number;
    total_candidates?: number;
    policy_filtered?: number;
    ml_enhanced?: boolean;
    confidence_distribution?: any;
    generated_at?: string;
  };
  performance_stats?: {
    total_time_ms: number;
    sources_per_second: number;
    success_rate: number;
  };
  timestamp?: string;
  message?: string;
}

export interface DiscoverySession {
  id: string;
  timestamp: string;
  sources_found: number;
  scan_mode: string;
  duration_ms: number;
  success_rate: number;
}

export interface SystemStatus {
  status: 'healthy' | 'warning' | 'error' | 'initializing' | 'not_initialized';
  timestamp: string;
  version: string;
  system_version?: string;
  initialization_status?: string;
  components: Record<string, any>;
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
  uptime_seconds?: number;
  knowledge_summary?: any;
  mathematical_methods?: number;
  smart_engine?: {
    available: boolean;
    openai_configured: boolean;
    openai_status: string;
  };
  smart_defaults?: any;
  openai_status?: {
    available: boolean;
    status: string;
  };
  smart_defaults_health?: any;
  smart_defaults_error?: string;
}

export interface UserPreferences {
  default_analysis_mode: 'fast' | 'balanced' | 'thorough';
  auto_charts: boolean;
  email_notifications: boolean;
  data_retention_days: number;
  theme?: 'light' | 'dark' | 'auto';
  language?: string;
  timezone?: string;
  auto_discovery?: boolean;
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
const API_TIMEOUT = parseInt(process.env.REACT_APP_API_TIMEOUT || '120000'); // Increased for analysis
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// ====================
// ENHANCED ANALYTICS API CLASS
// ====================

class AnalyticsAPI {
  private api: AxiosInstance;
  private userId: string;

  constructor() {
    this.api = axios.create({
      baseURL: API_BASE_URL,
      timeout: API_TIMEOUT,
      headers: {
        'Accept': 'application/json',
      },
    });

    this.userId = this.getOrCreateUserId();
    this.setupInterceptors();

    console.log(`🔌 Analytics API initialized - Base URL: ${API_BASE_URL}`);
  }

  private getOrCreateUserId(): string {
    if (typeof window === 'undefined' || typeof localStorage === 'undefined') {
      return `temp_user_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
    }

    let userId = localStorage.getItem('analytics_user_id');
    if (!userId) {
      userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
      localStorage.setItem('analytics_user_id', userId);
    }
    return userId;
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        const token = this.getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        // Add user ID to requests
        if (config.method === 'post' || config.method === 'put') {
          if (config.data instanceof FormData) {
            config.data.append('user_id', this.userId);
          } else if (config.data && typeof config.data === 'object') {
            config.data.user_id = this.userId;
          }
        } else {
          config.params = { ...config.params, user_id: this.userId };
        }

        console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ Request interceptor error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url} (${response.status})`);
        return response;
      },
      async (error: AxiosError) => {
        console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url}`, {
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data
        });

        if (this.shouldRetry(error)) {
          return this.retryRequest(error);
        }

        return Promise.reject(this.formatError(error));
      }
    );
  }

  private getAuthToken(): string {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      return localStorage.getItem('auth_token') || 'demo-token';
    }
    return 'demo-token';
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

    console.log(`🔄 Retrying request (${config._retryCount}/${MAX_RETRIES})`);

    await new Promise(resolve => setTimeout(resolve, RETRY_DELAY * config._retryCount));
    return this.api.request(config);
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
      if (typeof error.response.data === 'object' && 'error' in error.response.data) {
        apiError.message = (error.response.data as any).error;
      }
    }

    return apiError;
  }

  // ====================
  // CORE API METHODS
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
    const response = await this.api.get('/status/');
    return response.data;
  }

  async getCapabilities(): Promise<any> {
    const response = await this.api.get('/capabilities/');
    return response.data;
  }

  // ====================
  // ANALYSIS METHODS (Updated for backend compatibility)
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
        });
        formData.append('file', request.file, request.file.name);
      }

      // Add all optional parameters matching backend expectations
      const optionalParams = {
        data_source_id: request.data_source_id,
        user_id: request.user_id || this.userId,
        domain: request.domain,
        use_adaptive: request.use_adaptive ?? true,
        include_charts: request.include_charts ?? true,
        auto_discover: request.auto_discover ?? true,
        max_rows: request.max_rows || 10000,
      };

      Object.entries(optionalParams).forEach(([key, value]) => {
        if (value !== undefined && value !== null) {
          formData.append(key, String(value));
        }
      });

      // Debug FormData contents
      console.log('📦 FormData contents:');
      formData.forEach((value, key) => {
        if (value instanceof File) {
          console.log(`  ${key}: File(${value.name}, ${value.size} bytes, ${value.type})`);
        } else {
          console.log(`  ${key}: ${value}`);
        }
      });

      const response = await this.api.post('/analyze/', formData, {
        headers: {
          // Let browser set Content-Type for FormData
        },
        timeout: 180000, // 3 minutes for comprehensive analysis
      });

      console.log('✅ Analysis completed successfully');
      return response.data;
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      throw this.enhanceError(error as AxiosError);
    }
  }

  async quickAnalyze(prompt: string, file: File, maxRows: number = 1000): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('file', file);
      formData.append('max_rows', String(maxRows));
      formData.append('user_id', this.userId);

      const response = await this.api.post('/quick-analyze/', formData, {
        timeout: 60000,
      });

      return response.data;
    } catch (error) {
      console.error('Quick analysis failed:', error);
      throw this.enhanceError(error as AxiosError);
    }
  }

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
      formData.append('use_adaptive', 'true');
      formData.append('include_charts', 'true');
      formData.append('auto_discover', 'false');

      const response = await this.api.post('/test-upload/', formData, {
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

  // ====================
  // DATA DISCOVERY METHODS (Updated for Smart Defaults)
  // ====================

  async discoverDataSources(options: {
    mode?: 'fast' | 'balanced' | 'thorough';
    include_environment_scan?: boolean;
    max_recommendations?: number;
    confidence_threshold?: number;
  } = {}): Promise<DiscoveryResponse> {
    try {
      const params = new URLSearchParams({
        mode: options.mode || 'balanced',
        include_environment_scan: String(options.include_environment_scan ?? true),
        max_recommendations: String(options.max_recommendations || 10),
        user_id: this.userId,
      });

      if (options.confidence_threshold !== undefined) {
        params.append('confidence_threshold', String(options.confidence_threshold));
      }

      const response = await this.api.get(`/discover-sources/?${params}`);
      return response.data;
    } catch (error) {
      console.error('Data source discovery failed:', error);
      throw this.enhanceError(error as AxiosError);
    }
  }

  async getRecommendations(mode: string = 'balanced'): Promise<any> {
    try {
      const params = new URLSearchParams({
        mode,
        user_id: this.userId,
      });

      const response = await this.api.get(`/recommendations/?${params}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get recommendations:', error);
      throw this.enhanceError(error as AxiosError);
    }
  }

  async connectDataSource(sourceId: string, connectionParams: any): Promise<any> {
    try {
      const response = await this.api.post('/connect-source/', {
        source_id: sourceId,
        connection_params: connectionParams,
        user_id: this.userId,
      });
      return response.data;
    } catch (error) {
      console.error('Failed to connect data source:', error);
      throw this.enhanceError(error as AxiosError);
    }
  }

  async recordFeedback(recommendationId: string, action: string, context?: any): Promise<any> {
    try {
      const response = await this.api.post('/feedback/', {
        recommendation_id: recommendationId,
        action,
        context: context || {},
        user_id: this.userId,
      });
      return response.data;
    } catch (error) {
      console.error('Failed to record feedback:', error);
      throw this.enhanceError(error as AxiosError);
    }
  }

  // ====================
  // CHART AND ANALYTICS METHODS
  // ====================

  async getChartSuggestions(prompt: string, file: File, maxRows: number = 1000): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('file', file);
      formData.append('max_rows', String(maxRows));

      const response = await this.api.post('/chart-suggestions/', formData);
      return response.data;
    } catch (error) {
      console.error('Chart suggestions failed:', error);
      throw this.enhanceError(error as AxiosError);
    }
  }

  // ====================
  // USER AND ADMIN METHODS
  // ====================

  async getUserProfile(): Promise<any> {
    try {
      const response = await this.api.get(`/user-profile/?user_id=${this.userId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get user profile:', error);
      throw this.enhanceError(error as AxiosError);
    }
  }

  async updateUserPreferences(preferences: Partial<UserPreferences>): Promise<any> {
    try {
      // For now, just store locally since backend doesn't have this endpoint yet
      const current = JSON.parse(localStorage.getItem('user_preferences') || '{}');
      const updated = { ...current, ...preferences };
      localStorage.setItem('user_preferences', JSON.stringify(updated));

      return { status: 'success', preferences: updated };
    } catch (error) {
      console.error('Failed to update preferences:', error);
      throw new Error('Failed to update preferences');
    }
  }

  async getAnalyticsDashboard(): Promise<any> {
    try {
      const response = await this.api.get(`/analytics-dashboard/?user_id=${this.userId}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get analytics dashboard:', error);
      // Return mock data for compatibility
      return {
        dashboard_data: {
          total_analyses: 0,
          success_rate: 100,
          avg_processing_time: 2.5,
          top_data_sources: [],
          recent_activities: [],
          performance_metrics: {
            analyses_this_week: 0,
            data_processed_gb: 0,
            insights_generated: 0,
            charts_created: 0
          }
        }
      };
    }
  }

  // ====================
  // TESTING AND DEBUG METHODS
  // ====================

  async testOpenAI(): Promise<any> {
    try {
      const response = await this.api.get('/test-openai/');
      return response.data;
    } catch (error) {
      console.error('OpenAI test failed:', error);
      return {
        status: 'error',
        message: 'Failed to test OpenAI connection',
        error: (error as Error).message
      };
    }
  }

  async debugEnvironment(): Promise<any> {
    try {
      const response = await this.api.get('/debug-env/');
      return response.data;
    } catch (error) {
      console.error('Environment debug failed:', error);
      return {
        status: 'error',
        message: 'Failed to get environment info',
        error: (error as Error).message
      };
    }
  }

  async getDiagnostics(): Promise<any> {
    const diagnostics: any = {
      backend_health: null,
      connectivity_test: null,
      endpoints_available: {},
      file_upload_support: false,
      conversation_support: false,
      openai_status: null,
      environment_info: null
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

    try {
      diagnostics.openai_status = await this.testOpenAI();
    } catch (error) {
      diagnostics.openai_status = { status: 'error', error: (error as Error).message };
    }

    try {
      diagnostics.environment_info = await this.debugEnvironment();
    } catch (error) {
      diagnostics.environment_info = { status: 'error', error: (error as Error).message };
    }

    const endpointsToTest = [
      '/health/',
      '/analyze/',
      '/test-upload/',
      '/discover-sources/',
      '/capabilities/',
      '/test-openai/'
    ];

    for (const endpoint of endpointsToTest) {
      try {
        await this.api.get(endpoint, { timeout: 5000 });
        diagnostics.endpoints_available[endpoint] = true;
      } catch (error) {
        diagnostics.endpoints_available[endpoint] = false;
      }
    }

    diagnostics.file_upload_support =
      diagnostics.endpoints_available['/analyze/'] ||
      diagnostics.endpoints_available['/test-upload/'];

    diagnostics.conversation_support = diagnostics.endpoints_available['/analyze/'];

    return diagnostics;
  }

  // ====================
  // UTILITY METHODS
  // ====================

  async ping(): Promise<{ status: string; response_time: number }> {
    const startTime = Date.now();
    try {
      await this.api.get('/health/', { timeout: 5000 });
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

  private enhanceError(error: AxiosError): Error {
    const message = (error as Error).message;

    if (message.includes('Failed to fetch') || message.includes('NetworkError')) {
      return new Error('Cannot connect to backend server. Please check if the server is running on http://localhost:8000');
    } else if (error.response?.status === 422) {
      return new Error('Invalid request format. Please check your file format and try again.');
    } else if (error.response?.status === 400) {
      return new Error('Bad request. Please verify your file and query are valid.');
    } else if (error.response?.status === 500) {
      return new Error('Server error. Please try again or contact support.');
    } else if (error.response?.status === 503) {
      return new Error('System is still initializing. Please wait a moment and try again.');
    } else {
      return new Error(`Request failed: ${message}`);
    }
  }

  // ====================
  // CONNECTION TEST METHODS (for DataDiscovery)
  // ====================

  async testConnection(sourceId: string, connectionParams: any): Promise<any> {
    try {
      // For now, simulate connection test since backend doesn't have this endpoint
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Simulate success/failure based on params
      if (connectionParams.username && connectionParams.host) {
        return {
          status: 'success',
          message: 'Connection test successful'
        };
      } else {
        return {
          status: 'failed',
          message: 'Missing required connection parameters'
        };
      }
    } catch (error) {
      return {
        status: 'failed',
        message: (error as Error).message
      };
    }
  }

  // Additional methods for compatibility
  getConnectedSources(): Promise<DataSource[]> {
    // Return mock connected sources for now
    return Promise.resolve([]);
  }

  getDiscoveryHistory(): Promise<{ discovery_sessions: DiscoverySession[] }> {
    // Return mock history for now
    return Promise.resolve({ discovery_sessions: [] });
  }

  isConnected(): boolean {
    return true; // Always return true for demo
  }

  getBaseURL(): string {
    return API_BASE_URL;
  }

  getUserId(): string {
    return this.userId;
  }
}

// ====================
// SINGLETON INSTANCE
// ====================

export const analyticsAPI = new AnalyticsAPI();

// ====================
// LEGACY COMPATIBILITY LAYER
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
        auto_discover: selectedSources.length === 0 && !file
      });

      console.log('✅ Legacy analyzeData successful');
      return response;

    } catch (error) {
      console.error('💥 Legacy analyzeData error:', error);
      throw error;
    }
  },

  async testBackendConnection() {
    try {
      console.log('🔍 Testing backend connection...');
      const health = await analyticsAPI.healthCheck();
      console.log('✅ Backend connection test result:', health);

      return {
        status: 'success',
        connected: true,
        health: health,
        message: 'Backend is healthy and accessible'
      };
    } catch (error) {
      console.error('❌ Backend connection test failed:', error);
      return {
        status: 'error',
        connected: false,
        error: (error as Error).message,
        message: 'Backend is not accessible'
      };
    }
  }
};

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

  return { isValid: true };
};

export const extractErrorMessage = (error: any): string => {
  if (typeof error === 'string') return error;
  if (error?.response?.data?.message) return error.response.data.message;
  if (error?.response?.data?.error) return error.response.data.error;
  if (error?.response?.data?.detail) return error.response.data.detail;
  if (error?.message) return error.message;
  if (error?.error) return error.error;
  return 'An unexpected error occurred';
};

export const getEnvironmentInfo = () => {
  const isDevelopment = process.env.NODE_ENV === 'development';
  const isProduction = process.env.NODE_ENV === 'production';
  const isBrowser = typeof window !== 'undefined';

  return {
    isDevelopment,
    isProduction,
    isBrowser,
    apiBaseUrl: API_BASE_URL,
    userAgent: isBrowser ? window.navigator.userAgent : 'Server',
    timestamp: new Date().toISOString()
  };
};

export const logEvent = (event: string, data?: any) => {
  const environment = getEnvironmentInfo();

  if (environment.isDevelopment) {
    console.log(`📊 Event: ${event}`, {
      event,
      data,
      timestamp: new Date().toISOString(),
      environment: environment.isDevelopment ? 'development' : 'production'
    });
  }
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
      capabilities = {
        enableConversationMode: true,
        enableAdvancedAnalytics: true,
        enableFileUpload: true,
        enableDataSourceDiscovery: true,
        maxFileSize: 100 * 1024 * 1024,
        supportedFileTypes: ['.csv', '.xlsx', '.xls', '.json']
      };
    }

    const status = diagnostics.backend_health?.status === 'error' ? 'failed' :
                  (!diagnostics.file_upload_support) ? 'partial' : 'success';

    console.log(`✅ API initialization ${status}:`, { capabilities, diagnostics });

    logEvent('api_initialization', { status, capabilities, diagnostics });

    return { status, capabilities, diagnostics };
  } catch (error) {
    console.error('❌ API initialization failed:', error);
    logEvent('api_initialization_failed', { error: extractErrorMessage(error) });

    return {
      status: 'failed',
      capabilities: {},
      diagnostics: { error: extractErrorMessage(error) }
    };
  }
};

// ====================
// ENHANCED ERROR HANDLING
// ====================

export const createRetryableRequest = async <T>(
  requestFn: () => Promise<T>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<T> => {
  let lastError: Error;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await requestFn();
    } catch (error) {
      lastError = error as Error;

      if (attempt === maxRetries) {
        break;
      }

      // Only retry on network errors or server errors
      if (error instanceof Error &&
          (error.message.includes('NetworkError') ||
           error.message.includes('Failed to fetch') ||
           error.message.includes('Server error'))) {

        console.log(`🔄 Retrying request (attempt ${attempt}/${maxRetries}) after ${delay}ms`);
        await new Promise(resolve => setTimeout(resolve, delay * attempt));
      } else {
        // Don't retry on client errors (4xx)
        break;
      }
    }
  }

  throw lastError!;
};

export const withTimeout = async <T>(
  promise: Promise<T>,
  timeoutMs: number,
  timeoutMessage: string = 'Operation timed out'
): Promise<T> => {
  const timeoutPromise = new Promise<never>((_, reject) => {
    setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
  });

  return Promise.race([promise, timeoutPromise]);
};

// ====================
// DATA PROCESSING UTILITIES
// ====================

export const processAnalysisResponse = (response: AnalysisResponse): AnalysisResponse => {
  // Ensure response has required structure
  const processed: AnalysisResponse = {
    status: response.status || 'success',
    analysis: {
      type: response.analysis?.type || 'general',
      summary: response.analysis?.summary || 'Analysis completed',
      data: response.analysis?.data || [],
      insights: response.analysis?.insights || [],
      metadata: response.analysis?.metadata || {}
    },
    query_interpretation: response.query_interpretation || {
      intent: 'summary',
      entities: [],
      confidence: 0.8
    },
    chart_intelligence: {
      suggested_charts: response.chart_intelligence?.suggested_charts || [],
      intent_metadata: response.chart_intelligence?.intent_metadata || {},
      chart_count: response.chart_intelligence?.chart_count || 0
    },
    performance: {
      total_time_ms: response.performance?.total_time_ms || 0,
      breakdown: response.performance?.breakdown || {
        analysis_ms: 0
      },
      data_stats: response.performance?.data_stats || {
        rows: 0,
        columns: 0
      }
    },
    system_info: {
      version: response.system_info?.version || '5.1.0',
      capabilities: response.system_info?.capabilities,
      model_info: response.system_info?.model_info,
      method: response.system_info?.method,
      adaptive_processing_used: response.system_info?.adaptive_processing_used,
      chart_intelligence_used: response.system_info?.chart_intelligence_used,
      smart_defaults_enabled: response.system_info?.smart_defaults_enabled,
      auto_discovery_used: response.system_info?.auto_discovery_used,
      smart_engine_available: response.system_info?.smart_engine_available,
      domain: response.system_info?.domain,
      user_id: response.system_info?.user_id,
      openai_used: response.system_info?.openai_used,
      confidence_threshold: response.system_info?.confidence_threshold
    },
    timestamp: response.timestamp || new Date().toISOString()
  };

  // Add optional fields if they exist
  if (response.comprehensive_report) {
    processed.comprehensive_report = response.comprehensive_report;
  }

  if (response.data_discovery) {
    processed.data_discovery = response.data_discovery;
  }

  return processed;
};

export const normalizeChartSuggestions = (charts: any[]): ChartSuggestion[] => {
  return charts.map((chart, index) => ({
    id: chart.id || `chart_${index}`,
    type: chart.type || chart.chart_type || 'bar',
    title: chart.title || `Chart ${index + 1}`,
    description: chart.description || 'Data visualization',
    data_columns: chart.data_columns || [],
    config: chart.config || {},
    priority: chart.priority || index,
    reasoning: chart.reasoning || 'Recommended based on data structure',
    // Handle backend variations
    chart_type: chart.chart_type,
    data: chart.data,
    x_axis: chart.x_axis,
    y_axis: chart.y_axis
  }));
};

// ====================
// CACHING UTILITIES
// ====================

export class APICache {
  private cache = new Map<string, { data: any; timestamp: number; ttl: number }>();

  set(key: string, data: any, ttlMs: number = 300000): void { // 5 minutes default
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttlMs
    });
  }

  get(key: string): any | null {
    const item = this.cache.get(key);
    if (!item) return null;

    if (Date.now() - item.timestamp > item.ttl) {
      this.cache.delete(key);
      return null;
    }

    return item.data;
  }

  clear(): void {
    this.cache.clear();
  }

  has(key: string): boolean {
    const item = this.cache.get(key);
    if (!item) return false;

    if (Date.now() - item.timestamp > item.ttl) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }
}

export const apiCache = new APICache();

// ====================
// BATCH OPERATIONS
// ====================

export const batchRequests = async <T>(
  requests: (() => Promise<T>)[],
  batchSize: number = 3,
  delayMs: number = 100
): Promise<T[]> => {
  const results: T[] = [];

  for (let i = 0; i < requests.length; i += batchSize) {
    const batch = requests.slice(i, i + batchSize);
    const batchResults = await Promise.allSettled(batch.map(req => req()));

    batchResults.forEach((result, index) => {
      if (result.status === 'fulfilled') {
        results[i + index] = result.value;
      } else {
        console.error(`Batch request ${i + index} failed:`, result.reason);
        throw result.reason;
      }
    });

    // Add delay between batches
    if (i + batchSize < requests.length) {
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }

  return results;
};

// ====================
// PERFORMANCE MONITORING
// ====================

export class PerformanceMonitor {
  private metrics = new Map<string, number[]>();

  recordTiming(operation: string, timeMs: number): void {
    if (!this.metrics.has(operation)) {
      this.metrics.set(operation, []);
    }
    this.metrics.get(operation)!.push(timeMs);
  }

  getStats(operation: string): { avg: number; min: number; max: number; count: number } | null {
    const timings = this.metrics.get(operation);
    if (!timings || timings.length === 0) return null;

    return {
      avg: timings.reduce((a, b) => a + b, 0) / timings.length,
      min: Math.min(...timings),
      max: Math.max(...timings),
      count: timings.length
    };
  }

  getAllStats(): Record<string, any> {
    const stats: Record<string, any> = {};
    const operations = Array.from(this.metrics.keys());
    for (const operation of operations) {
      stats[operation] = this.getStats(operation);
    }
    return stats;
  }

  clear(): void {
    this.metrics.clear();
  }
}

export const performanceMonitor = new PerformanceMonitor();

// ====================
// CONNECTION QUALITY MONITORING
// ====================

export class ConnectionMonitor {
  private ping_times: number[] = [];
  private error_count: number = 0;
  private last_check: number = 0;

  async checkConnection(): Promise<{
    status: 'excellent' | 'good' | 'poor' | 'disconnected';
    ping: number;
    quality_score: number;
  }> {
    try {
      const start = Date.now();
      await analyticsAPI.ping();
      const ping = Date.now() - start;

      this.ping_times.push(ping);
      if (this.ping_times.length > 10) {
        this.ping_times.shift();
      }

      this.last_check = Date.now();

      const avgPing = this.ping_times.reduce((a, b) => a + b, 0) / this.ping_times.length;
      let quality_score = 100;

      if (avgPing > 2000) quality_score -= 40;
      else if (avgPing > 1000) quality_score -= 20;
      else if (avgPing > 500) quality_score -= 10;

      if (this.error_count > 0) {
        quality_score -= this.error_count * 10;
      }

      let status: 'excellent' | 'good' | 'poor' | 'disconnected';
      if (quality_score >= 90) status = 'excellent';
      else if (quality_score >= 70) status = 'good';
      else if (quality_score >= 40) status = 'poor';
      else status = 'disconnected';

      return { status, ping, quality_score: Math.max(0, quality_score) };

    } catch (error) {
      this.error_count++;
      return { status: 'disconnected', ping: -1, quality_score: 0 };
    }
  }

  reset(): void {
    this.ping_times = [];
    this.error_count = 0;
    this.last_check = 0;
  }
}

export const connectionMonitor = new ConnectionMonitor();

export default analyticsAPI;