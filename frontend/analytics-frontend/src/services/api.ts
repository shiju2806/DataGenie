import axios, { AxiosInstance, AxiosError } from 'axios';

// ====================
// CORE TYPE DEFINITIONS
// ====================

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
const API_TIMEOUT = parseInt(process.env.REACT_APP_API_TIMEOUT || '60000');
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

// ====================
// MAIN ANALYTICS API CLASS
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

        // Add user ID to non-FormData requests
        if (config.method === 'post' || config.method === 'put') {
          if (config.data && typeof config.data === 'object' && !(config.data instanceof FormData)) {
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
        console.log(`✅ API Response: ${response.config.method?.toUpperCase()} ${response.config.url}`);
        return response;
      },
      async (error: AxiosError) => {
        console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url}`, {
          status: error.response?.status,
          statusText: error.response?.statusText,
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
    }

    return apiError;
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
    const response = await this.api.get('/status/');
    return response.data;
  }

  async getCapabilities(): Promise<any> {
    const response = await this.api.get('/capabilities/');
    return response.data;
  }

  // ====================
  // DATA DISCOVERY METHODS
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
        confidence_threshold: String(options.confidence_threshold || 0.5),
      });

      const response = await this.api.get(`/discover-sources/?${params}`);
      return response.data;
    } catch (error) {
      console.error('Data source discovery failed:', error);

      // Return mock data as fallback
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
        });
        formData.append('file', request.file, request.file.name);
      }

      // Add optional parameters
      const optionalParams = {
        data_source_id: request.data_source_id,
        domain: request.domain,
        use_adaptive: request.use_adaptive ?? true,
        include_charts: request.include_charts ?? true,
        auto_discover: request.auto_discover ?? true,
        max_rows: request.max_rows || 10000,
      };

      Object.entries(optionalParams).forEach(([key, value]) => {
        if (value !== undefined) {
          formData.append(key, String(value));
        }
      });

      // Log FormData contents for debugging
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
          // Don't set Content-Type for FormData - browser handles this
        },
        timeout: 120000, // 2 minutes for analysis
      });

      console.log('✅ Analysis completed successfully');
      return response.data;
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      throw this.enhanceError(error as AxiosError);
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
    } else {
      return new Error(`Analysis failed: ${message}`);
    }
  }

  // ====================
  // TESTING METHODS
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
      '/discover-sources/',
      '/capabilities/'
    ];

    for (const endpoint of endpointsToTest) {
      try {
        await this.api.get(endpoint);
        diagnostics.endpoints_available[endpoint] = true;
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
  // UTILITY METHODS
  // ====================

  async ping(): Promise<{ status: string; response_time: number }> {
    const startTime = Date.now();
    try {
      await this.api.get('/health/');
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

  getUserId(): string {
    return this.userId;
  }
}

// ====================
// SINGLETON INSTANCE
// ====================

export const analyticsAPI = new AnalyticsAPI();

// ====================
// SIMPLIFIED API SERVICE FOR LEGACY COMPATIBILITY
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
  },

  // Enhanced method for FormData uploads (used by ConversationalAnalytics)
  async analyzeDataWithFormData(formData: FormData) {
    try {
      console.log('📡 FormData analyzeData called...');

      // Extract data from FormData for logging
      const prompt = formData.get('prompt') as string;
      const file = formData.get('file') as File | null;

      console.log('📊 FormData contents:', {
        prompt: prompt?.substring(0, 50) + (prompt?.length > 50 ? '...' : ''),
        hasFile: !!file,
        fileName: file?.name,
        fileSize: file ? `${Math.round(file.size / 1024)} KB` : 'N/A'
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
      const response = await analyticsAPI.analyze(analysisRequest);

      console.log('✅ FormData analyzeData successful');
      return response;

    } catch (error) {
      console.error('❌ FormData analyzeData failed:', error);
      throw error;
    }
  },

  // Test backend connectivity
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

export default analyticsAPI;