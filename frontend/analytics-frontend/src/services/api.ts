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

  // ====================
  // SETUP METHODS
  // ====================

  private getOrCreateUserId(): string {
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
        // Add authentication token
        const token = this.authToken || localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }

        // Add user ID to requests
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
        console.error(`❌ API Error: ${error.config?.method?.toUpperCase()} ${error.config?.url} (${duration}ms)`, error.response?.status);

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
    localStorage.removeItem('auth_token');
    // Optionally redirect to login or emit an event
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
    localStorage.setItem('auth_token', token);
  }

  clearAuth(): void {
    this.authToken = null;
    localStorage.removeItem('auth_token');
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
      throw error;
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
      throw error;
    }
  }

  async connectDataSource(sourceId: string, connectionParams: {
    host?: string;
    port?: number;
    username?: string;
    password?: string;
    database?: string;
    connection_string?: string;
    auth_method?: string;
    ssl_enabled?: boolean;
    [key: string]: any;
  }): Promise<{ status: string; source_id: string; connection_id?: string }> {
    try {
      const response = await this.api.post('/connect-source/', {
        source_id: sourceId,
        connection_params: connectionParams,
      });
      return response.data;
    } catch (error) {
      console.error('Failed to connect data source:', error);
      throw error;
    }
  }

  async testConnection(sourceId: string, connectionParams: any): Promise<{
    status: 'success' | 'failed';
    message: string;
    details?: any;
  }> {
    try {
      const response = await this.api.post('/test-connection/', {
        source_id: sourceId,
        connection_params: connectionParams,
      });
      return response.data;
    } catch (error) {
      console.error('Connection test failed:', error);
      throw error;
    }
  }

  async getConnectedSources(): Promise<DataSource[]> {
    try {
      const response = await this.api.get('/connected-sources/');
      return response.data.sources || [];
    } catch (error) {
      console.error('Failed to get connected sources:', error);
      throw error;
    }
  }

  // ====================
  // ANALYSIS METHODS
  // ====================

  async analyze(request: AnalysisRequest): Promise<AnalysisResponse> {
    try {
      const formData = new FormData();
      formData.append('prompt', request.prompt);

      if (request.file) {
        formData.append('file', request.file);
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

      const response = await this.api.post('/analyze/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000, // Extended timeout for analysis
      });

      return response.data;
    } catch (error) {
      console.error('Analysis failed:', error);
      throw error;
    }
  }

  async quickAnalyze(prompt: string, file: File, maxRows: number = 1000): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('file', file);
      formData.append('max_rows', String(maxRows));

      const response = await this.api.post('/quick-analyze/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Quick analysis failed:', error);
      throw error;
    }
  }

  async getChartSuggestions(prompt: string, file: File, maxRows: number = 1000): Promise<{
    status: string;
    suggestions: ChartSuggestion[];
    metadata: any;
  }> {
    try {
      const formData = new FormData();
      formData.append('prompt', prompt);
      formData.append('file', file);
      formData.append('max_rows', String(maxRows));

      const response = await this.api.post('/chart-suggestions/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Chart suggestions failed:', error);
      throw error;
    }
  }

  async generateChart(chartId: string, data: any[], config: any): Promise<{
    status: string;
    chart_url?: string;
    chart_data: any;
    format: string;
  }> {
    try {
      const response = await this.api.post('/generate-chart/', {
        chart_id: chartId,
        data,
        config,
      });
      return response.data;
    } catch (error) {
      console.error('Chart generation failed:', error);
      throw error;
    }
  }

  // ====================
  // RECOMMENDATIONS & PERSONALIZATION
  // ====================

  async getRecommendations(mode: 'conservative' | 'balanced' | 'aggressive' = 'balanced'): Promise<RecommendationResponse> {
    try {
      const response = await this.api.get(`/recommendations/?mode=${mode}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get recommendations:', error);
      throw error;
    }
  }

  async recordFeedback(type: 'recommendation' | 'analysis' | 'chart', targetId: string, action: 'like' | 'dislike' | 'ignore' | 'implement', context?: any): Promise<{ status: string }> {
    try {
      const response = await this.api.post('/feedback/', {
        type,
        target_id: targetId,
        action,
        context: context || {},
        timestamp: new Date().toISOString(),
      });
      return response.data;
    } catch (error) {
      console.error('Failed to record feedback:', error);
      throw error;
    }
  }

  async getUserProfile(): Promise<UserProfile> {
    try {
      const response = await this.api.get('/user-profile/');
      return response.data;
    } catch (error) {
      console.error('Failed to get user profile:', error);
      throw error;
    }
  }

  async updateUserPreferences(preferences: Partial<UserProfile['preferences']>): Promise<{ status: string }> {
    try {
      const response = await this.api.put('/user-profile/preferences/', preferences);
      return response.data;
    } catch (error) {
      console.error('Failed to update user preferences:', error);
      throw error;
    }
  }

  // ====================
  // DASHBOARD & ANALYTICS
  // ====================

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

  async getUsageStatistics(timeframe: '24h' | '7d' | '30d' | '90d' = '7d'): Promise<{
    timeframe: string;
    total_analyses: number;
    unique_data_sources: number;
    processing_time_avg: number;
    success_rate: number;
    daily_breakdown: Array<{
      date: string;
      analyses: number;
      processing_time: number;
    }>;
  }> {
    try {
      const response = await this.api.get(`/usage-statistics/?timeframe=${timeframe}`);
      return response.data;
    } catch (error) {
      console.error('Failed to get usage statistics:', error);
      throw error;
    }
  }

  // ====================
  // REPORTS & EXPORTS
  // ====================

  async generateReport(type: 'executive' | 'operational' | 'analytical', config: {
    data_sources?: string[];
    date_range?: { start: string; end: string };
    include_charts?: boolean;
    format?: 'pdf' | 'html' | 'json';
    sections?: string[];
  }): Promise<{
    status: string;
    report_id: string;
    estimated_completion: string;
  }> {
    try {
      const response = await this.api.post('/generate-report/', {
        type,
        config,
      });
      return response.data;
    } catch (error) {
      console.error('Report generation failed:', error);
      throw error;
    }
  }

  async getReportStatus(reportId: string): Promise<{
    status: 'pending' | 'processing' | 'completed' | 'failed';
    progress: number;
    estimated_completion?: string;
    download_url?: string;
    error_message?: string;
  }> {
    try {
      const response = await this.api.get(`/report-status/${reportId}/`);
      return response.data;
    } catch (error) {
      console.error('Failed to get report status:', error);
      throw error;
    }
  }

  async downloadReport(reportId: string, format: 'pdf' | 'html' | 'json' = 'pdf'): Promise<Blob> {
    try {
      const response = await this.api.get(`/download-report/${reportId}/?format=${format}`, {
        responseType: 'blob',
      });
      return response.data;
    } catch (error) {
      console.error('Report download failed:', error);
      throw error;
    }
  }

  // ====================
  // DATA MANAGEMENT
  // ====================

  async uploadFile(file: File, metadata?: { description?: string; tags?: string[] }): Promise<{
    status: string;
    file_id: string;
    file_info: {
      name: string;
      size: number;
      type: string;
      rows?: number;
      columns?: string[];
    };
  }> {
    try {
      const formData = new FormData();
      formData.append('file', file);

      if (metadata) {
        formData.append('metadata', JSON.stringify(metadata));
      }

      const response = await this.api.post('/upload-file/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('File upload failed:', error);
      throw error;
    }
  }

  async getUploadedFiles(): Promise<Array<{
    id: string;
    name: string;
    size: number;
    type: string;
    uploaded_at: string;
    status: string;
  }>> {
    try {
      const response = await this.api.get('/uploaded-files/');
      return response.data.files || [];
    } catch (error) {
      console.error('Failed to get uploaded files:', error);
      throw error;
    }
  }

  async deleteFile(fileId: string): Promise<{ status: string }> {
    try {
      const response = await this.api.delete(`/files/${fileId}/`);
      return response.data;
    } catch (error) {
      console.error('File deletion failed:', error);
      throw error;
    }
  }

  // ====================
  // SMART DEFAULTS & CONFIGURATION
  // ====================

  async getSmartDefaults(): Promise<{
    analysis_defaults: any;
    chart_preferences: any;
    data_source_configs: any;
    user_patterns: any;
  }> {
    try {
      const response = await this.api.get('/smart-defaults/');
      return response.data;
    } catch (error) {
      console.error('Failed to get smart defaults:', error);
      throw error;
    }
  }

  async updateSmartDefaults(config: any): Promise<{ status: string }> {
    try {
      const response = await this.api.put('/smart-defaults/', config);
      return response.data;
    } catch (error) {
      console.error('Failed to update smart defaults:', error);
      throw error;
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
    return this.authToken !== null || localStorage.getItem('auth_token') !== null;
  }
}

// ====================
// SINGLETON INSTANCE
// ====================

export const analyticsAPI = new AnalyticsAPI();
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
  const fileExtension = file.name.split('.').pop()?.toLowerCase();
  return allowedTypes.includes(fileExtension || '');
};

export const createDownloadLink = (blob: Blob, filename: string): void => {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  window.URL.revokeObjectURL(url);
};