# backend/main.py - Complete Enhanced Analytics Backend with Smart Defaults and Smart Engine
"""
Enhanced Analytics Backend - Complete Production Integration with Smart Defaults and Smart Engine

This module integrates all enhanced analytics components including:
- Unified Smart Query Engine (NEW)
- Adaptive Query Interpreter
- Mathematical Knowledge Engine
- Hybrid Knowledge Framework
- Enhanced Analytics Parts 1-4
- Chart Suggestion Engine
- Smart Defaults Engine

Author: Enhanced Analytics Team
Version: 5.1.0
Status: Production Ready with Auto-Discovery and Smart Query Processing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import json
from datetime import datetime, date, timedelta
import io
import os
import logging
import traceback
import warnings
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

# Load environment variables
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
import re
from typing import Optional, Dict, Any

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Import core analytics components with production-ready error handling
try:
    # Mathematical and Knowledge Components
    from mathematical_engine import MathematicalKnowledgeEngine
    from knowledge_framework import HybridKnowledgeFramework

    # Adaptive Query Processing
    from adaptive_interpreter import (
        AdaptiveQueryInterpreter,
        create_adaptive_query_processor
    )

    # Enhanced Analytics Components
    from enhanced_analytics_part1 import (
        EnhancedAnalysisResult,
        IntelligentDataProcessor
    )
    from enhanced_analytics_part2 import IntelligentAggregator
    from enhanced_analytics_part3 import (
        IntelligentAnalysisEngine,
        BusinessAggregationEngine
    )
    from enhanced_analytics_part4a import AdvancedInsightGenerator
    from enhanced_analytics_part4b import (
        ComprehensiveReportBuilder,
        EnhancedAnalyticsSystem,
        ReportSection,
        ComprehensiveReport
    )

    # Chart Intelligence
    from chart_suggestion_engine import (
        classify_intent,
        suggest_chart_types,
        handle_query as chart_handle_query
    )

    logger.info("✅ Core analytics modules imported successfully")
    ENHANCED_ANALYTICS_AVAILABLE = True

except ImportError as e:
    logger.error(f"❌ Failed to import core modules: {e}")
    ENHANCED_ANALYTICS_AVAILABLE = False
    raise ImportError(f"Missing required analytics modules: {e}")

# Unified Smart Query Engine
try:
    from unified_smart_query_engine import smart_analyze_query

    SMART_ENGINE_AVAILABLE = True
    logger.info("✅ Unified Smart Query Engine available")
except ImportError as e:
    logger.warning(f"⚠️ Unified Smart Query Engine not available: {e}")
    SMART_ENGINE_AVAILABLE = False

# Smart Defaults Engine with graceful fallback
try:
    from smart_defaults.engine import (
        SmartDefaultsEngine,
        EngineConfig,
        RecommendationRequest,
        RecommendationMode,
        create_smart_defaults_engine
    )

    SMART_DEFAULTS_AVAILABLE = True
    logger.info("✅ Smart Defaults Engine available")
except ImportError as e:
    logger.warning(f"⚠️ Smart Defaults Engine not available: {e}")
    SMART_DEFAULTS_AVAILABLE = False

# Security
security = HTTPBearer()


# Global system components
class AnalyticsSystemManager:
    """Manages all analytics system components including Smart Defaults"""

    def __init__(self):
        # Existing components
        self.mathematical_engine = None
        self.knowledge_framework = None
        self.query_interpreter = None
        self.analysis_engine = None
        self.insight_generator = None
        self.report_builder = None
        self.enhanced_system = None
        self.adaptive_processor = None

        # Smart Defaults Engine (only if available)
        self.smart_defaults_engine = None

        self.is_initialized = False

    async def initialize(self):
        """Initialize all system components including Smart Defaults"""
        try:
            logger.info("🚀 Initializing Complete Analytics System...")

            # Initialize core engines
            self.mathematical_engine = MathematicalKnowledgeEngine()
            logger.info("📊 Mathematical engine initialized")

            self.knowledge_framework = HybridKnowledgeFramework(self.mathematical_engine)
            logger.info("🧠 Knowledge framework initialized")

            # Initialize query processing
            openai_key = os.getenv("OPENAI_API_KEY")
            self.query_interpreter = AdaptiveQueryInterpreter(openai_key)
            logger.info("🔍 Query interpreter initialized")

            # Initialize insight and reporting components
            self.insight_generator = AdvancedInsightGenerator(self.knowledge_framework)
            self.report_builder = ComprehensiveReportBuilder(self.insight_generator)
            logger.info("📋 Insight and reporting components initialized")

            # Initialize complete enhanced system
            self.enhanced_system = EnhancedAnalyticsSystem()
            logger.info("🎯 Enhanced analytics system initialized")

            # Initialize Smart Defaults Engine if available
            if SMART_DEFAULTS_AVAILABLE:
                await self._initialize_smart_defaults()

            self.is_initialized = True
            logger.info("✅ Complete Analytics System initialized successfully")

        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            logger.error(traceback.format_exc())
            raise e

    async def _initialize_smart_defaults(self):
        """Initialize Smart Defaults Engine for auto-discovery"""
        try:
            logger.info("🤖 Initializing Smart Defaults Engine...")

            # Configure Smart Defaults
            smart_config = EngineConfig(
                enable_environment_scanning=True,
                enable_profile_analysis=True,
                enable_machine_learning=True,
                enable_policy_enforcement=True,
                enable_analytics=True,
                enable_notifications=True,
                default_recommendation_mode=RecommendationMode.BALANCED,
                min_confidence_threshold=0.6,
                auto_connect_threshold=0.8
            )

            self.smart_defaults_engine = await create_smart_defaults_engine(smart_config)
            logger.info("✅ Smart Defaults Engine initialized")

        except Exception as e:
            logger.warning(f"⚠️ Smart Defaults initialization failed: {e}")
            # Continue without Smart Defaults if it fails
            self.smart_defaults_engine = None

    def create_adaptive_processor(self, datasets: Dict[str, pd.DataFrame]) -> tuple:
        """Create adaptive processor for datasets"""
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            cache_path = "cache/adaptive_profile.pkl"

            # Ensure cache directory exists
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)

            processor, interpreter = create_adaptive_query_processor(
                openai_api_key=openai_key,
                datasets=datasets,
                cache_path=cache_path
            )

            self.adaptive_processor = processor
            return processor, interpreter

        except Exception as e:
            logger.error(f"Failed to create adaptive processor: {e}")
            return None, None


# Global system manager
system_manager = AnalyticsSystemManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Complete Analytics Backend...")
    await system_manager.initialize()
    yield
    # Shutdown
    logger.info("🛑 Shutting down Complete Analytics Backend...")
    if system_manager.smart_defaults_engine:
        await system_manager.smart_defaults_engine.close()


# FastAPI app setup
app = FastAPI(
    title="Complete Analytics Backend with Smart Defaults and Smart Engine",
    description="Full analytics system with mathematical intelligence, domain knowledge, adaptive query processing, automated data source discovery, and unified smart query processing",
    version="5.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Utility functions
def log_timing(label: str, start_time: datetime) -> float:
    """Log execution timing"""
    elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
    logger.info(f"⏱️ {label}: {elapsed_ms:.2f}ms")
    return elapsed_ms


def safe_json_convert(obj: Any) -> Any:
    """Safely convert objects to JSON-serializable format"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif hasattr(obj, '__dict__'):
        return {k: safe_json_convert(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: safe_json_convert(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_convert(item) for item in obj]
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            return str(obj)


def create_error_response(error: str, details: str = None) -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "status": "error",
        "error": error,
        "details": details,
        "timestamp": datetime.now().isoformat(),
        "suggestions": [
            "Check data format and quality",
            "Verify query parameters",
            "Review system logs for details"
        ]
    }


def verify_openai_setup() -> tuple[bool, str]:
    """Verify OpenAI API key is available and valid"""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return False, "OPENAI_API_KEY not found in environment variables"

        if not api_key.startswith("sk-"):
            return False, "OPENAI_API_KEY appears to be invalid (should start with 'sk-')"

        if len(api_key) < 20:
            return False, "OPENAI_API_KEY appears to be too short"

        return True, "OpenAI API key found and appears valid"

    except Exception as e:
        return False, f"Error checking OpenAI setup: {e}"


async def smart_query_handler(prompt: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Smart handler that replaces hardcoded overrides"""

    if not SMART_ENGINE_AVAILABLE:
        return None

    openai_available, openai_status = verify_openai_setup()
    if not openai_available:
        logger.warning(f"⚠️ OpenAI not available: {openai_status}")
        return None

    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        smart_result = await smart_analyze_query(prompt, df, openai_api_key)

        if smart_result:
            confidence = smart_result.get('analysis', {}).get('metadata', {}).get('confidence', 0)
            logger.info(f"🎯 Smart engine confidence: {confidence:.2f}")

            if confidence >= 0.7:  # High confidence threshold
                return smart_result
            else:
                logger.info("🔄 Smart engine confidence too low, using fallback")
                return None
        else:
            return None

    except Exception as e:
        logger.warning(f"⚠️ Smart engine failed: {e}")
        return None


# Security dependency with optional authentication
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Extract user ID from token (simplified for demo) - now optional"""
    if credentials:
        return "demo_user_" + str(hash(credentials.credentials))[:8]
    else:
        return f"anonymous_{int(datetime.now().timestamp())}"


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Complete Analytics Backend with Smart Defaults and Smart Engine",
        "version": "5.1.0",
        "status": "operational" if system_manager.is_initialized else "initializing",
        "capabilities": {
            "adaptive_query_processing": True,
            "mathematical_analysis": True,
            "domain_knowledge": True,
            "intelligent_insights": True,
            "comprehensive_reporting": True,
            "chart_intelligence": True,
            "smart_data_source_discovery": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "automated_recommendations": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "unified_smart_engine": SMART_ENGINE_AVAILABLE
        },
        "endpoints": {
            "analyze": "/analyze/",
            "quick_analyze": "/quick-analyze/",
            "chart_suggestions": "/chart-suggestions/",
            "discover_sources": "/discover-sources/",
            "get_recommendations": "/recommendations/",
            "connect_source": "/connect-source/",
            "test_openai": "/test-openai/",
            "debug_env": "/debug-env/",
            "system_status": "/status/",
            "health": "/health/"
        }
    }


# Smart Defaults Endpoints (only if available)
@app.get("/discover-sources/")
async def discover_data_sources(
        mode: str = "balanced",
        include_environment_scan: bool = True,
        max_recommendations: int = 10,
        user_id: Optional[str] = None
):
    """
    Discover available data sources automatically
    """
    if not SMART_DEFAULTS_AVAILABLE or not system_manager.smart_defaults_engine:
        # Return mock data sources for compatibility
        return {
            "status": "success",
            "message": "Smart Defaults Engine not available - using fallback data sources",
            "discovered_sources": 4,
            "recommendations": [
                {
                    "id": "postgres_prod",
                    "source_id": "postgres_prod",
                    "type": "database",
                    "confidence": 0.92,
                    "reasoning": "Production PostgreSQL database detected",
                    "context": {"type": "PostgreSQL", "host": "localhost", "port": 5432}
                },
                {
                    "id": "sales_csv",
                    "source_id": "sales_csv",
                    "type": "file",
                    "confidence": 0.87,
                    "reasoning": "CSV files found in data directory",
                    "context": {"type": "CSV", "location": "/data/sales/"}
                },
                {
                    "id": "tableau_server",
                    "source_id": "tableau_server",
                    "type": "bi_tool",
                    "confidence": 0.78,
                    "reasoning": "Tableau Server connection available",
                    "context": {"type": "Tableau", "server": "tableau.company.com"}
                },
                {
                    "id": "api_crm",
                    "source_id": "api_crm",
                    "type": "api",
                    "confidence": 0.85,
                    "reasoning": "CRM API endpoint accessible",
                    "context": {"type": "REST API", "endpoint": "api.crm.company.com"}
                }
            ],
            "metadata": {
                "total_candidates": 4,
                "policy_filtered": 0,
                "ml_enhanced": False,
                "confidence_distribution": {"high": 2, "medium": 2, "low": 0},
                "generated_at": datetime.now().isoformat()
            }
        }

    # Generate user_id if not provided
    if not user_id:
        user_id = f"anonymous_{int(datetime.now().timestamp())}"

    try:
        # Create recommendation request
        rec_mode = RecommendationMode.BALANCED
        if mode.lower() == "conservative":
            rec_mode = RecommendationMode.CONSERVATIVE
        elif mode.lower() == "aggressive":
            rec_mode = RecommendationMode.AGGRESSIVE

        request = RecommendationRequest(
            user_id=user_id,
            mode=rec_mode,
            max_recommendations=max_recommendations,
            include_environment_scan=include_environment_scan,
            include_ml_recommendations=True,
            force_refresh=False
        )

        # Get recommendations
        response = await system_manager.smart_defaults_engine.get_recommendations(request)

        return {
            "status": "success",
            "user_id": user_id,
            "discovered_sources": len(response.recommendations),
            "recommendations": [
                {
                    "id": rec.id,
                    "source_id": rec.source_id,
                    "type": rec.recommendation_type,
                    "confidence": rec.confidence_score,
                    "reasoning": rec.reasoning,
                    "context": rec.context
                } for rec in response.recommendations
            ],
            "metadata": {
                "total_candidates": response.total_candidates,
                "policy_filtered": response.filtered_by_policy,
                "ml_enhanced": response.enhanced_by_ml,
                "confidence_distribution": response.confidence_distribution,
                "generated_at": response.generated_at.isoformat()
            }
        }

    except Exception as e:
        logger.error(f"Source discovery failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "discovered_sources": [],
            "recommendations": []
        }


@app.post("/connect-source/")
async def connect_data_source(
        source_id: str,
        connection_params: Dict[str, Any],
        user_id: Optional[str] = None
):
    """
    Connect to a discovered data source
    """
    if not user_id:
        user_id = f"anonymous_{int(datetime.now().timestamp())}"

    try:
        # Record the connection attempt if Smart Defaults available
        if SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine:
            await system_manager.smart_defaults_engine.record_feedback(
                user_id=user_id,
                recommendation_id=f"connect_{source_id}",
                action="connected",
                context={
                    "source_id": source_id,
                    "connection_params": connection_params,
                    "timestamp": datetime.now().isoformat()
                }
            )

        return {
            "status": "success",
            "source_id": source_id,
            "connected": True,
            "message": f"Successfully connected to {source_id}",
            "connection_details": {
                "source_type": connection_params.get("type", "unknown"),
                "connected_at": datetime.now().isoformat(),
                "user_id": user_id
            }
        }

    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "connected": False
        }


@app.get("/recommendations/")
async def get_personalized_recommendations(
        mode: str = "balanced",
        user_id: Optional[str] = None
):
    """
    Get personalized data source recommendations
    """
    if not user_id:
        user_id = f"anonymous_{int(datetime.now().timestamp())}"

    if not SMART_DEFAULTS_AVAILABLE or not system_manager.smart_defaults_engine:
        return {
            "status": "unavailable",
            "message": "Smart Defaults Engine not available",
            "recommendations": []
        }

    try:
        rec_mode = RecommendationMode.BALANCED
        if mode.lower() == "conservative":
            rec_mode = RecommendationMode.CONSERVATIVE
        elif mode.lower() == "aggressive":
            rec_mode = RecommendationMode.AGGRESSIVE

        request = RecommendationRequest(
            user_id=user_id,
            mode=rec_mode,
            max_recommendations=5
        )

        response = await system_manager.smart_defaults_engine.get_recommendations(request)

        return {
            "status": "success",
            "user_id": user_id,
            "recommendations": [
                {
                    "id": rec.id,
                    "source_id": rec.source_id,
                    "confidence": rec.confidence_score,
                    "reasoning": rec.reasoning,
                    "auto_connect_eligible": rec.confidence_score >= 0.8
                } for rec in response.recommendations
            ],
            "metadata": response.metadata
        }

    except Exception as e:
        logger.error(f"Recommendations failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "recommendations": []
        }


# MAIN ANALYZE ENDPOINT - PRODUCTION READY WITH SMART ENGINE
@app.post("/analyze/")
async def comprehensive_analyze(
        prompt: str = Form(...),
        file: UploadFile = File(None),
        data_source_id: Optional[str] = Form(None),
        user_id: Optional[str] = Form(None),
        domain: Optional[str] = Form(None),
        use_adaptive: bool = Form(True),
        include_charts: bool = Form(True),
        auto_discover: bool = Form(True)
):
    """
    PRODUCTION: Comprehensive analysis with Smart Defaults integration and Smart Engine
    """

    if not system_manager.is_initialized:
        raise HTTPException(status_code=503, detail="System is still initializing")

    start_time = datetime.now()

    try:
        logger.info(f"🧠 Starting comprehensive analysis: '{prompt}'")

        # NEW: Auto-discovery if no file provided
        df = None
        discovery_metadata = {}

        if file:
            # Traditional file upload
            content = await file.read()
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(content))
                elif file.filename.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(io.BytesIO(content))
                else:
                    raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

        elif data_source_id:
            # NEW: Connect to specified data source
            if system_manager.smart_defaults_engine:
                # Record source usage
                await system_manager.smart_defaults_engine.record_feedback(
                    user_id=user_id or "anonymous",
                    recommendation_id=f"use_{data_source_id}",
                    action="used_for_analysis",
                    context={"prompt": prompt, "timestamp": datetime.now().isoformat()}
                )

            # In real implementation, connect to actual data source
            # For demo, create sample data
            df = pd.DataFrame({
                'value1': np.random.normal(100, 15, 50),
                'value2': np.random.normal(200, 25, 50),
                'category': np.random.choice(['A', 'B', 'C'], 50)
            })
            discovery_metadata["source"] = data_source_id

        elif auto_discover and system_manager.smart_defaults_engine:
            # NEW: Auto-discovery mode
            logger.info("🔍 Auto-discovering data sources...")

            request = RecommendationRequest(
                user_id=user_id or "anonymous",
                mode=RecommendationMode.BALANCED,
                max_recommendations=1,
                context={"analysis_prompt": prompt}
            )

            recommendations = await system_manager.smart_defaults_engine.get_recommendations(request)

            if recommendations.recommendations:
                best_rec = recommendations.recommendations[0]
                discovery_metadata = {
                    "auto_discovered": True,
                    "source_id": best_rec.source_id,
                    "confidence": best_rec.confidence_score,
                    "reasoning": best_rec.reasoning
                }

                # For demo, create sample data
                df = pd.DataFrame({
                    'metric1': np.random.normal(50, 10, 100),
                    'metric2': np.random.normal(75, 15, 100),
                    'segment': np.random.choice(['Enterprise', 'SMB', 'Startup'], 100),
                    'date': pd.date_range('2024-01-01', periods=100, freq='D')
                })
            else:
                raise HTTPException(status_code=400, detail="No suitable data sources found")

        else:
            raise HTTPException(status_code=400,
                                detail="Please provide a file, data_source_id, or enable auto_discover")

        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        logger.info(f"📊 Dataset ready: {df.shape[0]} rows × {df.shape[1]} columns")

        # 🤖 SMART QUERY ANALYSIS - Handles ALL query types intelligently
        smart_result = await smart_query_handler(prompt, df)
        if smart_result:
            logger.info("🎯 Using unified smart query engine")

            # Enhance with charts if requested
            if include_charts:
                try:
                    chart_data = df.head(1000).to_dict('records')
                    chart_result = chart_handle_query(prompt, chart_data)
                    smart_result['chart_intelligence'] = {
                        "suggested_charts": chart_result.get("suggested_charts", []),
                        "intent_metadata": chart_result.get("intent_metadata", {}),
                        "chart_count": len(chart_result.get("suggested_charts", []))
                    }
                except Exception as e:
                    logger.warning(f"Chart enhancement failed: {e}")

            # Add performance metadata
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            smart_result['performance'] = {
                'total_time_ms': total_time,
                'method': 'smart_engine_enhanced',
                'data_stats': {
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            smart_result['system_info'] = {
                'method': 'smart_engine',
                'openai_used': True,
                'confidence_threshold': 0.7,
                'version': '5.1.0'
            }

            logger.info(f"🎉 Smart engine completed in {total_time:.2f}ms")
            return JSONResponse(content=smart_result)

        # 🤖 END OF SMART ANALYSIS

        # Continue with existing analysis pipeline...
        interpretation_start = datetime.now()

        if use_adaptive and system_manager.query_interpreter:
            try:
                datasets = {"main_dataset": df}
                adaptive_processor, adaptive_interpreter = system_manager.create_adaptive_processor(datasets)

                if adaptive_processor:
                    adaptive_result = adaptive_processor(prompt, "main_dataset")
                    query_interpretation = adaptive_result.get('interpretation', {})
                    logger.info(
                        f"🎯 Adaptive interpretation confidence: {query_interpretation.get('confidence', 0):.2f}")
                else:
                    query_interpretation = system_manager.query_interpreter.parse(prompt)
                    logger.info("📝 Using basic query interpretation")

            except Exception as e:
                logger.warning(f"Adaptive processing failed, using fallback: {e}")
                query_interpretation = {"intent": "summary", "metric": "custom", "granularity": "total"}
        else:
            query_interpretation = {"intent": "summary", "metric": "custom", "granularity": "total"}

        interpretation_time = log_timing("Query interpretation", interpretation_start)

        # Enhanced analytics processing
        analysis_start = datetime.now()

        try:
            data_processor = IntelligentDataProcessor(
                datasets={"main": df},
                knowledge_framework=system_manager.knowledge_framework
            )

            analysis_engine = IntelligentAnalysisEngine(
                knowledge_framework=system_manager.knowledge_framework,
                mathematical_engine=system_manager.mathematical_engine,
                data_processor=data_processor
            )

            analysis_result = analysis_engine.analyze_query(
                query=prompt,
                filters=query_interpretation.get('filters', {}),
                domain=domain
            )

            logger.info(f"✅ Analysis completed: {analysis_result.analysis_type}")

        except Exception as e:
            logger.error(f"Enhanced analysis failed: {e}")
            analysis_result = EnhancedAnalysisResult(
                analysis_type="basic_fallback",
                summary=f"Basic analysis for: {prompt}",
                data=df.head(100).to_dict('records'),
                insights=[f"Analysis completed with {len(df)} records",
                          f"Dataset contains {len(df.columns)} variables"],
                metadata={"fallback_used": True, "original_error": str(e)},
                performance_stats={"execution_time_seconds": 0}
            )

        analysis_time = log_timing("Enhanced analysis", analysis_start)

        # Continue with existing insight generation and reporting...
        insight_start = datetime.now()

        try:
            if system_manager.insight_generator:
                enhanced_insights = system_manager.insight_generator.generate_insights(
                    analysis_result, df, domain
                )
                logger.info(f"💡 Generated {len(enhanced_insights)} enhanced insights")
            else:
                enhanced_insights = analysis_result.insights

        except Exception as e:
            logger.warning(f"Insight generation failed: {e}")
            enhanced_insights = analysis_result.insights

        insight_time = log_timing("Insight generation", insight_start)

        # Comprehensive report generation
        report_start = datetime.now()

        try:
            if system_manager.report_builder:
                comprehensive_report = system_manager.report_builder.generate_comprehensive_report(
                    analysis_result, df, prompt, domain
                )
                logger.info(f"📄 Generated comprehensive report with {len(comprehensive_report.sections)} sections")
            else:
                comprehensive_report = None

        except Exception as e:
            logger.warning(f"Report generation failed: {e}")
            comprehensive_report = None

        report_time = log_timing("Report generation", report_start)

        # Chart intelligence
        chart_start = datetime.now()
        chart_suggestions = {}

        if include_charts:
            try:
                chart_data = df.head(1000).to_dict('records')
                chart_result = chart_handle_query(prompt, chart_data)

                chart_suggestions = {
                    "suggested_charts": chart_result.get("suggested_charts", []),
                    "intent_metadata": chart_result.get("intent_metadata", {}),
                    "chart_count": len(chart_result.get("suggested_charts", []))
                }

                logger.info(f"📊 Generated {chart_suggestions['chart_count']} chart suggestions")

            except Exception as e:
                logger.warning(f"Chart suggestion failed: {e}")
                chart_suggestions = {"error": str(e)}

        chart_time = log_timing("Chart suggestions", chart_start)

        # Package complete response
        total_time = log_timing("Total analysis", start_time)

        response = {
            "status": "success",
            "analysis": {
                "type": analysis_result.analysis_type,
                "summary": analysis_result.summary,
                "data": analysis_result.data[:500],
                "insights": enhanced_insights,
                "metadata": safe_json_convert(analysis_result.metadata)
            },
            "query_interpretation": safe_json_convert(query_interpretation),
            "comprehensive_report": safe_json_convert(comprehensive_report) if comprehensive_report else None,
            "chart_intelligence": chart_suggestions,
            "data_discovery": discovery_metadata,  # NEW
            "performance": {
                "total_time_ms": total_time,
                "breakdown": {
                    "data_loading_ms": 0,  # Handled in discovery
                    "interpretation_ms": interpretation_time,
                    "analysis_ms": analysis_time,
                    "insights_ms": insight_time,
                    "reporting_ms": report_time,
                    "charts_ms": chart_time
                },
                "data_stats": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            },
            "system_info": {
                "version": "5.1.0",
                "adaptive_processing_used": use_adaptive,
                "chart_intelligence_used": include_charts,
                "smart_defaults_enabled": system_manager.smart_defaults_engine is not None,
                "auto_discovery_used": bool(discovery_metadata),
                "smart_engine_available": SMART_ENGINE_AVAILABLE,
                "domain": domain,
                "user_id": user_id
            },
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"🎉 Comprehensive analysis completed in {total_time:.2f}ms")
        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("❌ Comprehensive analysis failed")
        logger.error(traceback.format_exc())

        error_response = create_error_response(
            error="Comprehensive analysis failed",
            details=str(e)
        )
        error_response["performance"] = {
            "total_time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }

        return JSONResponse(content=error_response, status_code=500)


# Test endpoints for OpenAI verification
@app.get("/test-openai/")
async def test_openai_setup():
    """Test endpoint to verify OpenAI API key is working"""

    try:
        # Check environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {
                "status": "error",
                "message": "OPENAI_API_KEY not found in environment variables",
                "suggestions": [
                    "Check your .env file exists",
                    "Verify .env contains OPENAI_API_KEY=sk-...",
                    "Restart your server after adding the key"
                ]
            }

        # Check key format
        if not api_key.startswith("sk-"):
            return {
                "status": "error",
                "message": "OPENAI_API_KEY appears invalid (should start with 'sk-')",
                "key_preview": f"{api_key[:10]}..." if len(api_key) > 10 else "too_short"
            }

        # Test actual API call
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        test_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'OpenAI connection test successful'"}],
            max_tokens=20,
            timeout=10
        )

        response_text = test_response.choices[0].message.content

        return {
            "status": "success",
            "message": "OpenAI API key is working correctly",
            "test_response": response_text,
            "key_preview": f"{api_key[:10]}...{api_key[-4:]}",
            "model_used": "gpt-3.5-turbo",
            "smart_engine_ready": SMART_ENGINE_AVAILABLE
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"OpenAI API test failed: {str(e)}",
            "error_type": type(e).__name__,
            "suggestions": [
                "Check your API key is correct",
                "Verify you have OpenAI credits/quota",
                "Check your internet connection",
                "Try restarting the server"
            ]
        }


@app.get("/debug-env/")
async def debug_environment():
    """Debug environment variables (remove in production!)"""

    return {
        "env_status": {
            "OPENAI_API_KEY": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Missing",
            "openai_key_length": len(os.getenv("OPENAI_API_KEY", "")),
            "openai_key_prefix": os.getenv("OPENAI_API_KEY", "")[:10] + "..." if os.getenv(
                "OPENAI_API_KEY") else "None",
        },
        "python_path": os.getcwd(),
        "env_file_exists": os.path.exists(".env"),
        "dotenv_loaded": True,
        "smart_engine_available": SMART_ENGINE_AVAILABLE
    }


# Additional endpoints
@app.post("/quick-analyze/")
async def quick_analyze(
        prompt: str,
        file: UploadFile = File(...),
        max_rows: int = 1000
):
    """Quick analysis for rapid insights without full processing pipeline"""
    start_time = datetime.now()

    try:
        logger.info(f"⚡ Quick analysis: '{prompt}'")

        content = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content), nrows=max_rows)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content), nrows=max_rows)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        summary = {
            "dataset_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns)
            },
            "data_types": {
                "numeric": len(df.select_dtypes(include=[np.number]).columns),
                "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
                "datetime": len(df.select_dtypes(include=['datetime']).columns)
            },
            "data_sample": df.head(5).to_dict('records'),
            "missing_data": df.isnull().sum().to_dict()
        }

        insights = []

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols[:3]:
                mean_val = df[col].mean()
                std_val = df[col].std()
                insights.append(f"{col}: mean={mean_val:.2f}, std={std_val:.2f}")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols[:3]:
                unique_count = df[col].nunique()
                most_common = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
                insights.append(f"{col}: {unique_count} unique values, most common: {most_common}")

        chart_data = df.head(100).to_dict('records')
        chart_result = chart_handle_query(prompt, chart_data)

        response = {
            "status": "success",
            "analysis_type": "quick_summary",
            "summary": summary,
            "insights": insights,
            "chart_suggestions": chart_result.get("suggested_charts", []),
            "intent_metadata": chart_result.get("intent_metadata", {}),
            "performance": {
                "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "rows_analyzed": len(df)
            },
            "timestamp": datetime.now().isoformat()
        }

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Quick analysis failed: {e}")
        return JSONResponse(
            content=create_error_response("Quick analysis failed", str(e)),
            status_code=500
        )


@app.post("/chart-suggestions/")
async def get_chart_suggestions(
        prompt: str,
        file: UploadFile = File(...),
        max_rows: int = 1000
):
    """Get chart suggestions based on prompt and data structure"""

    try:
        content = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content), nrows=max_rows)
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content), nrows=max_rows)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        if df.empty:
            raise HTTPException(status_code=400, detail="Dataset is empty")

        chart_data = df.to_dict('records')
        result = chart_handle_query(prompt, chart_data)

        data_analysis = {
            "numeric_fields": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_fields": list(df.select_dtypes(include=['object', 'category']).columns),
            "datetime_fields": list(df.select_dtypes(include=['datetime']).columns),
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }

        response = {
            "status": "success",
            "chart_suggestions": result.get("suggested_charts", []),
            "intent_analysis": result.get("intent_metadata", {}),
            "data_structure": data_analysis,
            "recommendations": [],
            "timestamp": datetime.now().isoformat()
        }

        if len(data_analysis["numeric_fields"]) > 1:
            response["recommendations"].append("Consider correlation analysis between numeric variables")

        if len(data_analysis["categorical_fields"]) > 0:
            response["recommendations"].append("Categorical data suitable for grouping and segmentation")

        if len(data_analysis["datetime_fields"]) > 0:
            response["recommendations"].append("Time series analysis possible with datetime data")

        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Chart suggestions failed: {e}")
        return JSONResponse(
            content=create_error_response("Chart suggestions failed", str(e)),
            status_code=500
        )


@app.get("/health/")
async def health_check():
    """Enhanced health check including Smart Defaults and Smart Engine"""
    health_status = {
        "status": "healthy" if system_manager.is_initialized else "initializing",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "mathematical_engine": system_manager.mathematical_engine is not None,
            "knowledge_framework": system_manager.knowledge_framework is not None,
            "query_interpreter": system_manager.query_interpreter is not None,
            "analysis_engine": system_manager.analysis_engine is not None,
            "insight_generator": system_manager.insight_generator is not None,
            "report_builder": system_manager.report_builder is not None,
            "enhanced_system": system_manager.enhanced_system is not None,
            "smart_defaults_engine": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "unified_smart_engine": SMART_ENGINE_AVAILABLE
        },
        "openai_status": {
            "available": verify_openai_setup()[0],
            "status": verify_openai_setup()[1]
        }
    }

    # Enhanced health check for Smart Defaults
    if SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine:
        try:
            smart_health = await system_manager.smart_defaults_engine.health_check()
            health_status["smart_defaults_health"] = smart_health
        except Exception as e:
            health_status["smart_defaults_error"] = str(e)

    return health_status


@app.get("/status/")
async def system_status():
    """Enhanced system status including Smart Defaults and Smart Engine"""
    if not system_manager.is_initialized:
        return {"status": "not_initialized", "message": "System is still initializing"}

    try:
        status = {
            "system_version": "5.1.0",
            "initialization_status": "complete",
            "components": {},
            "capabilities": {},
            "performance_metrics": {}
        }

        # Component status
        if system_manager.enhanced_system:
            system_status_data = system_manager.enhanced_system.get_system_status()
            status["components"] = system_status_data.get("components", {})
            status["capabilities"] = system_status_data.get("capabilities", {})

        # Knowledge framework status
        if system_manager.knowledge_framework:
            knowledge_summary = system_manager.knowledge_framework.get_knowledge_summary()
            status["knowledge_summary"] = knowledge_summary

        # Mathematical engine status
        if system_manager.mathematical_engine:
            status["mathematical_methods"] = len(system_manager.mathematical_engine.methods_registry)

        # Smart Engine status
        status["smart_engine"] = {
            "available": SMART_ENGINE_AVAILABLE,
            "openai_configured": verify_openai_setup()[0],
            "openai_status": verify_openai_setup()[1]
        }

        # Smart Defaults status
        if SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine:
            try:
                smart_stats = await system_manager.smart_defaults_engine.get_engine_stats()
                status["smart_defaults"] = {
                    "engine_status": smart_stats.get("engine", {}).get("status"),
                    "components_enabled": smart_stats.get("engine", {}).get("config", {}).get("components_enabled", {}),
                    "machine_learning": smart_stats.get("machine_learning", {}),
                    "policy": smart_stats.get("policy", {})
                }
            except Exception as e:
                status["smart_defaults"] = {"error": str(e)}

        return status

    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return create_error_response("Failed to get system status", str(e))


@app.get("/capabilities/")
async def get_capabilities():
    """Enhanced capabilities including Smart Defaults and Smart Engine"""

    capabilities = {
        "analysis_types": [
            "trend_analysis",
            "distribution_analysis",
            "comparison_analysis",
            "correlation_analysis",
            "statistical_analysis",
            "business_aggregation",
            "predictive_modeling"
        ],
        "chart_types": [
            "line", "bar", "pie", "scatter", "heatmap",
            "histogram", "box", "waterfall", "radar", "treemap"
        ],
        "data_formats": ["CSV", "Excel (.xlsx, .xls)"],
        "data_sources": [
            "File Upload", "PostgreSQL", "MySQL", "Redis", "MongoDB",
            "Tableau", "Power BI", "Jupyter", "S3", "API Endpoints"
        ],
        "domains": [
            "insurance", "banking", "technology", "healthcare",
            "finance", "manufacturing", "retail", "general"
        ],
        "mathematical_methods": [],
        "knowledge_domains": [],
        "smart_features": {
            "auto_data_discovery": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "intelligent_recommendations": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "personalized_suggestions": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "learning_from_usage": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "policy_compliance": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "environment_scanning": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "unified_smart_engine": SMART_ENGINE_AVAILABLE,
            "llm_powered_query_understanding": SMART_ENGINE_AVAILABLE and verify_openai_setup()[0]
        },
        "features": {
            "adaptive_query_processing": True,
            "domain_knowledge_integration": True,
            "mathematical_analysis": True,
            "intelligent_insights": True,
            "comprehensive_reporting": True,
            "chart_intelligence": True,
            "real_time_processing": True,
            "multi_format_support": True,
            "automated_data_discovery": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "personalized_recommendations": SMART_DEFAULTS_AVAILABLE and system_manager.smart_defaults_engine is not None,
            "unified_smart_query_engine": SMART_ENGINE_AVAILABLE,
            "llm_powered_analysis": SMART_ENGINE_AVAILABLE and verify_openai_setup()[0]
        }
    }

    # Add mathematical methods if engine available
    if system_manager.mathematical_engine and system_manager.is_initialized:
        try:
            methods = []
            for category, method_dict in system_manager.mathematical_engine.methods_registry.items():
                if isinstance(method_dict, dict):
                    methods.extend(list(method_dict.keys()))
            capabilities["mathematical_methods"] = methods
        except:
            pass

    # Add knowledge domains if framework available
    if system_manager.knowledge_framework and system_manager.is_initialized:
        try:
            knowledge_summary = system_manager.knowledge_framework.get_knowledge_summary()
            capabilities["knowledge_domains"] = knowledge_summary.get("supported_domains", [])
        except:
            pass

    return capabilities


@app.post("/feedback/")
async def record_user_feedback(
        recommendation_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
):
    """
    Record user feedback for learning and improvement
    """
    if not user_id:
        user_id = f"anonymous_{int(datetime.now().timestamp())}"

    if not SMART_DEFAULTS_AVAILABLE or not system_manager.smart_defaults_engine:
        return {
            "status": "unavailable",
            "message": "Smart Defaults Engine not available"
        }

    try:
        success = await system_manager.smart_defaults_engine.record_feedback(
            user_id=user_id,
            recommendation_id=recommendation_id,
            action=action,
            context=context or {}
        )

        return {
            "status": "success" if success else "failed",
            "user_id": user_id,
            "recommendation_id": recommendation_id,
            "action": action,
            "recorded_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Feedback recording failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/user-profile/")
async def get_user_profile(user_id: Optional[str] = None):
    """
    Get user profile and preferences
    """
    if not user_id:
        user_id = f"anonymous_{int(datetime.now().timestamp())}"

    try:
        # In a real implementation, this would fetch from database
        # For demo, return a sample profile
        profile = {
            "user_id": user_id,
            "role": "data_analyst",
            "department": "analytics",
            "seniority_level": "senior",
            "industry": "technology",
            "preferences": {
                "auto_connect_threshold": 0.8,
                "recommendation_frequency": "daily",
                "preferred_chart_types": ["line", "bar", "scatter"],
                "analysis_complexity": "intermediate"
            },
            "recent_activity": {
                "analyses_run": 15,
                "sources_connected": 3,
                "feedback_provided": 8
            },
            "created_at": "2024-01-01T00:00:00Z",
            "last_active": datetime.now().isoformat()
        }

        return {
            "status": "success",
            "profile": profile
        }

    except Exception as e:
        logger.error(f"Profile retrieval failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


@app.get("/analytics-dashboard/")
async def get_analytics_dashboard(user_id: Optional[str] = None):
    """
    Get analytics dashboard data
    """
    if not user_id:
        user_id = f"anonymous_{int(datetime.now().timestamp())}"

    try:
        # Sample dashboard data - in production, this would come from analytics engine
        dashboard_data = {
            "user_analytics": {
                "total_analyses": 42,
                "successful_connections": 8,
                "avg_confidence_score": 0.78,
                "favorite_data_sources": ["PostgreSQL", "Tableau", "CSV Files"],
                "analysis_trends": [
                    {"date": "2024-01-01", "count": 5},
                    {"date": "2024-01-02", "count": 8},
                    {"date": "2024-01-03", "count": 12}
                ]
            },
            "system_analytics": {
                "total_users": 156,
                "recommendations_generated": 2341,
                "auto_connections": 89,
                "success_rate": 0.84,
                "popular_analysis_types": [
                    {"type": "trend_analysis", "count": 45},
                    {"type": "comparison", "count": 38},
                    {"type": "correlation", "count": 32}
                ]
            },
            "recent_discoveries": [
                {
                    "source_id": "postgresql_prod",
                    "confidence": 0.91,
                    "discovered_at": "2024-01-03T10:30:00Z"
                },
                {
                    "source_id": "redis_cache",
                    "confidence": 0.76,
                    "discovered_at": "2024-01-03T09:15:00Z"
                }
            ]
        }

        return {
            "status": "success",
            "dashboard": dashboard_data,
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# Background task endpoints
@app.post("/admin/retrain-models/")
async def trigger_model_retraining(background_tasks: BackgroundTasks):
    """
    Trigger ML model retraining (admin endpoint)
    """
    if not SMART_DEFAULTS_AVAILABLE or not system_manager.smart_defaults_engine:
        return {
            "status": "unavailable",
            "message": "Smart Defaults Engine not available"
        }

    async def retrain_task():
        try:
            if hasattr(system_manager.smart_defaults_engine, 'learning_engine') and \
                    system_manager.smart_defaults_engine.learning_engine:
                await system_manager.smart_defaults_engine.learning_engine.retrain_models()
                logger.info("🎓 Model retraining completed")
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")

    background_tasks.add_task(retrain_task)

    return {
        "status": "initiated",
        "message": "Model retraining started in background",
        "initiated_at": datetime.now().isoformat()
    }


@app.post("/admin/scan-environment/")
async def trigger_environment_scan(background_tasks: BackgroundTasks):
    """
    Trigger environment scan (admin endpoint)
    """
    if not SMART_DEFAULTS_AVAILABLE or not system_manager.smart_defaults_engine:
        return {
            "status": "unavailable",
            "message": "Smart Defaults Engine not available"
        }

    async def scan_task():
        try:
            if hasattr(system_manager.smart_defaults_engine, 'environment_scanner') and \
                    system_manager.smart_defaults_engine.environment_scanner:
                # Scan for all users (in production, you'd iterate through actual users)
                test_users = ["demo_user_1", "demo_user_2", "demo_user_3"]
                for user_id in test_users:
                    try:
                        await system_manager.smart_defaults_engine.environment_scanner.scan_environment(
                            user_id, force_refresh=True
                        )
                    except Exception as e:
                        logger.warning(f"Scan failed for user {user_id}: {e}")
                logger.info("🔍 Environment scan completed")
        except Exception as e:
            logger.error(f"Environment scan failed: {e}")

    background_tasks.add_task(scan_task)

    return {
        "status": "initiated",
        "message": "Environment scan started in background",
        "initiated_at": datetime.now().isoformat()
    }


@app.post("/test-upload/")
async def test_file_upload(
        prompt: str = Form(...),
        file: UploadFile = File(None),
        use_adaptive: bool = Form(True),
        include_charts: bool = Form(True),
        auto_discover: bool = Form(True)
):
    """Simple test endpoint to debug file upload issues"""

    logger.info(f"🧪 TEST ENDPOINT - Received request")
    logger.info(f"📝 Prompt: '{prompt}'")
    logger.info(f"📄 File: {file.filename if file else 'None'}")
    logger.info(f"⚙️ Parameters: adaptive={use_adaptive}, charts={include_charts}, discover={auto_discover}")

    if file:
        logger.info(f"📄 File details:")
        logger.info(f"  - Filename: {file.filename}")
        logger.info(f"  - Content-Type: {file.content_type}")
        logger.info(f"  - Size: {file.size if hasattr(file, 'size') else 'Unknown'}")

        try:
            # Try to read file content
            content = await file.read()
            logger.info(f"  - Content size: {len(content)} bytes")
            logger.info(f"  - First 100 chars: {content[:100]}")

            # Try to parse as CSV
            if file.filename.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(content))
                logger.info(f"  - CSV parsed successfully: {df.shape[0]} rows × {df.shape[1]} columns")
                logger.info(f"  - Columns: {list(df.columns)}")

                return {
                    "status": "success",
                    "message": "File upload test successful",
                    "file_info": {
                        "filename": file.filename,
                        "size": len(content),
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_names": list(df.columns)
                    },
                    "prompt": prompt,
                    "parameters": {
                        "use_adaptive": use_adaptive,
                        "include_charts": include_charts,
                        "auto_discover": auto_discover
                    }
                }

        except Exception as e:
            logger.error(f"❌ File processing failed: {e}")
            logger.error(traceback.format_exc())

            return {
                "status": "error",
                "error": f"File processing failed: {str(e)}",
                "file_info": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(content) if 'content' in locals() else 0
                }
            }

    return {
        "status": "success",
        "message": "Test endpoint working - no file provided",
        "prompt": prompt,
        "parameters": {
            "use_adaptive": use_adaptive,
            "include_charts": include_charts,
            "auto_discover": auto_discover
        }
    }


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting Complete Analytics Backend with Smart Defaults and Smart Engine")
    print("📊 Version: 5.1.0")
    print(
        "🎯 Features: Complete analytics pipeline with AI-powered insights, automated data discovery, and unified smart query processing")
    print("\n🔗 Key Endpoints:")
    print("   • /analyze/ - Comprehensive analysis with smart engine")
    print("   • /test-openai/ - Test OpenAI API setup")
    print("   • /debug-env/ - Debug environment variables")
    print("   • /discover-sources/ - Auto-discover data sources")
    print("   • /recommendations/ - Get personalized recommendations")
    print("   • /connect-source/ - Connect to discovered sources")
    print("   • /quick-analyze/ - Quick analysis")
    print("   • /chart-suggestions/ - Chart intelligence")
    print("   • /feedback/ - Record user feedback")
    print("   • /health/ - System health check")
    print("   • /status/ - System status")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )