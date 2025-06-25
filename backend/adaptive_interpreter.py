# backend/adaptive_interpreter.py - Industry-Agnostic Query Interpreter with Robust Error Handling
import pandas as pd
import numpy as np
import re
import json
import traceback
import time
from typing import Dict, Any, Optional, List, Tuple, Union, Set
from datetime import datetime, date, timedelta
import logging
from collections import Counter, defaultdict
import hashlib
import pickle
import os
import re
from datetime import datetime
import logging

# Optional imports with fallbacks
try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from dateutil.relativedelta import relativedelta

    DATEUTIL_AVAILABLE = True
except ImportError:
    DATEUTIL_AVAILABLE = False

try:
    from dataclasses import dataclass, field
    from enum import Enum

    DATACLASSES_AVAILABLE = True
except ImportError:
    DATACLASSES_AVAILABLE = False

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from functools import lru_cache
    import itertools
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

# Log what's available
if SPACY_AVAILABLE:
    logger.info("✅ spaCy available")
else:
    logger.warning("⚠️ spaCy not available. Install with: pip install spacy && python -m spacy download en_core_web_sm")

if DATEUTIL_AVAILABLE:
    logger.info("✅ python-dateutil available")
else:
    logger.warning("⚠️ python-dateutil not available. Install with: pip install python-dateutil")

if OPENAI_AVAILABLE:
    logger.info("✅ OpenAI available")
else:
    logger.warning("⚠️ OpenAI not available. Install with: pip install openai")

if SKLEARN_AVAILABLE:
    logger.info("✅ scikit-learn available")
else:
    logger.warning("⚠️ scikit-learn not available. Install with: pip install scikit-learn")


class DataProfiler:
    """Automatically profile any dataset to extract domain knowledge"""

    def __init__(self):
        self.column_types = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.date_columns = []
        self.text_columns = []
        self.metric_columns = []
        self.dimension_columns = []
        self.business_entities = set()
        self.domain_vocabulary = set()

    def profile_dataset(self, df: pd.DataFrame, sample_size: int = 10000) -> Dict[str, Any]:
        """Automatically profile any dataset to understand its structure"""
        if df.empty:
            return {}

        # Sample large datasets for performance
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df.copy()

        profile = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_analysis': {},
            'inferred_domain': None,
            'business_entities': [],
            'suggested_metrics': [],
            'suggested_dimensions': [],
            'temporal_columns': [],
            'categorical_mappings': {},
            'numerical_distributions': {},
            'domain_keywords': []
        }

        # Analyze each column
        for col in df.columns:
            col_analysis = self._analyze_column(df_sample[col], col)
            profile['column_analysis'][col] = col_analysis

            # Categorize columns by type
            if col_analysis['inferred_type'] == 'categorical':
                self.categorical_columns.append(col)
                profile['suggested_dimensions'].append(col)
            elif col_analysis['inferred_type'] == 'numerical':
                self.numerical_columns.append(col)
                if self._is_likely_metric(col, col_analysis):
                    profile['suggested_metrics'].append(col)
                    self.metric_columns.append(col)
                else:
                    profile['suggested_dimensions'].append(col)
                    self.dimension_columns.append(col)
            elif col_analysis['inferred_type'] == 'datetime':
                self.date_columns.append(col)
                profile['temporal_columns'].append(col)
            elif col_analysis['inferred_type'] == 'text':
                self.text_columns.append(col)

        # Infer business domain
        profile['inferred_domain'] = self._infer_business_domain(df_sample, profile)

        # Extract business vocabulary
        profile['domain_keywords'] = self._extract_domain_vocabulary(df_sample, profile)

        # Generate dynamic patterns
        profile['dynamic_patterns'] = self._generate_dynamic_patterns(profile)

        return profile

    def _analyze_column(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """Analyze individual column characteristics"""
        analysis = {
            'name': col_name,
            'dtype': str(series.dtype),
            'null_count': series.isnull().sum(),
            'unique_count': series.nunique(),
            'inferred_type': None,
            'business_meaning': None,
            'sample_values': [],
            'patterns': [],
            'likely_categorical': False,
            'likely_metric': False,
            'likely_dimension': False
        }

        # Get sample values (non-null)
        sample_values = series.dropna().head(10).tolist()
        analysis['sample_values'] = [str(v) for v in sample_values]

        # Infer data type
        if pd.api.types.is_datetime64_any_dtype(series):
            analysis['inferred_type'] = 'datetime'
        elif pd.api.types.is_numeric_dtype(series):
            analysis['inferred_type'] = 'numerical'
            analysis['likely_metric'] = self._is_likely_metric(col_name, analysis)
        elif series.nunique() / len(series) < 0.1 and series.nunique() < 50:
            analysis['inferred_type'] = 'categorical'
            analysis['likely_categorical'] = True
            analysis['likely_dimension'] = True
        else:
            analysis['inferred_type'] = 'text'

        # Infer business meaning from column name
        analysis['business_meaning'] = self._infer_business_meaning(col_name)

        # Extract patterns from values
        analysis['patterns'] = self._extract_value_patterns(sample_values)

        return analysis

    def _is_likely_metric(self, col_name: str, col_analysis: Dict[str, Any]) -> bool:
        """Determine if column is likely a business metric"""
        col_lower = col_name.lower()

        # Metric indicators in column name
        metric_indicators = [
            'amount', 'total', 'sum', 'count', 'rate', 'ratio', 'percent',
            'revenue', 'cost', 'price', 'value', 'income', 'expense',
            'profit', 'loss', 'margin', 'volume', 'quantity', 'score',
            'balance', 'outstanding', 'due', 'paid', 'received', 'earned',
            'sales'  # Added sales as explicit metric indicator
        ]

        # Check if column name contains metric indicators
        if any(indicator in col_lower for indicator in metric_indicators):
            return True

        # Check if values look like metrics (large numbers, decimals, monetary)
        if col_analysis['inferred_type'] == 'numerical':
            sample_values = col_analysis['sample_values']
            if sample_values:
                try:
                    # Convert sample values to float
                    numeric_values = [float(v) for v in sample_values[:5] if
                                      str(v).replace('.', '').replace('-', '').isdigit()]
                    if numeric_values:
                        # Large numbers might be metrics
                        if any(abs(v) > 1000 for v in numeric_values):
                            return True
                        # Decimal values between 0-1 might be rates/ratios
                        if any(0 < abs(v) < 1 for v in numeric_values):
                            return True
                except:
                    pass

        return False

    def _infer_business_meaning(self, col_name: str) -> str:
        """Infer business meaning from column name"""
        col_lower = col_name.lower().replace('_', ' ').replace('-', ' ')

        # Business domain mappings
        domain_mappings = {
            'financial': ['amount', 'cost', 'price', 'revenue', 'profit', 'balance', 'payment', 'transaction', 'sales'],
            'temporal': ['date', 'time', 'created', 'updated', 'modified', 'start', 'end', 'duration', 'month', 'year'],
            'geographic': ['country', 'state', 'city', 'region', 'location', 'address', 'zip', 'postal'],
            'demographic': ['age', 'gender', 'sex', 'birth', 'married', 'education', 'income_level'],
            'product': ['product', 'item', 'sku', 'category', 'type', 'model', 'brand', 'service'],
            'customer': ['customer', 'client', 'user', 'account', 'member', 'subscriber'],
            'operational': ['status', 'stage', 'priority', 'level', 'grade', 'class', 'tier'],
            'performance': ['score', 'rating', 'rank', 'performance', 'efficiency', 'quality'],
            'identifier': ['id', 'key', 'code', 'number', 'reference', 'identifier']
        }

        for domain, keywords in domain_mappings.items():
            if any(keyword in col_lower for keyword in keywords):
                return domain

        return 'unknown'

    def _extract_value_patterns(self, values: List[str]) -> List[str]:
        """Extract patterns from column values"""
        patterns = []

        for value in values[:5]:  # Analyze first 5 values
            value_str = str(value)

            # Check for common patterns
            if re.match(r'^\d{4}-\d{2}-\d{2}', value_str):
                patterns.append('date_iso')
            elif re.match(r'^\d+\.\d+$', value_str):
                patterns.append('decimal')
            elif re.match(r'^\d+$', value_str):
                patterns.append('integer')
            elif re.match(r'^[A-Z]{2,}$', value_str):
                patterns.append('code_uppercase')
            elif re.match(r'^[a-zA-Z\s]+$', value_str):
                patterns.append('text_alphabetic')
            elif '$' in value_str:
                patterns.append('currency')
            elif '%' in value_str:
                patterns.append('percentage')

        return list(set(patterns))

    def _infer_business_domain(self, df: pd.DataFrame, profile: Dict[str, Any]) -> str:
        """Infer the business domain from column names and values"""
        column_names = ' '.join(df.columns).lower()

        # Domain indicators
        domain_indicators = {
            'insurance': ['policy', 'claim', 'premium', 'coverage', 'deductible', 'beneficiary', 'mortality', 'lapse'],
            'banking': ['account', 'balance', 'transaction', 'deposit', 'withdrawal', 'loan', 'credit', 'interest'],
            'ecommerce': ['order', 'product', 'customer', 'cart', 'purchase', 'price', 'inventory', 'shipping',
                          'sales'],
            'healthcare': ['patient', 'diagnosis', 'treatment', 'hospital', 'doctor', 'medication', 'visit'],
            'finance': ['investment', 'portfolio', 'stock', 'bond', 'return', 'risk', 'market', 'trading'],
            'hr': ['employee', 'salary', 'department', 'hire', 'performance', 'review', 'training'],
            'marketing': ['campaign', 'lead', 'conversion', 'impression', 'click', 'engagement', 'audience'],
            'technology': ['user', 'session', 'event', 'feature', 'bug', 'deployment', 'api', 'server'],
            'sales': ['lead', 'opportunity', 'deal', 'pipeline', 'quota', 'commission', 'territory', 'sales',
                      'revenue'],
            'manufacturing': ['production', 'inventory', 'quality', 'defect', 'batch', 'supplier', 'warehouse']
        }

        domain_scores = {}
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in column_names)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return 'general'

    def _extract_domain_vocabulary(self, df: pd.DataFrame, profile: Dict[str, Any]) -> List[str]:
        """Extract domain-specific vocabulary from the dataset"""
        vocabulary = set()

        # Extract from column names
        for col in df.columns:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', col.lower().replace('_', ' '))
            vocabulary.update(words)

        # Extract from categorical values
        for col in self.categorical_columns[:5]:  # Limit to avoid performance issues
            if col in df.columns:
                unique_values = df[col].dropna().unique()[:20]  # Sample values
                for val in unique_values:
                    words = re.findall(r'\b[a-zA-Z]{3,}\b', str(val).lower())
                    vocabulary.update(words)

        # Filter out common English words
        common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one',
                        'our', 'has', 'have'}
        vocabulary = vocabulary - common_words

        return sorted(list(vocabulary))

    def _generate_dynamic_patterns(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dynamic patterns based on discovered data"""
        patterns = {
            'intent_patterns': {},
            'metric_patterns': {},
            'dimension_patterns': {},
            'domain_patterns': {}
        }

        # Generate intent patterns based on metrics
        if profile['suggested_metrics']:
            patterns['intent_patterns']['trend_analysis'] = ['trend', 'over time', 'pattern', 'change'] + [
                f"{metric} trend" for metric in profile['suggested_metrics'][:3]]
            patterns['intent_patterns']['summary'] = ['total', 'sum', 'aggregate'] + [f"total {metric}" for metric in
                                                                                      profile['suggested_metrics'][:3]]
            patterns['intent_patterns']['comparison'] = ['compare', 'vs', 'versus'] + [f"compare {metric}" for metric in
                                                                                       profile['suggested_metrics'][:3]]

        # Generate metric patterns from discovered metrics
        for metric in profile['suggested_metrics']:
            metric_variations = [
                metric.lower(),
                metric.lower().replace('_', ' '),
                metric.lower().replace('_', ''),
                ' '.join(metric.split('_'))
            ]
            patterns['metric_patterns'][metric] = list(set(metric_variations))

        # Generate dimension patterns from discovered dimensions
        for dim in profile['suggested_dimensions']:
            dim_variations = [
                dim.lower(),
                dim.lower().replace('_', ' '),
                f"by {dim.lower()}",
                f"per {dim.lower()}"
            ]
            patterns['dimension_patterns'][dim] = list(set(dim_variations))

        # Add domain-specific patterns
        domain = profile.get('inferred_domain', 'general')
        patterns['domain_patterns'][domain] = profile.get('domain_keywords', [])

        return patterns


class AdaptiveOpenAISchema:
    """Dynamically generate OpenAI function schemas based on data profile"""

    def __init__(self, data_profile: Dict[str, Any]):
        self.data_profile = data_profile

    def generate_schema(self) -> Dict[str, Any]:
        """Generate dynamic OpenAI function schema"""
        # Extract available options from data
        available_metrics = self.data_profile.get('unified_metrics', ['value', 'amount', 'count'])
        available_dimensions = self.data_profile.get('unified_dimensions', ['category', 'type', 'status'])
        domain = self.data_profile.get('primary_domain', 'general')

        # Generate domain-specific intents
        base_intents = [
            "trend_analysis", "summary", "comparison", "correlation",
            "distribution", "forecast", "risk_analysis"
        ]

        domain_specific_intents = {
            'insurance': ["profitability_assessment", "claims_review", "risk_analysis"],
            'banking': ["credit_analysis", "transaction_review", "fraud_detection"],
            'ecommerce': ["sales_analysis", "customer_behavior", "inventory_analysis"],
            'finance': ["portfolio_analysis", "performance_review", "risk_assessment"],
            'technology': ["usage_analysis", "performance_monitoring", "feature_adoption"],
            'sales': ["sales_analysis", "pipeline_review", "performance_tracking"]
        }

        intents = base_intents + domain_specific_intents.get(domain, [])

        schema = {
            "name": "interpret_adaptive_query",
            "description": f"Extract structured analytics parameters for {domain} domain with discovered metrics and dimensions",
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": intents,
                        "description": f"Primary analytical intent for {domain} domain"
                    },
                    "metric": {
                        "type": "string",
                        "enum": available_metrics + ["custom"],
                        "description": f"Primary metric from available columns: {', '.join(available_metrics[:10])}"
                    },
                    "custom_metric": {
                        "type": "string",
                        "description": "Custom metric name if not in predefined list"
                    },
                    "granularity": {
                        "type": "string",
                        "enum": ["daily", "weekly", "monthly", "quarterly", "yearly", "total"],
                        "description": "Time aggregation level"
                    },
                    "temporal_expression": {
                        "type": "object",
                        "properties": {
                            "type": {"enum": ["specific_period", "date_range", "relative", "all_time"]},
                            "value": {"type": "string"},
                            "start_date": {"type": "string"},
                            "end_date": {"type": "string"}
                        }
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"enum": available_dimensions + ["custom"]},
                        "description": f"Grouping dimensions from: {', '.join(available_dimensions[:10])}"
                    },
                    "custom_group_by": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Custom grouping dimensions if not in predefined list"
                    },
                    "filters": {
                        "type": "object",
                        "description": f"Filters based on available categorical columns"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence in interpretation accuracy"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation of interpretation reasoning"
                    }
                },
                "required": ["intent", "metric", "granularity", "confidence"]
            }
        }

        return schema


class AdaptiveQueryInterpreter:
    """Industry-agnostic query interpreter that adapts to any dataset"""

    def __init__(self, openai_api_key: Optional[str] = None):
        # Initialize OpenAI client only if available and key provided
        if OPENAI_AVAILABLE and openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=openai_api_key)
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI client initialization failed: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
            if openai_api_key and not OPENAI_AVAILABLE:
                logger.warning("⚠️ OpenAI API key provided but OpenAI package not available")

        self.data_profiler = DataProfiler()
        self.current_profile = None
        self.adaptive_schema = None
        self.domain_patterns = {}
        self.learned_patterns = {}

        # Load spaCy if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("✅ spaCy model loaded")
            except OSError:
                self.nlp = None
                logger.warning("⚠️ spaCy model not available. Install with: python -m spacy download en_core_web_sm")
        else:
            self.nlp = None

    def learn_from_data(self, datasets: Dict[str, pd.DataFrame], cache_path: Optional[str] = None) -> Dict[str, Any]:
        """Learn patterns and vocabulary from provided datasets"""
        logger.info("🧠 Learning from data...")

        # Check cache first
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_profile = pickle.load(f)
                    logger.info("📁 Loaded cached data profile")
                    self.current_profile = cached_profile
                    if OPENAI_AVAILABLE:
                        self.adaptive_schema = AdaptiveOpenAISchema(cached_profile)
                    return cached_profile
            except Exception as e:
                logger.warning(f"Failed to load cached profile: {e}, regenerating...")

        # Combine all datasets for comprehensive learning
        combined_profile = {
            'datasets': {},
            'unified_metrics': set(),
            'unified_dimensions': set(),
            'unified_vocabulary': set(),
            'inferred_domains': [],
            'column_mappings': {},
            'business_rules': []
        }

        for dataset_name, df in datasets.items():
            logger.info(f"📊 Profiling dataset: {dataset_name}")
            profile = self.data_profiler.profile_dataset(df)
            combined_profile['datasets'][dataset_name] = profile

            # Accumulate learning
            combined_profile['unified_metrics'].update(profile.get('suggested_metrics', []))
            combined_profile['unified_dimensions'].update(profile.get('suggested_dimensions', []))
            combined_profile['unified_vocabulary'].update(profile.get('domain_keywords', []))
            if profile.get('inferred_domain'):
                combined_profile['inferred_domains'].append(profile['inferred_domain'])

        # Determine primary domain
        if combined_profile['inferred_domains']:
            domain_counts = Counter(combined_profile['inferred_domains'])
            combined_profile['primary_domain'] = domain_counts.most_common(1)[0][0]
        else:
            combined_profile['primary_domain'] = 'general'

        # Convert sets to lists for JSON serialization
        combined_profile['unified_metrics'] = sorted(list(combined_profile['unified_metrics']))
        combined_profile['unified_dimensions'] = sorted(list(combined_profile['unified_dimensions']))
        combined_profile['unified_vocabulary'] = sorted(list(combined_profile['unified_vocabulary']))

        # Generate adaptive patterns
        combined_profile['adaptive_patterns'] = self._generate_adaptive_patterns(combined_profile)

        # Cache the learning
        if cache_path:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, 'wb') as f:
                    pickle.dump(combined_profile, f)
                logger.info(f"💾 Cached learning to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to cache profile: {e}")

        self.current_profile = combined_profile
        if OPENAI_AVAILABLE:
            self.adaptive_schema = AdaptiveOpenAISchema(combined_profile)

        logger.info(f"✅ Learning complete!")
        logger.info(f"   📈 Discovered {len(combined_profile['unified_metrics'])} metrics")
        logger.info(f"   📊 Discovered {len(combined_profile['unified_dimensions'])} dimensions")
        logger.info(f"   🎯 Primary domain: {combined_profile['primary_domain']}")

        return combined_profile

    def _generate_adaptive_patterns(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate patterns that adapt to the discovered data"""
        patterns = {
            'intent_keywords': defaultdict(list),
            'metric_synonyms': defaultdict(list),
            'dimension_synonyms': defaultdict(list),
            'domain_vocabulary': defaultdict(list)
        }

        # Generate intent patterns based on discovered metrics and domain
        domain = profile.get('primary_domain', 'general')
        metrics = profile.get('unified_metrics', [])
        dimensions = profile.get('unified_dimensions', [])

        # Intent patterns
        patterns['intent_keywords']['trend_analysis'].extend([
            'trend', 'over time', 'pattern', 'change', 'evolution', 'progression', 'highest', 'lowest', 'maximum',
            'minimum', 'peak'
        ])
        patterns['intent_keywords']['summary'].extend([
            'total', 'sum', 'aggregate', 'overall', 'summary', 'grand total'
        ])
        patterns['intent_keywords']['comparison'].extend([
            'compare', 'vs', 'versus', 'difference', 'between', 'against'
        ])
        patterns['intent_keywords']['distribution'].extend([
            'breakdown', 'distribution', 'spread', 'across', 'by category'
        ])

        # Add domain-specific intent patterns
        domain_intents = {
            'insurance': {
                'risk_analysis': ['risk', 'exposure', 'vulnerability', 'threat'],
                'profitability_assessment': ['profit', 'profitability', 'margin', 'earnings', 'ROI']
            },
            'banking': {
                'credit_analysis': ['credit', 'creditworthiness', 'default', 'risk'],
                'fraud_detection': ['fraud', 'suspicious', 'anomaly', 'irregular']
            },
            'technology': {
                'performance_monitoring': ['performance', 'latency', 'throughput', 'uptime'],
                'usage_analysis': ['usage', 'adoption', 'engagement', 'activity']
            },
            'sales': {
                'sales_analysis': ['sales', 'revenue', 'performance', 'growth'],
                'pipeline_review': ['pipeline', 'funnel', 'conversion', 'leads']
            }
        }

        if domain in domain_intents:
            for intent, keywords in domain_intents[domain].items():
                patterns['intent_keywords'][intent].extend(keywords)

        # Generate metric synonyms
        for metric in metrics:
            synonyms = [metric.lower()]

            # Add variations
            synonyms.append(metric.lower().replace('_', ' '))
            synonyms.append(metric.lower().replace('_', ''))

            # Add domain-specific synonyms
            if 'amount' in metric.lower():
                synonyms.extend(['value', 'sum', 'total'])
            if 'count' in metric.lower():
                synonyms.extend(['number', 'quantity', 'volume'])
            if 'rate' in metric.lower():
                synonyms.extend(['ratio', 'percentage', 'proportion'])
            if 'sales' in metric.lower():
                synonyms.extend(['revenue', 'income', 'earnings'])

            patterns['metric_synonyms'][metric] = list(set(synonyms))

        # Generate dimension synonyms
        for dimension in dimensions:
            synonyms = [dimension.lower()]
            synonyms.append(dimension.lower().replace('_', ' '))
            synonyms.append(f"by {dimension.lower()}")
            synonyms.append(f"per {dimension.lower()}")

            # Add semantic variations
            if 'type' in dimension.lower():
                synonyms.extend(['category', 'kind', 'class'])
            if 'status' in dimension.lower():
                synonyms.extend(['state', 'condition', 'stage'])
            if 'date' in dimension.lower():
                synonyms.extend(['time', 'period', 'when', 'month', 'year'])

            patterns['dimension_synonyms'][dimension] = list(set(synonyms))

        return patterns

    def parse(self, query: str, target_dataset: Optional[str] = None) -> Dict[str, Any]:
        """Parse query using adaptive patterns learned from data"""
        if not self.current_profile:
            # If no profile, create a basic one for fallback
            logger.warning("No data profile available. Using basic parsing.")
            return self._parse_with_basic_patterns(query)

        start_time = time.time()

        # Try multiple parsing approaches
        result = None

        # 1. Try OpenAI with adaptive schema (if available)
        if self.openai_client and self.adaptive_schema:
            try:
                result = self._parse_with_adaptive_openai(query, target_dataset, start_time)
                if result and result.get('confidence', 0) > 0.7:
                    logger.info(f"✅ High-confidence OpenAI parsing: {result['confidence']:.3f}")
                    return result
            except Exception as e:
                logger.warning(f"OpenAI parsing failed: {e}")

        # 2. Try adaptive pattern matching
        try:
            result = self._parse_with_adaptive_patterns(query, target_dataset)
            if result and result.get('confidence', 0) > 0.5:
                logger.info(f"✅ Adaptive pattern parsing: {result['confidence']:.3f}")
                return result
        except Exception as e:
            logger.warning(f"Adaptive pattern parsing failed: {e}")

        # 3. Fallback to basic parsing
        try:
            result = self._parse_with_basic_patterns(query)
            logger.info(f"⚠️ Fallback parsing: {result.get('confidence', 0):.3f}")
            return result
        except Exception as e:
            logger.error(f"All parsing methods failed: {e}")
            return self._create_error_result(query, str(e))

    def _parse_with_adaptive_openai(self, query: str, target_dataset: Optional[str] = None, start_time=None) -> Dict[
        str, Any]:
        """Parse using OpenAI with dynamically generated schema"""
        if not self.adaptive_schema:
            raise ValueError("Adaptive schema not available")

        # Get relevant profile
        if target_dataset and target_dataset in self.current_profile.get('datasets', {}):
            dataset_profile = self.current_profile['datasets'][target_dataset]
        else:
            dataset_profile = self.current_profile

        schema = self.adaptive_schema.generate_schema()

        # Enhanced system prompt with discovered information
        available_metrics = dataset_profile.get('unified_metrics', [])
        available_dimensions = dataset_profile.get('unified_dimensions', [])
        domain = dataset_profile.get('primary_domain', 'general')

        system_prompt = f"""You are an expert {domain} analytics interpreter. Current date: {datetime.now().strftime('%Y-%m-%d')}.

DISCOVERED DATA CONTEXT:
- Domain: {domain}
- Available Metrics: {', '.join(available_metrics[:15])}
- Available Dimensions: {', '.join(available_dimensions[:15])}

INSTRUCTIONS:
1. Map user query to available metrics and dimensions
2. If exact match not found, use closest semantic match or set metric/group_by to "custom"
3. Provide high confidence for exact matches, lower for semantic matches
4. For custom metrics/dimensions, specify in custom_metric/custom_group_by fields
5. Always provide reasoning for your interpretation

EXAMPLES for {domain} domain:
- "total sales by region" → metric="sales_amount" (if available), group_by=["region_code"]
- "monthly revenue trends" → metric="revenue", granularity="monthly", intent="trend_analysis"
- "compare performance across teams" → intent="comparison", group_by=["team_name"]
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                functions=[schema],
                function_call={"name": "interpret_adaptive_query"},
                temperature=0.1,
                timeout=15
            )

            func_call = response.choices[0].message.function_call
            if not func_call:
                raise ValueError("No function call in response")

            args = json.loads(func_call.arguments)

            # Post-process the result
            result = self._post_process_adaptive_result(args, query, available_metrics, available_dimensions)
            result['method'] = 'adaptive_openai'
            if start_time:
                result['parsing_time_ms'] = (time.time() - start_time) * 1000

            return result

        except Exception as e:
            logger.error(f"Adaptive OpenAI parsing failed: {e}")
            raise e

    def _parse_multiple_grouping_dimensions(self, query: str, available_dimensions: List[str]) -> List[str]:
        """Enhanced parsing for multiple grouping dimensions"""
        print(f"=== MULTIPLE GROUPING DIMENSION PARSING ===")
        print(f"Query: '{query}'")
        print(f"Available dimensions: {available_dimensions}")

        found_dimensions = []

        # Enhanced patterns for multiple groupings
        multi_grouping_patterns = [
            # "by region, by month and product type"
            r'\bby\s+([^,]+(?:,\s*by\s+[^,]+)*(?:\s+and\s+[^,]+)?)',
            # "breakdown by region and month"
            r'\bbreakdown\s+by\s+([^,]+(?:\s+and\s+[^,]+)*)',
            # "group by region, month, product"
            r'\bgroup\s+by\s+([^,]+(?:,\s*[^,]+)*)',
            # "per region and month"
            r'\bper\s+([^,]+(?:\s+and\s+[^,]+)*)',
            # "across region, month and product"
            r'\bacross\s+([^,]+(?:,\s*[^,]+)*)'
        ]

        for pattern in multi_grouping_patterns:
            match = re.search(pattern, query.lower())
            if match:
                grouping_text = match.group(1)
                print(f"Found grouping text: '{grouping_text}'")

                # Split on common separators
                dimensions_text = re.split(r',\s*(?:by\s+)?|and\s+', grouping_text)

                for dim_text in dimensions_text:
                    dim_text = dim_text.strip()
                    if not dim_text:
                        continue

                    print(f"Processing dimension: '{dim_text}'")

                    # Direct match
                    if dim_text in available_dimensions:
                        found_dimensions.append(dim_text)
                        print(f"Direct match: {dim_text}")
                    else:
                        # Fuzzy matching for common terms
                        dimension_mappings = {
                            'region': ['region', 'area', 'zone', 'territory'],
                            'product': ['product', 'item', 'sku'],
                            'category': ['category', 'type', 'class', 'segment'],
                            'month': ['month', 'monthly', 'mon'],
                            'year': ['year', 'yearly', 'yr'],
                            'quarter': ['quarter', 'quarterly', 'q'],
                            'customer': ['customer', 'client', 'account'],
                            'channel': ['channel', 'source', 'medium'],
                            'store': ['store', 'location', 'branch']
                        }

                        # Check if any mapping matches
                        for canonical, variants in dimension_mappings.items():
                            if any(variant in dim_text.lower() for variant in variants):
                                # Find actual column that matches
                                matching_cols = [col for col in available_dimensions
                                                 if any(variant in col.lower() for variant in variants)]
                                if matching_cols:
                                    found_dimensions.append(matching_cols[0])
                                    print(f"Mapped '{dim_text}' -> '{matching_cols[0]}'")
                                    break
                break  # Use first pattern that matches

        # Remove duplicates while preserving order
        unique_dimensions = []
        for dim in found_dimensions:
            if dim not in unique_dimensions:
                unique_dimensions.append(dim)

        print(f"Final dimensions: {unique_dimensions}")
        return unique_dimensions

    def _parse_with_adaptive_patterns(self, query: str, target_dataset: Optional[str] = None) -> Dict[str, Any]:
        """Enhanced parsing with multiple grouping support"""
        start_time = time.time()
        query_lower = query.lower().strip()

        print(f"=== ENHANCED MULTI-GROUPING PARSER ===")
        print(f"Query: '{query}'")

        # Get relevant patterns
        patterns = self.current_profile.get('adaptive_patterns', {})
        available_metrics = self.current_profile.get('unified_metrics', [])
        available_dimensions = self.current_profile.get('unified_dimensions', [])

        result = {
            'intent': 'summary',
            'metric': None,
            'custom_metric': None,
            'granularity': 'total',
            'temporal_expression': {'type': 'all_time'},
            'group_by': [],
            'custom_group_by': [],
            'filters': {},
            'confidence': 0.5,
            'reasoning': '',
            'method': 'adaptive_patterns',
            'raw_query': query
        }

        confidence_factors = []

        # Intent detection
        intent_found = False

        if any(word in query_lower for word in ['breakdown', 'break down', 'group by', 'by ', 'per ', 'across']):
            if 'breakdown' in query_lower or 'break down' in query_lower:
                result['intent'] = 'distribution_analysis'
                confidence_factors.append(('breakdown_intent', 0.25))
                intent_found = True
            elif any(word in query_lower for word in ['by ', 'per ', 'group by']):
                result['intent'] = 'comparison_analysis'
                confidence_factors.append(('grouping_intent', 0.25))
                intent_found = True

        # Metric detection
        metric_found = False

        for available_metric in available_metrics:
            if available_metric.lower() in query_lower:
                result['metric'] = available_metric
                metric_found = True
                confidence_factors.append(('direct_metric_match', 0.3))
                break

        if not metric_found:
            metric_mappings = {
                'sales': 'sales',
                'revenue': 'sales',
                'profit': 'profit',
                'amount': 'sales',
                'value': 'sales',
                'total': 'sales',
                'sum': 'sales'
            }

            for term, mapped_metric in metric_mappings.items():
                if term in query_lower and mapped_metric in available_metrics:
                    result['metric'] = mapped_metric
                    metric_found = True
                    confidence_factors.append(('mapped_metric', 0.25))
                    break

        # ENHANCED: Multiple dimension detection
        found_dimensions = self._parse_multiple_grouping_dimensions(query, available_dimensions)

        if found_dimensions:
            result['group_by'] = found_dimensions
            result['granularity'] = 'grouped'
            confidence_factors.append(('multiple_dimensions', 0.3))

            # Boost confidence for multiple groupings
            if len(found_dimensions) > 1:
                confidence_factors.append(('complex_grouping', 0.2))

        # Enhanced temporal detection
        temporal_result = self._extract_temporal_expression(query)
        if temporal_result:
            result['temporal_expression'] = temporal_result
            confidence_factors.append(('temporal_found', 0.15))

        # Calculate final confidence
        base_confidence = 0.4
        confidence_boost = sum(factor[1] for factor in confidence_factors)
        result['confidence'] = min(base_confidence + confidence_boost, 0.95)

        # Generate reasoning
        reasoning_parts = []
        if metric_found or result.get('custom_metric'):
            metric_name = result.get('metric') or result.get('custom_metric')
            reasoning_parts.append(f"Found metric '{metric_name}'")
        if found_dimensions:
            reasoning_parts.append(f"Found {len(found_dimensions)} grouping dimensions: {', '.join(found_dimensions)}")
        if result['intent'] in ['distribution_analysis', 'comparison_analysis']:
            reasoning_parts.append(f"Detected {result['intent'].replace('_', ' ')}")

        result['reasoning'] = '; '.join(reasoning_parts) if reasoning_parts else 'Basic pattern matching'
        result['parsing_time_ms'] = (time.time() - start_time) * 1000

        print(f"Final result: {result}")
        return result

    def _extract_potential_metrics(self, query: str, known_metrics: List[str]) -> List[str]:
        """Extract potential metric names from query when no exact match found"""
        # Look for metric-like words
        metric_indicators = ['amount', 'total', 'sum', 'count', 'rate', 'ratio', 'value', 'score', 'number', 'sales',
                             'revenue']

        words = re.findall(r'\b\w+\b', query)
        potential_metrics = []

        for i, word in enumerate(words):
            # Check if word + next word could be a metric
            if i < len(words) - 1:
                compound = f"{word}_{words[i + 1]}"
                if any(indicator in compound.lower() for indicator in metric_indicators):
                    potential_metrics.append(compound)

            # Check single words
            if any(indicator in word.lower() for indicator in metric_indicators):
                potential_metrics.append(word)

        return potential_metrics[:3]  # Return top 3 candidates

    def _extract_potential_dimensions(self, query: str, known_dimensions: List[str]) -> List[str]:
        """Extract potential dimension names from query"""
        # Look for grouping indicators
        grouping_patterns = [
            r'by\s+(\w+)',
            r'per\s+(\w+)',
            r'across\s+(\w+)',
            r'group\s+by\s+(\w+)',
            r'broken\s+down\s+by\s+(\w+)'
        ]

        potential_dims = []
        for pattern in grouping_patterns:
            matches = re.findall(pattern, query)
            potential_dims.extend(matches)

        return potential_dims[:3]

    def _extract_temporal_expression(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract temporal expressions adaptively - FIXED for year queries"""
        current_year = datetime.now().year

        # Enhanced year pattern matching
        year_patterns = [
            r'\b(20\d{2})\b',  # 2024, 2023, etc.
            r'\byear\s+(20\d{2})\b',  # "year 2024"
            r'\bin\s+(20\d{2})\b',  # "in 2024"
            r'\bof\s+(20\d{2})\b',  # "of 2024"
            r'\bfor\s+(20\d{2})\b',  # "for 2024"
        ]

        for pattern in year_patterns:
            year_match = re.search(pattern, query, re.IGNORECASE)
            if year_match:
                year = int(year_match.group(1))
                return {
                    'type': 'specific_period',
                    'value': str(year),
                    'start_date': f'{year}-01-01',
                    'end_date': f'{year}-12-31'
                }

        # Month patterns
        months = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }

        for month_name, month_num in months.items():
            if month_name in query.lower():
                return {
                    'type': 'specific_period',
                    'value': f'{current_year}-{month_num:02d}',
                    'start_date': f'{current_year}-{month_num:02d}-01',
                    'end_date': f'{current_year}-{month_num:02d}-{self._get_month_end_day(current_year, month_num)}'
                }

        # Relative patterns
        relative_patterns = {
            r'last\s+(\d+)\s+months?': lambda m: self._create_relative_range('months', int(m.group(1))),
            r'past\s+(\d+)\s+months?': lambda m: self._create_relative_range('months', int(m.group(1))),
            r'last\s+year': lambda m: self._create_relative_range('years', 1),
            r'ytd|year.to.date': lambda m: self._create_ytd_range()
        }

        for pattern, handler in relative_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return handler(match)

        return None

    def _get_month_end_day(self, year: int, month: int) -> int:
        """Get last day of month"""
        if month == 12:
            next_month = date(year + 1, 1, 1)
        else:
            next_month = date(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        return last_day.day

    def _create_relative_range(self, unit: str, amount: int) -> Dict[str, Any]:
        """Create relative date range"""
        end_date = datetime.now().date()

        if unit == 'months' and DATEUTIL_AVAILABLE:
            start_date = end_date - relativedelta(months=amount)
        elif unit == 'years' and DATEUTIL_AVAILABLE:
            start_date = end_date - relativedelta(years=amount)
        else:
            # Fallback without dateutil
            days_approx = amount * 365 if unit == 'years' else amount * 30
            start_date = end_date - timedelta(days=days_approx)

        return {
            'type': 'date_range',
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'description': f'Last {amount} {unit}'
        }

    def _create_ytd_range(self) -> Dict[str, Any]:
        """Create year-to-date range"""
        today = datetime.now().date()
        start_of_year = date(today.year, 1, 1)

        return {
            'type': 'date_range',
            'start_date': start_of_year.isoformat(),
            'end_date': today.isoformat(),
            'description': f'Year to date {today.year}'
        }

    def _extract_adaptive_filters(self, query: str, target_dataset: Optional[str] = None) -> Dict[str, Any]:
        """Extract filters based on learned categorical values"""
        filters = {}

        if not target_dataset or target_dataset not in self.current_profile.get('datasets', {}):
            return filters

        dataset_profile = self.current_profile['datasets'][target_dataset]
        column_analysis = dataset_profile.get('column_analysis', {})

        # Look for potential filter values in categorical columns
        for col_name, col_info in column_analysis.items():
            if col_info.get('inferred_type') == 'categorical':
                sample_values = col_info.get('sample_values', [])
                for value in sample_values:
                    if str(value).lower() in query:
                        filters[col_name] = value
                        break

        return filters

    def _parse_with_basic_patterns(self, query: str) -> Dict[str, Any]:
        """Enhanced basic fallback parsing when adaptive methods fail"""
        start_time = time.time()
        query_lower = query.lower().strip()

        result = {
            'intent': 'summary',
            'metric': 'custom',
            'custom_metric': 'value',
            'granularity': 'total',
            'temporal_expression': {'type': 'all_time'},
            'group_by': [],
            'filters': {},
            'confidence': 0.3,
            'reasoning': 'Fallback basic parsing',
            'method': 'basic_fallback',
            'raw_query': query
        }

        # Enhanced intent detection
        if any(word in query_lower for word in ['trend', 'over time', 'monthly', 'quarterly', 'pattern']):
            result['intent'] = 'trend_analysis'
            result['granularity'] = 'monthly'
        elif any(word in query_lower for word in ['compare', 'vs', 'versus', 'difference']):
            result['intent'] = 'comparison'
        elif any(word in query_lower for word in ['total', 'sum', 'aggregate', 'overall']):
            result['intent'] = 'summary'
            result['granularity'] = 'total'
        elif any(word in query_lower for word in ['highest', 'lowest', 'maximum', 'minimum', 'peak', 'best', 'worst']):
            result['intent'] = 'trend_analysis'

        # Enhanced metric extraction
        words = re.findall(r'\b\w+\b', query_lower)
        metric_indicators = ['sales', 'revenue', 'amount', 'total', 'sum', 'count', 'volume', 'value', 'profit', 'cost']

        for word in words:
            if word in metric_indicators:
                result['custom_metric'] = word
                break

        # If no obvious metric, use first meaningful word
        if result['custom_metric'] == 'value':
            meaningful_words = [w for w in words if
                                len(w) > 3 and w not in ['show', 'give', 'tell', 'what', 'how', 'with', 'from']]
            if meaningful_words:
                result['custom_metric'] = meaningful_words[0]

        # Enhanced temporal detection
        temporal_result = self._extract_temporal_expression(query_lower)
        if temporal_result:
            result['temporal_expression'] = temporal_result

        # Enhanced granularity detection
        if 'monthly' in query_lower or 'month' in query_lower:
            result['granularity'] = 'monthly'
        elif 'quarterly' in query_lower or 'quarter' in query_lower:
            result['granularity'] = 'quarterly'
        elif 'yearly' in query_lower or 'annual' in query_lower:
            result['granularity'] = 'yearly'
        elif 'daily' in query_lower or 'day' in query_lower:
            result['granularity'] = 'daily'

        # Basic grouping detection
        grouping_words = ['by', 'per', 'across', 'category', 'type', 'region', 'department']
        for word in grouping_words:
            if word in query_lower:
                result['group_by'] = ['custom']
                # Try to find the actual grouping dimension
                words = query_lower.split()
                try:
                    idx = words.index(word)
                    if idx < len(words) - 1:
                        result['custom_group_by'] = [words[idx + 1]]
                except:
                    result['custom_group_by'] = [word]
                break

        result['parsing_time_ms'] = (time.time() - start_time) * 1000
        return result

    def _post_process_adaptive_result(self, args: Dict[str, Any], query: str,
                                      available_metrics: List[str], available_dimensions: List[str]) -> Dict[str, Any]:
        """Post-process OpenAI result with adaptive context"""
        result = {
            'intent': args.get('intent', 'summary'),
            'metric': args.get('metric', 'custom'),
            'custom_metric': args.get('custom_metric'),
            'granularity': args.get('granularity', 'total'),
            'temporal_expression': args.get('temporal_expression', {'type': 'all_time'}),
            'group_by': args.get('group_by', []),
            'custom_group_by': args.get('custom_group_by', []),
            'filters': args.get('filters', {}),
            'confidence': args.get('confidence', 0.8),
            'reasoning': args.get('reasoning', ''),
            'raw_query': query,
            'available_context': {
                'metrics': available_metrics,
                'dimensions': available_dimensions,
                'domain': self.current_profile.get('primary_domain', 'general')
            }
        }

        # Validate metric availability
        if result['metric'] not in available_metrics and result['metric'] != 'custom':
            # Try to find closest match
            closest_metric = self._find_closest_match(result['metric'], available_metrics)
            if closest_metric:
                result['metric'] = closest_metric
                result['reasoning'] += f" (mapped to closest available metric: {closest_metric})"
            else:
                result['metric'] = 'custom'
                result['custom_metric'] = args.get('metric', 'value')

        # Validate dimensions
        validated_dimensions = []
        custom_dimensions = []

        for dim in result['group_by']:
            if dim in available_dimensions:
                validated_dimensions.append(dim)
            elif dim != 'custom':
                custom_dimensions.append(dim)

        result['group_by'] = validated_dimensions
        if custom_dimensions:
            result['group_by'].append('custom')
            result['custom_group_by'].extend(custom_dimensions)

        return result

    def _find_closest_match(self, target: str, candidates: List[str], threshold: float = 0.6) -> Optional[str]:
        """Find closest matching string using simple similarity"""
        target_lower = target.lower()
        best_match = None
        best_score = 0

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Simple substring matching
            if target_lower in candidate_lower or candidate_lower in target_lower:
                score = min(len(target_lower), len(candidate_lower)) / max(len(target_lower), len(candidate_lower))
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = candidate

        return best_match

    def _create_error_result(self, query: str, error: str) -> Dict[str, Any]:
        """Create error result when all parsing fails"""
        return {
            'intent': 'summary',
            'metric': 'custom',
            'custom_metric': 'value',
            'granularity': 'total',
            'temporal_expression': {'type': 'all_time'},
            'group_by': [],
            'filters': {},
            'confidence': 0.1,
            'reasoning': f'Parsing failed: {error}',
            'method': 'error_fallback',
            'raw_query': query,
            'error': error
        }

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get summary of what the interpreter has learned"""
        if not self.current_profile:
            return {"error": "No learning data available"}

        return {
            'primary_domain': self.current_profile.get('primary_domain', 'unknown'),
            'datasets_learned': list(self.current_profile.get('datasets', {}).keys()),
            'metrics_discovered': len(self.current_profile.get('unified_metrics', [])),
            'dimensions_discovered': len(self.current_profile.get('unified_dimensions', [])),
            'vocabulary_size': len(self.current_profile.get('unified_vocabulary', [])),
            'available_metrics': self.current_profile.get('unified_metrics', [])[:20],
            'available_dimensions': self.current_profile.get('unified_dimensions', [])[:20],
            'sample_vocabulary': self.current_profile.get('unified_vocabulary', [])[:20],
            'adaptive_patterns_count': len(self.current_profile.get('adaptive_patterns', {}))
        }

    def suggest_queries(self, count: int = 10) -> List[str]:
        """Generate suggested queries based on learned data"""
        if not self.current_profile:
            return ["Please run learn_from_data() first"]

        metrics = self.current_profile.get('unified_metrics', [])
        dimensions = self.current_profile.get('unified_dimensions', [])
        domain = self.current_profile.get('primary_domain', 'general')

        suggestions = []

        # Basic patterns
        if metrics:
            suggestions.extend([
                f"Show me total {metrics[0]}",
                f"What are the monthly {metrics[0]} trends?",
                f"Compare {metrics[0]} across different categories"
            ])

        if len(metrics) > 1:
            suggestions.extend([
                f"Compare {metrics[0]} vs {metrics[1]}",
                f"Show me {metrics[1]} by {dimensions[0] if dimensions else 'category'}"
            ])

        if dimensions:
            suggestions.extend([
                f"Break down performance by {dimensions[0]}",
                f"Show me distribution across {dimensions[0]}"
            ])

        # Domain-specific suggestions
        domain_suggestions = {
            'insurance': [
                "What are our claims trends?",
                "Show me profitability by product type",
                "Analyze mortality experience"
            ],
            'banking': [
                "What are our transaction volumes?",
                "Show me account balance trends",
                "Analyze credit risk patterns"
            ],
            'ecommerce': [
                "What are our sales trends?",
                "Show me revenue by product category",
                "Analyze customer behavior patterns"
            ],
            'technology': [
                "What are our usage patterns?",
                "Show me performance metrics",
                "Analyze user engagement trends"
            ],
            'sales': [
                "What are our sales performance trends?",
                "Show me revenue by region",
                "Analyze pipeline conversion rates"
            ]
        }

        if domain in domain_suggestions:
            suggestions.extend(domain_suggestions[domain])

        return suggestions[:count]


# Integration function
def create_adaptive_query_processor(openai_api_key: Optional[str] = None,
                                    datasets: Optional[Dict[str, pd.DataFrame]] = None,
                                    cache_path: str = "cache/adaptive_profile.pkl"):
    """Create adaptive query processor that learns from your data"""

    interpreter = AdaptiveQueryInterpreter(openai_api_key)

    # Learn from provided datasets
    if datasets:
        logger.info("🧠 Initializing adaptive learning...")
        interpreter.learn_from_data(datasets, cache_path)
        logger.info("✅ Adaptive learning complete!")

    def process_adaptive_query(query: str, target_dataset: Optional[str] = None) -> Dict[str, Any]:
        """Process query with adaptive interpretation"""
        try:
            start_time = time.time()

            # Parse query
            result = interpreter.parse(query, target_dataset)

            # Validate confidence
            if result.get('confidence', 0) < 0.3:
                return {
                    'analysis_type': 'low_confidence_adaptive',
                    'summary': f'Query interpretation has very low confidence ({result.get("confidence", 0):.1%})',
                    'data': [],
                    'insights': [
                        f'Confidence: {result.get("confidence", 0):.1%}',
                        f'Method: {result.get("method", "unknown")}',
                        'Try rephrasing with specific metric and dimension names',
                        f'Available metrics: {", ".join(interpreter.current_profile.get("unified_metrics", [])[:5]) if interpreter.current_profile else "None"}'
                    ],
                    'interpretation': result,
                    'suggestions': interpreter.suggest_queries(5)
                }

            # Create successful response
            processing_time = (time.time() - start_time) * 1000

            return {
                'analysis_type': f'adaptive_{result.get("metric", "custom")}',
                'summary': f'Adaptive Analysis - {result.get("granularity", "total").title()}',
                'data': [],  # Will be populated by analytics engine
                'insights': [
                    f'Adaptive interpretation ({result.get("confidence", 0):.1%} confidence)',
                    f'Method: {result.get("method", "unknown")}',
                    f'Reasoning: {result.get("reasoning", "No reasoning provided")}',
                    f'Domain: {interpreter.current_profile.get("primary_domain", "general") if interpreter.current_profile else "general"}'
                ],
                'interpretation': result,
                'processing_metadata': {
                    'total_time_ms': processing_time,
                    'available_metrics': interpreter.current_profile.get('unified_metrics', [])[
                                         :10] if interpreter.current_profile else [],
                    'available_dimensions': interpreter.current_profile.get('unified_dimensions', [])[
                                            :10] if interpreter.current_profile else []
                }
            }

        except Exception as e:
            logger.error(f"Adaptive query processing failed: {e}")
            return {
                'analysis_type': 'adaptive_error',
                'summary': f'Adaptive processing failed: {str(e)}',
                'data': [],
                'insights': [f'Error: {str(e)}'],
                'interpretation': {'error': str(e), 'raw_query': query}
            }

    return process_adaptive_query, interpreter


# Example usage (optional - for testing only)
if __name__ == "__main__":
    # Simple test example
    sample_datasets = {
        'sales_data': pd.DataFrame({
            'sales_amount': np.random.uniform(1000, 10000, 100),
            'product_category': np.random.choice(['A', 'B', 'C'], 100),
            'region': np.random.choice(['North', 'South'], 100),
            'date': pd.date_range('2024-01-01', periods=100, freq='D')
        })
    }

    # Test the adaptive processor
    processor, interpreter = create_adaptive_query_processor(datasets=sample_datasets)

    # Test queries
    test_queries = [
        "show me total sales for 2024",
        "what are the monthly sales trends?",
        "compare sales by region",
        "show me sales breakdown by region and month"
    ]

    print("🧪 Testing Adaptive Interpreter")
    print("=" * 50)

    for query in test_queries:
        print(f"Query: '{query}'")
        result = processor(query)
        print(
            f"Result: {result['analysis_type']} (confidence: {result.get('interpretation', {}).get('confidence', 0):.1%})")
        print()