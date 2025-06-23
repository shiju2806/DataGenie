# backend/enhanced_analytics_part3.py - Intelligent Analysis Engine
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import traceback
import warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from functools import lru_cache

from knowledge_framework import HybridKnowledgeFramework
from mathematical_engine import MathematicalKnowledgeEngine, AnalysisResult
from enhanced_analytics_part1 import EnhancedAnalysisResult, IntelligentDataProcessor

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class IntelligentAnalysisEngine:
    """Core analysis engine with mathematical and business intelligence"""

    def __init__(self,
                 knowledge_framework: HybridKnowledgeFramework,
                 mathematical_engine: MathematicalKnowledgeEngine,
                 data_processor: IntelligentDataProcessor):
        self.knowledge_framework = knowledge_framework
        self.mathematical_engine = mathematical_engine
        self.data_processor = data_processor
        self.analysis_cache = {}
        self.performance_stats = {}

        logger.info("🧠 Intelligent Analysis Engine initialized")

    def analyze_query(self, query: str, filters: Optional[Dict[str, Any]] = None,
                      domain: str = None) -> EnhancedAnalysisResult:
        """Main analysis method that orchestrates all intelligence"""

        start_time = datetime.now()
        analysis_id = f"analysis_{hash(query)}_{int(start_time.timestamp())}"

        try:
            # Step 1: Enhanced query understanding
            query_enhancement = self.knowledge_framework.enhance_query_with_concepts(query, domain)
            logger.info(f"🔍 Query understanding confidence: {query_enhancement.get('confidence', 0):.2f}")

            # Step 2: Prepare data with filters
            prepared_data = self._prepare_analysis_data(filters)
            if prepared_data.empty:
                return self._create_empty_result("No data available after applying filters")

            # Step 3: Determine analysis strategy
            analysis_strategy = self._determine_analysis_strategy(query, query_enhancement, prepared_data)

            # Step 4: Execute analysis based on strategy
            if analysis_strategy['type'] == 'mathematical':
                result = self._execute_mathematical_analysis(query, prepared_data, analysis_strategy, domain)
            elif analysis_strategy['type'] == 'business_aggregation':
                result = self._execute_business_analysis(query, prepared_data, analysis_strategy, domain)
            elif analysis_strategy['type'] == 'hybrid':
                result = self._execute_hybrid_analysis(query, prepared_data, analysis_strategy, domain)
            else:
                result = self._execute_exploratory_analysis(query, prepared_data, domain)

            # Step 5: Enhance result with concept knowledge
            enhanced_result = self._enhance_result_with_concepts(result, query_enhancement, domain)

            # Step 6: Add performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            enhanced_result.performance_stats.update({
                'execution_time_seconds': execution_time,
                'analysis_id': analysis_id,
                'query_confidence': query_enhancement.get('confidence', 0),
                'strategy_used': analysis_strategy['type']
            })

            logger.info(f"✅ Analysis completed in {execution_time:.2f}s")
            return enhanced_result

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            logger.error(traceback.format_exc())
            return self._create_error_result(str(e), analysis_id)

    def _prepare_analysis_data(self, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Prepare data for analysis with intelligent filtering"""
        try:
            data = self.data_processor.unified_dataset.copy()

            if filters:
                # Apply filters intelligently
                for column, filter_value in filters.items():
                    if column in data.columns:
                        if isinstance(filter_value, dict):
                            # Range filter
                            if 'min' in filter_value and 'max' in filter_value:
                                data = data[
                                    (data[column] >= filter_value['min']) &
                                    (data[column] <= filter_value['max'])
                                    ]
                            # Date range filter
                            elif 'start_date' in filter_value and 'end_date' in filter_value:
                                data[column] = pd.to_datetime(data[column], errors='coerce')
                                data = data[
                                    (data[column] >= filter_value['start_date']) &
                                    (data[column] <= filter_value['end_date'])
                                    ]
                        elif isinstance(filter_value, list):
                            # Multiple value filter
                            data = data[data[column].isin(filter_value)]
                        else:
                            # Single value filter
                            data = data[data[column] == filter_value]

            return data

        except Exception as e:
            logger.warning(f"Error preparing data: {e}")
            return self.data_processor.unified_dataset.copy()

    def _determine_analysis_strategy(self, query: str, query_enhancement: Dict[str, Any],
                                     data: pd.DataFrame) -> Dict[str, Any]:
        """Determine the best analysis strategy based on query and data"""

        query_lower = query.lower()
        mathematical_terms = query_enhancement.get('enhanced_understanding', {})

        # Mathematical analysis indicators
        math_indicators = [
            'correlation', 'regression', 'trend', 'test', 'significant',
            'variance', 'mean', 'median', 'distribution', 'predict'
        ]

        # Business aggregation indicators
        business_indicators = [
            'by', 'group', 'sum', 'total', 'count', 'average', 'breakdown',
            'segment', 'category', 'compare', 'versus', 'across'
        ]

        # Count indicators
        math_score = sum(1 for term in math_indicators if term in query_lower)
        business_score = sum(1 for term in business_indicators if term in query_lower)

        # Check for mathematical concepts in query enhancement
        has_math_concepts = any(
            concept_info.get('type') == 'mathematical'
            for concept_info in mathematical_terms.values()
        )

        # Determine strategy
        if math_score > business_score and (math_score > 0 or has_math_concepts):
            if business_score > 0:
                strategy_type = 'hybrid'
            else:
                strategy_type = 'mathematical'
        elif business_score > 0:
            strategy_type = 'business_aggregation'
        else:
            strategy_type = 'exploratory'

        return {
            'type': strategy_type,
            'math_score': math_score,
            'business_score': business_score,
            'has_math_concepts': has_math_concepts,
            'confidence': max(math_score, business_score) / 5.0  # Normalize to 0-1
        }

    def _execute_mathematical_analysis(self, query: str, data: pd.DataFrame,
                                       strategy: Dict[str, Any], domain: str = None) -> EnhancedAnalysisResult:
        """Execute mathematical/statistical analysis"""

        try:
            # Get data characteristics
            data_characteristics = self.mathematical_engine.analyze_data_characteristics(data)

            # Get method recommendations
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            variables = numeric_columns[:5]  # Limit for performance

            if not variables:
                return self._create_empty_result("No numeric variables found for mathematical analysis")

            method_recommendations = self.mathematical_engine.recommend_methods(
                query, data_characteristics, variables
            )

            if not method_recommendations:
                return self._create_empty_result("No suitable mathematical methods found")

            # Execute top method
            best_method, confidence = method_recommendations[0]
            mathematical_result = self.mathematical_engine.execute_analysis(
                best_method, data, variables
            )

            # Create visualization data based on analysis type
            viz_data = self._create_mathematical_visualization_data(
                best_method, data, variables, mathematical_result
            )

            # Generate insights
            insights = self._generate_mathematical_insights(
                mathematical_result, best_method, variables, domain
            )

            return EnhancedAnalysisResult(
                analysis_type='mathematical',
                summary=f"Mathematical analysis using {best_method.name}",
                data=viz_data,
                insights=insights,
                metadata={
                    'method_used': best_method.name,
                    'variables_analyzed': variables,
                    'sample_size': len(data),
                    'assumptions_met': mathematical_result.assumptions_met,
                    'method_confidence': confidence
                },
                performance_stats={},
                mathematical_analysis=mathematical_result,
                method_justification=f"Selected {best_method.name} based on query analysis and data characteristics"
            )

        except Exception as e:
            logger.error(f"Mathematical analysis failed: {e}")
            return self._create_error_result(f"Mathematical analysis failed: {str(e)}")

    def _execute_business_analysis(self, query: str, data: pd.DataFrame,
                                   strategy: Dict[str, Any], domain: str = None) -> EnhancedAnalysisResult:
        """Execute business aggregation analysis"""

        try:
            # Identify grouping variables from query
            group_by_cols = self._identify_grouping_variables(query, data)

            # Identify metrics to analyze
            metric_cols = self._identify_metric_variables(query, data)

            if not metric_cols:
                metric_cols = data.select_dtypes(include=[np.number]).columns.tolist()[:3]

            # Perform aggregation
            if group_by_cols:
                aggregated_data = self._perform_intelligent_aggregation(
                    data, group_by_cols, metric_cols
                )
            else:
                # Overall summary if no grouping identified
                aggregated_data = self._create_summary_statistics(data, metric_cols)

            # Create visualization data
            viz_data = self._format_business_analysis_data(aggregated_data, group_by_cols, metric_cols)

            # Generate business insights
            insights = self._generate_business_insights(
                aggregated_data, group_by_cols, metric_cols, domain
            )

            return EnhancedAnalysisResult(
                analysis_type='business_aggregation',
                summary=f"Business analysis grouped by {', '.join(group_by_cols) if group_by_cols else 'overall summary'}",
                data=viz_data,
                insights=insights,
                metadata={
                    'group_by_columns': group_by_cols,
                    'metric_columns': metric_cols,
                    'aggregation_method': 'intelligent_grouping',
                    'total_groups': len(aggregated_data) if isinstance(aggregated_data, pd.DataFrame) else 1
                },
                performance_stats={}
            )

        except Exception as e:
            logger.error(f"Business analysis failed: {e}")
            return self._create_error_result(f"Business analysis failed: {str(e)}")

    def _execute_hybrid_analysis(self, query: str, data: pd.DataFrame,
                                 strategy: Dict[str, Any], domain: str = None) -> EnhancedAnalysisResult:
        """Execute hybrid mathematical + business analysis"""

        try:
            # Execute both types of analysis
            math_result = self._execute_mathematical_analysis(query, data, strategy, domain)
            business_result = self._execute_business_analysis(query, data, strategy, domain)

            # Combine results intelligently
            combined_data = []

            # Add mathematical results
            if math_result.data:
                for item in math_result.data:
                    item['analysis_type'] = 'mathematical'
                    combined_data.append(item)

            # Add business results
            if business_result.data:
                for item in business_result.data:
                    item['analysis_type'] = 'business'
                    combined_data.append(item)

            # Combine insights
            combined_insights = []
            combined_insights.extend(math_result.insights)
            combined_insights.extend(business_result.insights)

            # Add hybrid-specific insights
            combined_insights.append(
                "This hybrid analysis combines statistical methods with business aggregation for comprehensive insights"
            )

            return EnhancedAnalysisResult(
                analysis_type='hybrid',
                summary="Hybrid mathematical and business analysis",
                data=combined_data,
                insights=combined_insights,
                metadata={
                    'mathematical_method': math_result.metadata.get('method_used'),
                    'business_grouping': business_result.metadata.get('group_by_columns'),
                    'hybrid_confidence': (
                                                 math_result.metadata.get('method_confidence', 0) +
                                                 strategy.get('confidence', 0)
                                         ) / 2
                },
                performance_stats={},
                mathematical_analysis=math_result.mathematical_analysis
            )

        except Exception as e:
            logger.error(f"Hybrid analysis failed: {e}")
            return self._create_error_result(f"Hybrid analysis failed: {str(e)}")

    def _execute_exploratory_analysis(self, query: str, data: pd.DataFrame,
                                      domain: str = None) -> EnhancedAnalysisResult:
        """Execute exploratory data analysis when strategy is unclear"""

        try:
            # Basic data exploration
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns

            exploration_data = []
            insights = []

            # Summary statistics for numeric columns
            if len(numeric_cols) > 0:
                summary_stats = data[numeric_cols].describe()

                for col in numeric_cols[:5]:  # Limit for performance
                    col_stats = summary_stats[col]
                    exploration_data.append({
                        'type': 'summary_statistic',
                        'variable': col,
                        'mean': col_stats['mean'],
                        'median': data[col].median(),
                        'std': col_stats['std'],
                        'min': col_stats['min'],
                        'max': col_stats['max'],
                        'missing_count': data[col].isnull().sum()
                    })

                insights.append(f"Dataset contains {len(numeric_cols)} numeric variables")

            # Value counts for categorical columns
            if len(categorical_cols) > 0:
                for col in categorical_cols[:3]:  # Limit for performance
                    value_counts = data[col].value_counts().head(10)

                    for value, count in value_counts.items():
                        exploration_data.append({
                            'type': 'value_count',
                            'variable': col,
                            'value': str(value),
                            'count': int(count),
                            'percentage': round(count / len(data) * 100, 1)
                        })

                insights.append(f"Dataset contains {len(categorical_cols)} categorical variables")

            # Data quality insights
            total_missing = data.isnull().sum().sum()
            total_cells = len(data) * len(data.columns)
            missing_percentage = (total_missing / total_cells) * 100

            insights.append(f"Data quality: {missing_percentage:.1f}% missing values")
            insights.append(f"Dataset shape: {len(data)} rows × {len(data.columns)} columns")

            return EnhancedAnalysisResult(
                analysis_type='exploratory',
                summary="Exploratory data analysis",
                data=exploration_data,
                insights=insights,
                metadata={
                    'numeric_columns': len(numeric_cols),
                    'categorical_columns': len(categorical_cols),
                    'missing_percentage': missing_percentage,
                    'analysis_scope': 'data_exploration'
                },
                performance_stats={}
            )

        except Exception as e:
            logger.error(f"Exploratory analysis failed: {e}")
            return self._create_error_result(f"Exploratory analysis failed: {str(e)}")

    def _identify_grouping_variables(self, query: str, data: pd.DataFrame) -> List[str]:
        """Identify variables to group by from query"""

        group_by_indicators = ['by', 'group by', 'per', 'for each', 'across']
        query_lower = query.lower()

        potential_groups = []

        # Look for explicit grouping keywords
        for indicator in group_by_indicators:
            if indicator in query_lower:
                # Extract words after the indicator
                words_after = query_lower.split(indicator)[-1].split()[:3]
                potential_groups.extend(words_after)

        # Match with actual columns
        group_cols = []
        for potential in potential_groups:
            # Direct match
            matching_cols = [col for col in data.columns if potential in col.lower()]
            if matching_cols:
                group_cols.extend(matching_cols[:1])  # Take first match

        # If no explicit grouping found, suggest categorical columns
        if not group_cols:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                # Prefer columns with reasonable number of unique values
                for col in categorical_cols:
                    unique_count = data[col].nunique()
                    if 2 <= unique_count <= 20:  # Reasonable grouping size
                        group_cols.append(col)
                        break

        return group_cols[:2]  # Limit to 2 grouping variables for performance

    def _identify_metric_variables(self, query: str, data: pd.DataFrame) -> List[str]:
        """Identify metric variables from query"""

        metric_keywords = [
            'revenue', 'cost', 'profit', 'amount', 'value', 'price', 'total',
            'sum', 'count', 'average', 'mean', 'sales', 'volume', 'quantity',
            'rate', 'ratio', 'percentage', 'score', 'premium', 'claims'
        ]

        query_lower = query.lower()
        metric_cols = []

        # Find columns mentioned in query
        for keyword in metric_keywords:
            if keyword in query_lower:
                matching_cols = [col for col in data.columns if keyword in col.lower()]
                metric_cols.extend(matching_cols)

        # Remove duplicates while preserving order
        metric_cols = list(dict.fromkeys(metric_cols))

        return metric_cols[:5]  # Limit for performance

    def _perform_intelligent_aggregation(self, data: pd.DataFrame,
                                         group_cols: List[str],
                                         metric_cols: List[str]) -> pd.DataFrame:
        """Perform intelligent aggregation based on data types"""

        try:
            # Determine appropriate aggregation methods
            agg_methods = {}

            for col in metric_cols:
                if col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        agg_methods[col] = ['sum', 'mean', 'count']
                    else:
                        agg_methods[col] = ['count']

            # Perform groupby aggregation
            if agg_methods:
                grouped = data.groupby(group_cols).agg(agg_methods).reset_index()

                # Flatten column names
                new_cols = []
                for col in grouped.columns:
                    if isinstance(col, tuple):
                        new_cols.append(f"{col[0]}_{col[1]}")
                    else:
                        new_cols.append(col)
                grouped.columns = new_cols

                return grouped
            else:
                # Fallback to simple value counts
                return data[group_cols].value_counts().reset_index()

        except Exception as e:
            logger.warning(f"Aggregation failed: {e}")
            return data[group_cols].value_counts().reset_index()

    def _create_summary_statistics(self, data: pd.DataFrame,
                                   metric_cols: List[str]) -> Dict[str, Any]:
        """Create summary statistics when no grouping is identified"""

        summary = {}

        for col in metric_cols:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                summary[col] = {
                    'total': data[col].sum(),
                    'average': data[col].mean(),
                    'median': data[col].median(),
                    'count': data[col].count(),
                    'std': data[col].std()
                }

        return summary

    def _create_mathematical_visualization_data(self, method, data: pd.DataFrame,
                                                variables: List[str],
                                                result: AnalysisResult) -> List[Dict[str, Any]]:
        """Create visualization data for mathematical analysis"""

        viz_data = []

        try:
            if 'correlation' in method.name and len(variables) >= 2:
                # Correlation scatter plot data
                x_col, y_col = variables[0], variables[1]
                sample_data = data[[x_col, y_col]].dropna().sample(min(1000, len(data)))

                for _, row in sample_data.iterrows():
                    viz_data.append({
                        'type': 'scatter',
                        'x': float(row[x_col]),
                        'y': float(row[y_col]),
                        'x_label': x_col,
                        'y_label': y_col
                    })

            elif 'regression' in method.name:
                # Regression line data
                if result.results.get('predictions'):
                    predictions = result.results['predictions'][:100]  # Limit data points
                    actual_values = data[variables[-1]].dropna().head(100).tolist()

                    for i, (pred, actual) in enumerate(zip(predictions, actual_values)):
                        viz_data.append({
                            'type': 'regression',
                            'index': i,
                            'predicted': float(pred),
                            'actual': float(actual),
                            'residual': float(actual - pred)
                        })

            else:
                # Generic statistical summary
                for var in variables:
                    if var in data.columns:
                        viz_data.append({
                            'type': 'summary',
                            'variable': var,
                            'mean': float(data[var].mean()),
                            'median': float(data[var].median()),
                            'std': float(data[var].std()),
                            'count': int(data[var].count())
                        })

        except Exception as e:
            logger.warning(f"Error creating mathematical visualization data: {e}")

        return viz_data

    def _format_business_analysis_data(self, aggregated_data, group_cols: List[str],
                                       metric_cols: List[str]) -> List[Dict[str, Any]]:
        """Format business analysis data for visualization"""

        viz_data = []

        try:
            if isinstance(aggregated_data, pd.DataFrame):
                # Convert aggregated DataFrame to list of dicts
                for _, row in aggregated_data.head(100).iterrows():  # Limit rows
                    data_point = {'type': 'aggregated'}

                    for col in row.index:
                        if pd.notna(row[col]):
                            if isinstance(row[col], (int, float)):
                                data_point[col] = float(row[col])
                            else:
                                data_point[col] = str(row[col])

                    viz_data.append(data_point)

            elif isinstance(aggregated_data, dict):
                # Convert summary statistics to visualization format
                for metric, stats in aggregated_data.items():
                    viz_data.append({
                        'type': 'summary',
                        'metric': metric,
                        **{k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in stats.items()}
                    })

        except Exception as e:
            logger.warning(f"Error formatting business analysis data: {e}")

        return viz_data

    def _generate_mathematical_insights(self, result: AnalysisResult, method,
                                        variables: List[str], domain: str = None) -> List[str]:
        """Generate insights from mathematical analysis"""

        insights = []

        try:
            # Add method-specific insights
            insights.append(result.interpretation)

            # Add confidence assessment
            confidence_level = "high" if result.confidence > 0.8 else "medium" if result.confidence > 0.5 else "low"
            insights.append(f"Analysis confidence: {confidence_level} ({result.confidence:.2f})")

            # Add assumption warnings
            violated_assumptions = [
                assumption for assumption, met in result.assumptions_met.items()
                if not met
            ]

            if violated_assumptions:
                insights.append(f"⚠️ Assumption violations detected: {', '.join(violated_assumptions)}")

            # Add domain-specific insights
            if domain == 'insurance' and 'correlation' in method.name:
                insights.append("Consider regulatory requirements when interpreting correlation results")
            elif domain == 'banking' and 'regression' in method.name:
                insights.append("Model results should be validated against regulatory stress testing")

            # Add general recommendations
            insights.extend(result.recommendations)

        except Exception as e:
            logger.warning(f"Error generating mathematical insights: {e}")
            insights.append("Analysis completed successfully")

        return insights

    def _generate_business_insights(self, aggregated_data, group_cols: List[str],
                                    metric_cols: List[str], domain: str = None) -> List[str]:
        """Generate insights from business analysis"""

        insights = []

        try:
            if isinstance(aggregated_data, pd.DataFrame) and not aggregated_data.empty:
                # Find top performers
                numeric_cols = aggregated_data.select_dtypes(include=[np.number]).columns

                if len(numeric_cols) > 0:
                    top_col = numeric_cols[0]
                    top_performer = aggregated_data.loc[aggregated_data[top_col].idxmax()]

                    insights.append(
                        f"Top performer: {top_performer[group_cols[0]]} with {top_col}: {top_performer[top_col]:,.2f}")

                # Add variability insights
                if len(aggregated_data) > 1:
                    for col in numeric_cols[:2]:
                        cv = aggregated_data[col].std() / aggregated_data[col].mean()
                        variability = "high" if cv > 0.5 else "medium" if cv > 0.2 else "low"
                        insights.append(f"{col} shows {variability} variability across groups")

            # Add domain-specific insights
            if domain == 'insurance':
                insights.append("Consider seasonality and risk factors in insurance metrics")
            elif domain == 'banking':
                insights.append("Monitor regulatory capital requirements across segments")
            elif domain == 'technology':
                insights.append("Track user engagement and system performance metrics")

            # Add general business insights
            insights.append(f"Analysis covers {len(group_cols)} dimensions and {len(metric_cols)} metrics")

        except Exception as e:
            logger.warning(f"Error generating business insights: {e}")
            insights.append("Business analysis completed successfully")

        return insights

    def _enhance_result_with_concepts(self, result: EnhancedAnalysisResult,
                                      query_enhancement: Dict[str, Any],
                                      domain: str = None) -> EnhancedAnalysisResult:
        """Enhance analysis result with concept explanations"""

        try:
            concept_explanations = {}
            enhanced_understanding = query_enhancement.get('enhanced_understanding', {})

            # Add concept explanations
            for term, concept_info in enhanced_understanding.items():
                if concept_info.get('definition'):
                    concept_explanations[term] = concept_info['definition']

            # Add domain context
            if domain:
                concept_explanations['domain_context'] = f"Analysis performed in {domain} domain"

            # Update result
            result.concept_explanations = concept_explanations

            # Add confidence assessment
            overall_confidence = self._calculate_overall_confidence(result, query_enhancement)
            result.confidence_assessment = overall_confidence

        except Exception as e:
            logger.warning(f"Error enhancing result with concepts: {e}")

        return result

    def _calculate_overall_confidence(self, result: EnhancedAnalysisResult,
                                      query_enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall confidence in analysis results"""

        confidence_factors = []

        # Query understanding confidence
        query_confidence = query_enhancement.get('confidence', 0.5)
        confidence_factors.append(('query_understanding', query_confidence))

        # Mathematical confidence (if applicable)
        if result.mathematical_analysis:
            math_confidence = result.mathematical_analysis.confidence
            confidence_factors.append(('mathematical_analysis', math_confidence))

        # Data quality confidence
        data_quality = result.metadata.get('data_quality_score', 0.7)
        confidence_factors.append(('data_quality', data_quality))

        # Method confidence
        method_confidence = result.metadata.get('method_confidence', 0.7)
        confidence_factors.append(('method_selection', method_confidence))

        # Calculate weighted average
        total_weight = len(confidence_factors)
        overall_confidence = sum(conf for _, conf in confidence_factors) / total_weight

        # Determine recommendation
        if overall_confidence > 0.8:
            recommendation = "high_confidence"
        elif overall_confidence > 0.6:
            recommendation = "medium_confidence"
        else:
            recommendation = "low_confidence"

        return {
            'overall_confidence': overall_confidence,
            'confidence_factors': dict(confidence_factors),
            'recommendation': recommendation,
            'explanation': self._get_confidence_explanation(recommendation)
        }

    def _get_confidence_explanation(self, recommendation: str) -> str:
        """Get explanation for confidence level"""

        explanations = {
            'high_confidence': "Results are highly reliable with strong statistical support and clear data quality",
            'medium_confidence': "Results are reasonably reliable but may have some limitations or assumptions",
            'low_confidence': "Results should be interpreted with caution due to data quality or methodological concerns"
        }

        return explanations.get(recommendation, "Confidence level assessment unavailable")

    def _create_empty_result(self, message: str) -> EnhancedAnalysisResult:
        """Create empty result with informative message"""

        return EnhancedAnalysisResult(
            analysis_type='empty',
            summary=message,
            data=[],
            insights=[message, "Try adjusting filters or check data availability"],
            metadata={'status': 'no_data'},
            performance_stats={'execution_time_seconds': 0},
            confidence_assessment={'overall_confidence': 0.0, 'recommendation': 'no_data'}
        )

    def _create_error_result(self, error_message: str, analysis_id: str = None) -> EnhancedAnalysisResult:
        """Create error result with diagnostic information"""

        return EnhancedAnalysisResult(
            analysis_type='error',
            summary=f"Analysis failed: {error_message}",
            data=[],
            insights=[
                "Analysis encountered an error",
                "Please check your query and data format",
                "Contact support if the issue persists"
            ],
            metadata={
                'status': 'error',
                'error_message': error_message,
                'analysis_id': analysis_id
            },
            performance_stats={'execution_time_seconds': 0},
            confidence_assessment={'overall_confidence': 0.0, 'recommendation': 'error'}
        )

    @lru_cache(maxsize=100)
    def get_analysis_suggestions(self, query: str, domain: str = None) -> List[Dict[str, Any]]:
        """Get analysis method suggestions for a query"""

        try:
            # Use knowledge framework to get suggestions
            mock_data_characteristics = {
                'rows': 1000,
                'columns': ['metric1', 'metric2', 'category'],
                'numeric_columns': ['metric1', 'metric2'],
                'categorical_columns': ['category']
            }

            suggestions = self.knowledge_framework.suggest_analysis_methods(
                query, mock_data_characteristics, domain
            )

            return suggestions

        except Exception as e:
            logger.error(f"Error getting analysis suggestions: {e}")
            return []

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the analysis engine"""

        return {
            'cache_size': len(self.analysis_cache),
            'mathematical_methods_available': len(self.mathematical_engine.methods_registry),
            'knowledge_concepts': self.knowledge_framework.get_knowledge_summary(),
            'data_processor_stats': self.data_processor.get_processing_summary()
        }

    def clear_cache(self):
        """Clear analysis cache"""
        self.analysis_cache.clear()
        logger.info("Analysis cache cleared")

    def get_analysis_cache_summary(self) -> Dict[str, Any]:
        """Get summary of cached analyses"""
        return {
            'total_cached_analyses': len(self.analysis_cache),
            'cache_keys': list(self.analysis_cache.keys())[:10],  # Show first 10 keys
            'memory_usage_estimate': len(str(self.analysis_cache)) / 1024  # Rough estimate in KB
        }


class BusinessAggregationEngine:
    """Enhanced business aggregation with intelligent grouping and metrics"""

    def __init__(self, knowledge_framework: HybridKnowledgeFramework):
        self.knowledge_framework = knowledge_framework
        self.aggregation_cache = {}

        logger.info("📊 Business Aggregation Engine initialized")

    def aggregate_data(self, data: pd.DataFrame, query: str,
                       filters: Optional[Dict[str, Any]] = None) -> EnhancedAnalysisResult:
        """Perform intelligent business aggregation based on query"""

        try:
            # Apply filters first
            if filters:
                data = self._apply_filters(data, filters)

            # Identify grouping variables and metrics from query
            group_by_cols = self._identify_grouping_variables(query, data)
            metric_cols = self._identify_metric_variables(query, data)

            # Perform aggregation
            if group_by_cols:
                aggregated_data = self._perform_aggregation(data, group_by_cols, metric_cols)
                summary = f"Business analysis grouped by {', '.join(group_by_cols)}"
            else:
                aggregated_data = self._create_summary_analysis(data, metric_cols)
                summary = "Overall business summary analysis"

            # Convert to visualization format
            viz_data = self._format_aggregation_results(aggregated_data, group_by_cols, metric_cols)

            # Generate insights
            insights = self._generate_business_insights(aggregated_data, group_by_cols, metric_cols, data)

            return EnhancedAnalysisResult(
                analysis_type='business_aggregation',
                summary=summary,
                data=viz_data,
                insights=insights,
                metadata={
                    'group_by_columns': group_by_cols,
                    'metric_columns': metric_cols,
                    'total_groups': len(aggregated_data) if isinstance(aggregated_data, pd.DataFrame) else 1,
                    'sample_size': len(data),
                    'aggregation_method': 'intelligent_grouping'
                },
                performance_stats={'execution_time_seconds': 0}  # Will be updated by caller
            )

        except Exception as e:
            logger.error(f"Business aggregation failed: {e}")
            return self._create_error_result(f"Business aggregation failed: {str(e)}")

    def _apply_filters(self, data: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply user-specified filters to data"""

        filtered_data = data.copy()

        for column, filter_value in filters.items():
            if column in filtered_data.columns:
                try:
                    if isinstance(filter_value, dict):
                        # Range filter
                        if 'min' in filter_value and 'max' in filter_value:
                            filtered_data = filtered_data[
                                (filtered_data[column] >= filter_value['min']) &
                                (filtered_data[column] <= filter_value['max'])
                                ]
                        # Date range filter
                        elif 'start_date' in filter_value and 'end_date' in filter_value:
                            filtered_data[column] = pd.to_datetime(filtered_data[column], errors='coerce')
                            filtered_data = filtered_data[
                                (filtered_data[column] >= filter_value['start_date']) &
                                (filtered_data[column] <= filter_value['end_date'])
                                ]
                    elif isinstance(filter_value, list):
                        # Multiple value filter
                        filtered_data = filtered_data[filtered_data[column].isin(filter_value)]
                    else:
                        # Single value filter
                        filtered_data = filtered_data[filtered_data[column] == filter_value]
                except Exception as e:
                    logger.warning(f"Failed to apply filter for {column}: {e}")

        return filtered_data

    def _identify_grouping_variables(self, query: str, data: pd.DataFrame) -> List[str]:
        """Identify variables to group by from query"""

        group_by_indicators = ['by', 'group by', 'per', 'for each', 'across', 'segment']
        query_lower = query.lower()

        potential_groups = []

        # Look for explicit grouping keywords
        for indicator in group_by_indicators:
            if indicator in query_lower:
                # Extract words after the indicator
                words_after = query_lower.split(indicator)[-1].split()[:3]
                potential_groups.extend(words_after)

        # Match with actual columns
        group_cols = []
        for potential in potential_groups:
            # Direct match
            matching_cols = [col for col in data.columns if potential in col.lower()]
            if matching_cols:
                group_cols.extend(matching_cols[:1])  # Take first match

        # If no explicit grouping found, suggest categorical columns
        if not group_cols:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                # Prefer columns with reasonable number of unique values
                for col in categorical_cols:
                    unique_count = data[col].nunique()
                    if 2 <= unique_count <= 20:  # Reasonable grouping size
                        group_cols.append(col)
                        break

        return group_cols[:2]  # Limit to 2 grouping variables for performance

    def _identify_metric_variables(self, query: str, data: pd.DataFrame) -> List[str]:
        """Identify metric variables from query"""

        metric_keywords = [
            'revenue', 'cost', 'profit', 'amount', 'value', 'price', 'total',
            'sum', 'count', 'average', 'mean', 'sales', 'volume', 'quantity',
            'rate', 'ratio', 'percentage', 'score', 'premium', 'claims'
        ]

        query_lower = query.lower()
        metric_cols = []

        # Find columns mentioned in query
        for keyword in metric_keywords:
            if keyword in query_lower:
                matching_cols = [col for col in data.columns if keyword in col.lower()]
                metric_cols.extend(matching_cols)

        # If no specific metrics found, use numeric columns
        if not metric_cols:
            metric_cols = list(data.select_dtypes(include=[np.number]).columns)

        # Remove duplicates while preserving order
        metric_cols = list(dict.fromkeys(metric_cols))

        return metric_cols[:5]  # Limit for performance

    def _perform_aggregation(self, data: pd.DataFrame, group_cols: List[str],
                             metric_cols: List[str]) -> pd.DataFrame:
        """Perform intelligent aggregation"""

        try:
            # Determine appropriate aggregation methods for each metric
            agg_dict = {}

            for col in metric_cols:
                if col in data.columns:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        agg_dict[col] = ['sum', 'mean', 'count', 'std']
                    else:
                        agg_dict[col] = ['count', 'nunique']

            # Perform groupby aggregation
            if agg_dict:
                grouped = data.groupby(group_cols).agg(agg_dict)

                # Flatten column names
                new_cols = []
                for col in grouped.columns:
                    if isinstance(col, tuple):
                        new_cols.append(f"{col[0]}_{col[1]}")
                    else:
                        new_cols.append(col)
                grouped.columns = new_cols

                # Reset index to make group columns regular columns
                grouped = grouped.reset_index()

                # Add calculated metrics
                grouped = self._add_calculated_metrics(grouped, group_cols)

                return grouped
            else:
                # Fallback to simple value counts
                return data[group_cols].value_counts().reset_index()

        except Exception as e:
            logger.warning(f"Aggregation failed: {e}")
            return data[group_cols].value_counts().reset_index()

    def _add_calculated_metrics(self, aggregated_data: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
        """Add calculated metrics to aggregated data"""

        try:
            # Add percentage of total for sum columns
            sum_cols = [col for col in aggregated_data.columns if col.endswith('_sum')]
            for col in sum_cols:
                if col in aggregated_data.columns:
                    total = aggregated_data[col].sum()
                    if total > 0:
                        pct_col = col.replace('_sum', '_pct_of_total')
                        aggregated_data[pct_col] = (aggregated_data[col] / total * 100).round(2)

            # Add growth rates if there's a time component
            if any('date' in col.lower() or 'time' in col.lower() for col in group_cols):
                # Sort by time column and calculate period-over-period growth
                time_cols = [col for col in group_cols if 'date' in col.lower() or 'time' in col.lower()]
                if time_cols:
                    time_col = time_cols[0]
                    aggregated_data = aggregated_data.sort_values(time_col)

                    for col in sum_cols:
                        if col in aggregated_data.columns:
                            growth_col = col.replace('_sum', '_growth_rate')
                            aggregated_data[growth_col] = aggregated_data[col].pct_change() * 100

            return aggregated_data

        except Exception as e:
            logger.warning(f"Failed to add calculated metrics: {e}")
            return aggregated_data

    def _create_summary_analysis(self, data: pd.DataFrame, metric_cols: List[str]) -> Dict[str, Any]:
        """Create summary analysis when no grouping is specified"""

        summary = {
            'analysis_type': 'overall_summary',
            'total_records': len(data),
            'metrics': {}
        }

        for col in metric_cols:
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                summary['metrics'][col] = {
                    'total': float(data[col].sum()),
                    'average': float(data[col].mean()),
                    'median': float(data[col].median()),
                    'std': float(data[col].std()),
                    'min': float(data[col].min()),
                    'max': float(data[col].max()),
                    'count': int(data[col].count())
                }

        return summary

    def _format_aggregation_results(self, aggregated_data: Union[pd.DataFrame, Dict],
                                    group_cols: List[str], metric_cols: List[str]) -> List[Dict[str, Any]]:
        """Format aggregation results for visualization"""

        viz_data = []

        try:
            if isinstance(aggregated_data, pd.DataFrame):
                # Convert DataFrame to list of dictionaries
                for _, row in aggregated_data.head(100).iterrows():  # Limit to 100 rows
                    row_dict = {'type': 'aggregated'}

                    for col in row.index:
                        value = row[col]
                        if pd.notna(value):
                            if isinstance(value, (int, float)):
                                row_dict[col] = float(value)
                            else:
                                row_dict[col] = str(value)

                    viz_data.append(row_dict)

            elif isinstance(aggregated_data, dict):
                # Convert summary dictionary to visualization format
                viz_data.append({
                    'type': 'summary',
                    'analysis_type': aggregated_data.get('analysis_type', 'summary'),
                    'total_records': aggregated_data.get('total_records', 0),
                    'metrics': aggregated_data.get('metrics', {})
                })

        except Exception as e:
            logger.warning(f"Error formatting aggregation results: {e}")

        return viz_data

    def _generate_business_insights(self, aggregated_data: Union[pd.DataFrame, Dict],
                                    group_cols: List[str], metric_cols: List[str],
                                    original_data: pd.DataFrame) -> List[str]:
        """Generate business insights from aggregated data"""

        insights = []

        try:
            if isinstance(aggregated_data, pd.DataFrame) and not aggregated_data.empty:
                # Find top performers
                numeric_cols = aggregated_data.select_dtypes(include=[np.number]).columns

                for metric in numeric_cols[:3]:  # Analyze top 3 numeric columns
                    if metric in aggregated_data.columns:
                        # Top performer insight
                        max_idx = aggregated_data[metric].idxmax()
                        max_row = aggregated_data.loc[max_idx]

                        group_description = ', '.join(
                            [f"{col}: {max_row[col]}" for col in group_cols if col in max_row.index])
                        insights.append(f"Top performer in {metric}: {group_description} ({max_row[metric]:,.2f})")

                        # Performance distribution insight
                        if len(aggregated_data) > 1:
                            mean_val = aggregated_data[metric].mean()
                            std_val = aggregated_data[metric].std()
                            cv = std_val / mean_val if mean_val != 0 else 0

                            variability = "high" if cv > 0.5 else "moderate" if cv > 0.2 else "low"
                            insights.append(f"{metric} shows {variability} variability across groups (CV: {cv:.2f})")

                        # Concentration insight
                        if len(aggregated_data) > 2:
                            total = aggregated_data[metric].sum()
                            top_3_sum = aggregated_data[metric].nlargest(3).sum()
                            concentration = (top_3_sum / total) * 100 if total > 0 else 0

                            if concentration > 70:
                                insights.append(
                                    f"High concentration: Top 3 groups account for {concentration:.1f}% of {metric}")

            elif isinstance(aggregated_data, dict):
                # Summary insights
                total_records = aggregated_data.get('total_records', 0)
                metrics = aggregated_data.get('metrics', {})

                insights.append(f"Overall analysis covers {total_records:,} records")

                for metric, stats in metrics.items():
                    total_val = stats.get('total', 0)
                    avg_val = stats.get('average', 0)
                    insights.append(f"{metric}: Total = {total_val:,.2f}, Average = {avg_val:,.2f}")

            # Data quality insights
            missing_pct = (original_data.isnull().sum().sum() / (len(original_data) * len(original_data.columns))) * 100
            if missing_pct > 10:
                insights.append(
                    f"Data quality note: {missing_pct:.1f}% missing values may affect analysis completeness")

            # Sample size insights
            if len(original_data) < 100:
                insights.append(
                    f"Small sample size ({len(original_data)} records) - consider collecting additional data")

        except Exception as e:
            logger.warning(f"Error generating business insights: {e}")
            insights.append("Business analysis completed successfully")

        return insights

    def _create_error_result(self, error_message: str) -> EnhancedAnalysisResult:
        """Create error result for business aggregation"""

        return EnhancedAnalysisResult(
            analysis_type='error',
            summary=f"Business aggregation failed: {error_message}",
            data=[],
            insights=[
                "Business aggregation encountered an error",
                "Please check data format and query structure",
                "Ensure sufficient data is available for grouping"
            ],
            metadata={'status': 'error', 'error_message': error_message},
            performance_stats={'execution_time_seconds': 0}
        )