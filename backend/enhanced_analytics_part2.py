# backend/enhanced_analytics_part2.py - Intelligent Aggregator Core
from enhanced_analytics_part1 import EnhancedAnalysisResult
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import traceback
import warnings
from knowledge_framework import HybridKnowledgeFramework
from mathematical_engine import MathematicalKnowledgeEngine, AnalysisResult
import logging
import traceback
import re
from datetime import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class IntelligentAggregator:
    """Enhanced aggregator with mathematical and concept intelligence"""

    def __init__(self, unified_dataset: pd.DataFrame,
                 knowledge_framework: HybridKnowledgeFramework,
                 mathematical_engine: MathematicalKnowledgeEngine):
        self.df = unified_dataset
        self.knowledge_framework = knowledge_framework
        self.mathematical_engine = mathematical_engine

    def execute_intelligent_analysis(self, interpretation: Dict[str, Any],
                                     domain: str = None) -> EnhancedAnalysisResult:
        """Execute analysis with full mathematical and concept intelligence"""

        try:
            start_time = datetime.now()

            # Enhanced query understanding
            query = interpretation.get('raw_query', '')
            query_enhancement = self.knowledge_framework.enhance_query_with_concepts(query, domain)

            # Get method suggestions from knowledge framework
            data_characteristics = self._analyze_data_characteristics()
            method_suggestions = self.knowledge_framework.suggest_analysis_methods(
                query, data_characteristics, domain
            )

            # Resolve variables and method
            variables = self._resolve_analysis_variables(interpretation, query_enhancement)
            analysis_method = self._select_best_method(interpretation, method_suggestions, variables)

            # Execute mathematical analysis if appropriate
            mathematical_result = None
            if analysis_method.get('type') == 'mathematical':
                mathematical_result = self._execute_mathematical_analysis(
                    analysis_method, variables, interpretation
                )

            # Execute business aggregation
            business_data = self._execute_business_aggregation(interpretation, variables)

            # Generate enhanced insights (implemented in Part 3)
            insights = self._generate_intelligent_insights(
                business_data, mathematical_result, query_enhancement, domain, interpretation
            )

            # Create concept explanations (implemented in Part 3)
            concept_explanations = self._create_concept_explanations(query_enhancement, variables)

            # Calculate confidence assessment (implemented in Part 3)
            confidence_assessment = self._assess_analysis_confidence(
                mathematical_result, business_data, query_enhancement, method_suggestions
            )

            # Performance stats
            end_time = datetime.now()
            performance_stats = {
                'execution_time_ms': (end_time - start_time).total_seconds() * 1000,
                'rows_processed': len(self.df),
                'records_returned': len(business_data) if business_data else 0,
                'mathematical_analysis': mathematical_result is not None,
                'concept_enhancements': len(query_enhancement.get('found_concepts', []))
            }

            # Generate method justification (implemented in Part 3)
            method_justification = self._generate_method_justification(
                analysis_method, mathematical_result, query_enhancement
            )

            return EnhancedAnalysisResult(
                analysis_type=f"intelligent_{interpretation.get('intent', 'summary')}",
                summary=self._generate_intelligent_summary(interpretation, mathematical_result, concept_explanations),
                data=business_data,
                insights=insights,
                metadata={
                    'interpretation': interpretation,
                    'query_enhancement': query_enhancement,
                    'method_suggestions': method_suggestions,
                    'selected_method': analysis_method,
                    'variables_used': variables,
                    'domain': domain,
                    'timestamp': datetime.now().isoformat()
                },
                performance_stats=performance_stats,
                mathematical_analysis=mathematical_result,
                concept_explanations=concept_explanations,
                method_justification=method_justification,
                confidence_assessment=confidence_assessment
            )

        except Exception as e:
            logger.error(f"Intelligent analysis failed: {e}")
            logger.error(traceback.format_exc())

            # Fallback to basic analysis
            return self._create_fallback_result(interpretation, str(e))

    def _analyze_data_characteristics(self) -> Dict[str, Any]:
        """Analyze data characteristics for method selection"""
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()

            return {
                'rows': len(self.df),
                'columns': list(self.df.columns),
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'datetime_columns': datetime_cols,
                'missing_data_pct': (self.df.isnull().sum() / len(self.df)).to_dict() if len(self.df) > 0 else {},
                'has_time_series': len(datetime_cols) > 0,
                'data_types': {col: str(self.df[col].dtype) for col in self.df.columns}
            }
        except Exception as e:
            logger.error(f"Error analyzing data characteristics: {e}")
            return {
                'rows': 0,
                'columns': [],
                'numeric_columns': [],
                'categorical_columns': [],
                'datetime_columns': [],
                'missing_data_pct': {},
                'has_time_series': False,
                'data_types': {}
            }

    def _resolve_analysis_variables(self, interpretation: Dict[str, Any],
                                    query_enhancement: Dict[str, Any]) -> List[str]:
        """Intelligently resolve which variables to analyze - FIXED for sales data"""
        print(f"=== VARIABLE RESOLUTION DEBUG ===")
        print(f"Available columns: {list(self.df.columns)}")
        print(f"Interpretation: {interpretation}")

        variables = []

        try:
            # Start with interpretation variables
            metric = interpretation.get('metric')
            custom_metric = interpretation.get('custom_metric')
            group_by = interpretation.get('group_by', [])
            custom_group_by = interpretation.get('custom_group_by', [])

            print(f"Metric: {metric}, Custom metric: {custom_metric}")

            # Resolve metric
            if metric and metric != 'custom' and metric in self.df.columns:
                variables.append(metric)
                print(f"Direct metric match: {metric}")
            elif custom_metric:
                print(f"Looking for custom metric: {custom_metric}")

                # Find best matching column for custom metric
                matching_cols = [col for col in self.df.columns
                                 if custom_metric.lower() in col.lower()]
                if matching_cols:
                    variables.append(matching_cols[0])
                    print(f"Found matching column: {matching_cols[0]}")
                else:
                    # NEW: Enhanced fuzzy matching for common terms
                    metric_mappings = {
                        'sales': ['sales', 'revenue', 'amount'],
                        'revenue': ['sales', 'revenue', 'amount'],
                        'profit': ['profit', 'margin', 'earnings'],
                        'total': ['sales', 'profit', 'revenue', 'amount'],
                        'sum': ['sales', 'profit', 'revenue', 'amount'],
                        'value': ['sales', 'profit', 'revenue', 'amount', 'value']
                    }

                    metric_lower = custom_metric.lower()
                    print(f"Checking metric mappings for: {metric_lower}")

                    for term, possible_cols in metric_mappings.items():
                        if term in metric_lower:
                            print(f"Found term '{term}' in metric, checking columns: {possible_cols}")
                            for col_name in possible_cols:
                                matching_cols = [col for col in self.df.columns
                                                 if col_name.lower() == col.lower()]
                                if matching_cols:
                                    variables.append(matching_cols[0])
                                    print(f"Mapped '{custom_metric}' -> '{matching_cols[0]}'")
                                    break
                            if variables:
                                break

            # Add grouping variables (existing logic)
            all_group_vars = group_by + custom_group_by
            for dim in all_group_vars:
                if dim and dim != 'custom' and dim in self.df.columns:
                    variables.append(dim)
                elif dim and dim != 'custom':
                    # Find similar columns
                    matching_cols = [col for col in self.df.columns
                                     if dim.lower() in col.lower()]
                    if matching_cols:
                        variables.append(matching_cols[0])

            # If no variables found, use sensible defaults for sales data
            if not variables:
                print("No variables found, using defaults")
                # Look for common sales metrics
                for potential_metric in ['sales', 'profit', 'revenue', 'amount']:
                    matching_cols = [col for col in self.df.columns
                                     if potential_metric.lower() == col.lower()]
                    if matching_cols:
                        variables.append(matching_cols[0])
                        print(f"Added default metric: {matching_cols[0]}")
                        break

                # Add common grouping variables
                for potential_group in ['region', 'category', 'product', 'year', 'month']:
                    if potential_group in self.df.columns:
                        variables.append(potential_group)
                        print(f"Added default grouping: {potential_group}")

            # Remove duplicates and ensure we have valid variables
            variables = list(dict.fromkeys(variables))  # Remove duplicates while preserving order
            variables = [var for var in variables if var in self.df.columns]

            print(f"Final resolved variables: {variables}")
            return variables[:5]  # Limit to 5 variables for performance

        except Exception as e:
            logger.error(f"Error resolving analysis variables: {e}")
            print(f"Variable resolution error: {e}")
            # Fallback to first available columns
            if len(self.df.columns) > 0:
                variables = list(self.df.columns)[:3]

        return variables

    def _select_best_method(self, interpretation: Dict[str, Any],
                            method_suggestions: List[Dict[str, Any]],
                            variables: List[str]) -> Dict[str, Any]:
        """Select the best analysis method"""

        if not method_suggestions:
            return {
                'method': 'basic_aggregation',
                'type': 'business',
                'confidence': 0.6,
                'description': 'Basic business aggregation'
            }

        try:
            # Prefer mathematical methods for appropriate queries
            intent = interpretation.get('intent', 'summary')

            if intent in ['correlation', 'trend_analysis', 'comparative_analysis'] and len(variables) >= 2:
                math_methods = [m for m in method_suggestions if m.get('type') == 'mathematical']
                if math_methods:
                    return math_methods[0]

            # For statistical queries, prefer mathematical methods
            query = interpretation.get('raw_query', '').lower()
            statistical_keywords = ['correlation', 'regression', 'test', 'significance', 'distribution', 'variance']
            if any(keyword in query for keyword in statistical_keywords):
                math_methods = [m for m in method_suggestions if m.get('type') == 'mathematical']
                if math_methods:
                    return math_methods[0]

            # Otherwise, return highest confidence method
            return max(method_suggestions, key=lambda x: x.get('confidence', 0))

        except Exception as e:
            logger.error(f"Error selecting best method: {e}")
            return {
                'method': 'basic_aggregation',
                'type': 'business',
                'confidence': 0.5,
                'description': 'Fallback basic aggregation'
            }

    def _execute_mathematical_analysis(self, method: Dict[str, Any],
                                       variables: List[str],
                                       interpretation: Dict[str, Any]) -> Optional[AnalysisResult]:
        """Execute mathematical analysis using the mathematical engine"""

        try:
            # Get the mathematical method from the engine
            method_name = method.get('method', '')

            # Find the corresponding method in mathematical engine
            mathematical_method = None
            for category in self.mathematical_engine.methods_registry.values():
                if isinstance(category, dict):
                    for name, math_method in category.items():
                        if (name in method_name or
                                math_method.name in method_name or
                                method_name in math_method.name):
                            mathematical_method = math_method
                            break
                    if mathematical_method:
                        break

            if mathematical_method and len(variables) >= 1:
                # Prepare clean data for analysis
                clean_variables = [var for var in variables if var in self.df.columns]
                if not clean_variables:
                    return None

                # Filter to rows with valid data for these variables
                analysis_df = self.df[clean_variables].dropna()

                if len(analysis_df) < 3:  # Need minimum data for most analyses
                    return None

                # Execute the mathematical analysis
                result = self.mathematical_engine.execute_analysis(
                    mathematical_method, analysis_df, clean_variables
                )
                return result

        except Exception as e:
            logger.warning(f"Mathematical analysis failed: {e}")

        return None

    def _execute_business_aggregation(self, interpretation: Dict[str, Any],
                                      variables: List[str]) -> List[Dict[str, Any]]:
        """Execute business-focused aggregation"""

        try:
            # Apply filters
            filtered_df = self._apply_filters(interpretation.get('filters', {}))

            # Apply temporal filtering
            filtered_df = self._apply_temporal_filter(
                filtered_df, interpretation.get('temporal_expression', {})
            )

            # Add time grouping if needed
            granularity = interpretation.get('granularity', 'total')
            if granularity != 'total':
                filtered_df = self._add_time_grouping(filtered_df, granularity)
                if 'time_period' not in variables:
                    variables = ['time_period'] + variables

            # Ensure we have valid variables for this dataset
            valid_variables = [var for var in variables if var in filtered_df.columns]

            if not valid_variables:
                # Fallback to basic count
                return self._aggregate_total(filtered_df, 'count')

            # Execute aggregation
            group_by_vars = [var for var in valid_variables if var in filtered_df.columns and
                             (filtered_df[var].dtype == 'object' or
                              filtered_df[var].dtype.name == 'category' or
                              var == 'time_period')]
            metric_vars = [var for var in valid_variables if var in filtered_df.columns and
                           pd.api.types.is_numeric_dtype(filtered_df[var])]

            if group_by_vars and metric_vars:
                return self._aggregate_grouped(filtered_df, metric_vars[0], group_by_vars)
            elif metric_vars:
                return self._aggregate_total(filtered_df, metric_vars[0])
            elif group_by_vars:
                return self._aggregate_grouped(filtered_df, 'count', group_by_vars)
            else:
                return self._aggregate_total(filtered_df, 'count')

        except Exception as e:
            logger.error(f"Business aggregation failed: {e}")
            logger.error(traceback.format_exc())
            return [{'error': str(e), 'note': 'Business aggregation failed'}]

    def _apply_filters(self, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataset"""
        df = self.df.copy()

        try:
            for filter_key, filter_value in filters.items():
                if filter_key in df.columns and filter_value is not None:
                    if isinstance(filter_value, list):
                        df = df[df[filter_key].isin(filter_value)]
                    else:
                        df = df[df[filter_key] == filter_value]
        except Exception as e:
            logger.warning(f"Filter application failed: {e}")

        return df

    def _apply_temporal_filter(self, df: pd.DataFrame, temporal_expression: Dict[str, Any]) -> pd.DataFrame:
        """Apply temporal filtering - FIXED for integer year/month columns"""
        print(f"=== TEMPORAL FILTER DEBUG ===")
        print(f"Input data shape: {df.shape}")
        print(f"Temporal expression: {temporal_expression}")

        if not temporal_expression or temporal_expression.get('type') == 'all_time':
            print("No temporal filtering needed")
            return df

        try:
            # First check for datetime columns
            date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

            # NEW: Also check for integer year columns
            year_columns = [col for col in df.columns if
                            col.lower() in ['year', 'yr'] and
                            pd.api.types.is_integer_dtype(df[col])]

            print(f"Found date columns: {date_columns}")
            print(f"Found year columns: {year_columns}")

            if not date_columns and not year_columns:
                print("No date or year columns found")
                return df

            # Handle datetime columns (existing logic)
            if date_columns:
                date_col = date_columns[0]
                start_date = temporal_expression.get('start_date')
                end_date = temporal_expression.get('end_date')

                if start_date:
                    start_date = pd.to_datetime(start_date)
                    df = df[df[date_col] >= start_date]
                if end_date:
                    end_date = pd.to_datetime(end_date)
                    df = df[df[date_col] <= end_date]

            # NEW: Handle integer year columns
            elif year_columns:
                year_col = year_columns[0]
                print(f"Using year column: {year_col}")

                # Extract year from temporal expression
                if temporal_expression.get('type') == 'specific_period':
                    value = temporal_expression.get('value', '')

                    # Try to extract year from value or start_date
                    target_year = None
                    if value and value.isdigit() and len(value) == 4:
                        target_year = int(value)
                    elif temporal_expression.get('start_date'):
                        try:
                            start_date = pd.to_datetime(temporal_expression['start_date'])
                            target_year = start_date.year
                        except:
                            pass

                    if target_year:
                        print(f"Filtering by year: {target_year}")
                        original_years = sorted(df[year_col].unique())
                        print(f"Years in original data: {original_years}")

                        df = df[df[year_col] == target_year]

                        filtered_years = sorted(df[year_col].unique()) if len(df) > 0 else []
                        print(f"Years in filtered data: {filtered_years}")
                        print(f"Rows after year filter: {len(df)}")

        except Exception as e:
            logger.warning(f"Temporal filtering failed: {e}")
            print(f"Temporal filtering error: {e}")

        print(f"Output data shape: {df.shape}")
        return df

    def _add_time_grouping(self, df: pd.DataFrame, granularity: str) -> pd.DataFrame:
        """Add time grouping column"""
        try:
            date_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

            if not date_columns:
                df = df.copy()
                df['time_period'] = 'All Time'
                return df

            df = df.copy()
            date_col = date_columns[0]

            if granularity == 'monthly':
                df['time_period'] = df[date_col].dt.to_period('M').astype(str)
            elif granularity == 'quarterly':
                df['time_period'] = df[date_col].dt.to_period('Q').astype(str)
            elif granularity == 'yearly':
                df['time_period'] = df[date_col].dt.year.astype(str)
            elif granularity == 'weekly':
                df['time_period'] = df[date_col].dt.to_period('W').astype(str)
            elif granularity == 'daily':
                df['time_period'] = df[date_col].dt.date.astype(str)
            else:
                df['time_period'] = 'All Time'

        except Exception as e:
            logger.warning(f"Time grouping failed: {e}")
            df = df.copy()
            df['time_period'] = 'All Time'

        return df

    def _aggregate_total(self, df: pd.DataFrame, metric: str) -> List[Dict[str, Any]]:
        """Aggregate without grouping - FIXED to handle year filtering"""
        if df.empty:
            return [{'total_records': 0, 'note': 'No data available'}]

        try:
            if metric == 'count':
                return [{'total_records': len(df)}]
            elif metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                clean_data = df[metric].dropna()
                if len(clean_data) == 0:
                    return [{'total_records': len(df), 'note': f'No valid data for {metric}'}]

                result = {
                    'total_records': len(df),
                    f'{metric}_sum': float(clean_data.sum()),
                    f'{metric}_mean': float(clean_data.mean()),
                    f'{metric}_median': float(clean_data.median()),
                    f'{metric}_min': float(clean_data.min()),
                    f'{metric}_max': float(clean_data.max())
                }

                if len(clean_data) > 1:
                    result[f'{metric}_std'] = float(clean_data.std())

                # NEW: Add debug info for year filtering
                if 'year' in df.columns:
                    years_in_data = df['year'].unique()
                    result['years_included'] = sorted(years_in_data.tolist())
                    print(f"Years in filtered data: {result['years_included']}")  # Debug

                return [result]
            else:
                return [{'total_records': len(df), 'note': f'Metric {metric} not found or not numeric'}]

        except Exception as e:
            logger.error(f"Total aggregation failed: {e}")
            return [{'error': str(e), 'total_records': len(df)}]

    def _aggregate_grouped(self, df: pd.DataFrame, metric: str,
                           dimensions: List[str]) -> List[Dict[str, Any]]:
        """Aggregate with multiple grouping - ENHANCED for complex queries"""
        if df.empty:
            return [{'note': 'No data available for grouping'}]

        print(f"=== ENHANCED MULTIPLE GROUPING AGGREGATION ===")
        print(f"Metric: {metric}, Dimensions: {dimensions}")
        print(f"Available columns: {df.columns.tolist()}")
        print(f"Input data shape: {df.shape}")

        try:
            # Filter dimensions to only include existing columns
            valid_dimensions = [dim for dim in dimensions if dim in df.columns]
            print(f"Valid dimensions: {valid_dimensions}")

            if not valid_dimensions:
                print("No valid dimensions found, falling back to total aggregation")
                return self._aggregate_total(df, metric)

            # ENHANCED: Dynamic limit based on data size and dimensions
            max_dimensions = min(len(valid_dimensions), self._calculate_safe_dimension_limit(df, valid_dimensions))
            if len(valid_dimensions) > max_dimensions:
                print(f"Limiting dimensions from {len(valid_dimensions)} to {max_dimensions} for performance")
                valid_dimensions = valid_dimensions[:max_dimensions]

            # Check estimated result size
            estimated_groups = self._estimate_group_count(df, valid_dimensions)
            print(f"Estimated groups: {estimated_groups}")

            if estimated_groups > 10000:
                print("Too many groups, adding filters or limiting dimensions")
                # Take top categories by frequency for each dimension
                valid_dimensions = self._reduce_high_cardinality_dimensions(df, valid_dimensions)

            if metric == 'count':
                result = df.groupby(valid_dimensions, dropna=False).size().reset_index(name='count')
                print(f"Count aggregation: {len(result)} groups")
            elif metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
                # Enhanced aggregation functions
                agg_functions = ['sum', 'mean', 'count']

                if len(df) > 1:
                    agg_functions.extend(['min', 'max'])

                # Add std only if we have enough data
                if len(df) > 10:
                    agg_functions.append('std')

                # Perform aggregation
                grouped = df.groupby(valid_dimensions, dropna=False)[metric]
                agg_result = grouped.agg(agg_functions).reset_index()

                print(f"Numeric aggregation: {len(agg_result)} groups with functions: {agg_functions}")

                # Rename columns for clarity
                column_mapping = {}
                for func in agg_functions:
                    if func in agg_result.columns:
                        column_mapping[func] = f'{metric}_{func}'

                result = agg_result.rename(columns=column_mapping)

                # Convert numpy types to native Python types
                for col in result.select_dtypes(include=[np.number]).columns:
                    result[col] = result[col].astype(float)

            else:
                # For non-numeric metrics, just count
                result = df.groupby(valid_dimensions, dropna=False).size().reset_index(name='count')
                print(f"Count fallback: {len(result)} groups")

            # Enhanced sorting for multiple dimensions
            if len(result) > 0:
                sort_columns = []

                # Primary sort by main metric
                if f'{metric}_sum' in result.columns:
                    sort_columns.append((f'{metric}_sum', False))  # Descending
                elif 'count' in result.columns:
                    sort_columns.append(('count', False))  # Descending

                # Secondary sort by first dimension
                if len(valid_dimensions) > 0 and valid_dimensions[0] in result.columns:
                    sort_columns.append((valid_dimensions[0], True))  # Ascending

                # Apply sorting
                if sort_columns:
                    sort_by = [col[0] for col in sort_columns]
                    sort_ascending = [col[1] for col in sort_columns]
                    result = result.sort_values(sort_by, ascending=sort_ascending)
                    print(f"Sorted by: {sort_by}")

            # Smart result limiting
            result_limit = self._calculate_result_limit(len(valid_dimensions))
            if len(result) > result_limit:
                print(f"Limiting results from {len(result)} to {result_limit}")
                result = result.head(result_limit)

            # Handle NaN values
            result = result.fillna({'count': 0})
            for col in result.select_dtypes(include=[np.number]).columns:
                result[col] = result[col].fillna(0)

            result_list = result.to_dict('records')
            print(f"Final result: {len(result_list)} records")

            # Debug: Show sample results with all dimensions
            print("Sample results:")
            for i, record in enumerate(result_list[:5]):
                dim_values = {dim: record.get(dim, 'N/A') for dim in valid_dimensions}
                metric_value = record.get(f'{metric}_sum', record.get('count', 0))
                print(f"  {i + 1}. {dim_values} -> {metric}: {metric_value}")

            return result_list

        except Exception as e:
            logger.error(f"Enhanced grouping aggregation failed: {e}")
            logger.error(traceback.format_exc())
            return [{'error': f'Multiple grouping failed: {str(e)}', 'note': 'Try simplifying your query'}]

    def _calculate_safe_dimension_limit(self, df: pd.DataFrame, dimensions: List[str]) -> int:
        """Calculate safe number of dimensions based on data size"""
        data_size = len(df)

        if data_size < 1000:
            return min(4, len(dimensions))
        elif data_size < 10000:
            return min(3, len(dimensions))
        else:
            return min(2, len(dimensions))

    def _estimate_group_count(self, df: pd.DataFrame, dimensions: List[str]) -> int:
        """Estimate number of groups that would be created"""
        if not dimensions:
            return 1

        unique_counts = []
        for dim in dimensions:
            if dim in df.columns:
                unique_counts.append(df[dim].nunique())

        if not unique_counts:
            return 1

        # Estimate as product of unique values (conservative)
        import math
        estimated = math.prod(unique_counts)

        # But limit by actual data size
        return min(estimated, len(df))

    def _reduce_high_cardinality_dimensions(self, df: pd.DataFrame, dimensions: List[str]) -> List[str]:
        """Reduce high cardinality dimensions to manageable size"""
        reduced_dims = []

        for dim in dimensions:
            if dim in df.columns:
                unique_count = df[dim].nunique()
                if unique_count > 50:  # High cardinality
                    print(f"Dimension '{dim}' has {unique_count} unique values, may need filtering")
                    # Could add top-N filtering here
                reduced_dims.append(dim)

        return reduced_dims[:3]  # Max 3 dimensions for now

    def _calculate_result_limit(self, dimension_count: int) -> int:
        """Calculate appropriate result limit based on dimension count"""
        if dimension_count == 1:
            return 100
        elif dimension_count == 2:
            return 500
        elif dimension_count == 3:
            return 1000
        else:
            return 2000

    # Placeholder methods for Part 3 implementation
    def _generate_intelligent_insights(self, business_data: List[Dict[str, Any]],
                                       mathematical_result: Optional[AnalysisResult],
                                       query_enhancement: Dict[str, Any],
                                       domain: str,
                                       interpretation: Dict[str, Any]) -> List[str]:
        """Placeholder - implemented in Part 3"""
        return ["Basic insights - enhanced insights in Part 3"]

    def _create_concept_explanations(self, query_enhancement: Dict[str, Any],
                                     variables: List[str]) -> Dict[str, str]:
        """Placeholder - implemented in Part 3"""
        return {}

    def _assess_analysis_confidence(self, mathematical_result: Optional[AnalysisResult],
                                    business_data: List[Dict[str, Any]],
                                    query_enhancement: Dict[str, Any],
                                    method_suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Placeholder - implemented in Part 3"""
        return {'overall_confidence': 0.7, 'recommendation': 'medium_confidence'}

    def _generate_method_justification(self, method: Dict[str, Any],
                                       mathematical_result: Optional[AnalysisResult],
                                       query_enhancement: Dict[str, Any]) -> str:
        """Placeholder - implemented in Part 3"""
        return "Method selection based on query analysis"

    def _generate_intelligent_summary(self, interpretation: Dict[str, Any],
                                      mathematical_result: Optional[AnalysisResult],
                                      concept_explanations: Dict[str, str]) -> str:
        """Generate intelligent summary incorporating all analysis aspects"""

        try:
            # Base summary components
            intent = interpretation.get('intent', 'summary').replace('_', ' ').title()
            metric = interpretation.get('custom_metric') or interpretation.get('metric', 'data')
            granularity = interpretation.get('granularity', 'total')

            summary_parts = [f"{intent} Analysis"]

            # Add metric information
            if metric and metric != 'count':
                metric_display = metric.replace('_', ' ').title()
                if metric in concept_explanations:
                    summary_parts.append(f"of {metric_display} (domain concept)")
                else:
                    summary_parts.append(f"of {metric_display}")

            # Add granularity
            if granularity != 'total':
                summary_parts.append(f"with {granularity.title()} granularity")

            # Add mathematical analysis indicator
            if mathematical_result:
                method_name = mathematical_result.method_used.replace('_', ' ').title()
                summary_parts.append(f"including {method_name}")

            # Add concept enhancement indicator
            concept_count = len(concept_explanations)
            if concept_count > 0:
                summary_parts.append(f"enhanced with {concept_count} domain concept{'s' if concept_count > 1 else ''}")

            return " ".join(summary_parts)

        except Exception as e:
            logger.error(f"Error generating intelligent summary: {e}")
            return "Analysis Summary"

    def _create_fallback_result(self, interpretation: Dict[str, Any], error: str) -> EnhancedAnalysisResult:
        """Create fallback result when intelligent analysis fails"""
        return EnhancedAnalysisResult(
            analysis_type="fallback_analysis",
            summary=f"Basic analysis (intelligent analysis failed: {error})",
            data=[],
            insights=[f"Analysis error: {error}", "Falling back to basic aggregation"],
            metadata={'interpretation': interpretation, 'error': error},
            performance_stats={'execution_time_ms': 0, 'fallback_used': True},
            mathematical_analysis=None,
            concept_explanations={},
            method_justification="Fallback to basic analysis due to processing error",
            confidence_assessment={'overall_confidence': 0.3, 'recommendation': 'low_confidence'}
        )