# backend/enhanced_analytics_part1.py - Core Classes and Data Processor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from dataclasses import dataclass
import traceback
import warnings
from knowledge_framework import HybridKnowledgeFramework
from mathematical_engine import MathematicalKnowledgeEngine, AnalysisResult

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class EnhancedAnalysisResult:
    """Enhanced analysis result with concept knowledge"""
    analysis_type: str
    summary: str
    data: List[Dict[str, Any]]
    insights: List[str]
    metadata: Dict[str, Any]
    performance_stats: Dict[str, Any]
    mathematical_analysis: Optional[AnalysisResult] = None
    concept_explanations: Optional[Dict[str, str]] = None
    method_justification: str = ""
    confidence_assessment: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure all fields are properly initialized"""
        if self.concept_explanations is None:
            self.concept_explanations = {}
        if self.confidence_assessment is None:
            self.confidence_assessment = {'overall_confidence': 0.5, 'recommendation': 'medium_confidence'}


class IntelligentDataProcessor:
    """Enhanced data processor with concept-driven transformations"""

    def __init__(self, datasets: Dict[str, pd.DataFrame],
                 knowledge_framework: HybridKnowledgeFramework):
        self.datasets = datasets if datasets else {}
        self.knowledge_framework = knowledge_framework
        self.unified_dataset = pd.DataFrame()
        self.transformation_log = []
        self.enhanced_columns = {}

        # Enhanced data processing with concept awareness
        self._create_intelligent_unified_dataset()

    def _create_intelligent_unified_dataset(self):
        """Create unified dataset with concept-driven enhancements"""
        try:
            if not self.datasets:
                logger.warning("No datasets provided - creating empty unified dataset")
                self.unified_dataset = pd.DataFrame()
                return

            # Start with largest dataset as base
            primary_dataset_name = max(self.datasets.items(), key=lambda x: len(x[1]))[0]
            self.unified_dataset = self.datasets[primary_dataset_name].copy()

            logger.info(f"📊 Starting with primary dataset: {primary_dataset_name}")

            # Intelligent column enhancement based on concepts
            self._enhance_columns_with_concepts()

            # Smart joins with other datasets
            self._perform_intelligent_joins()

            # Add concept-driven derived fields
            self._add_concept_driven_fields()

            # Clean and validate the dataset
            self._clean_and_validate_dataset()

            logger.info(
                f"✅ Enhanced unified dataset: {len(self.unified_dataset)} rows × {len(self.unified_dataset.columns)} columns")

        except Exception as e:
            logger.error(f"Error creating intelligent unified dataset: {e}")
            logger.error(traceback.format_exc())
            # Fallback to simple approach
            if self.datasets:
                largest_dataset = max(self.datasets.items(), key=lambda x: len(x[1]))[0]
                self.unified_dataset = self.datasets[largest_dataset].copy()
            else:
                self.unified_dataset = pd.DataFrame()

    def _enhance_columns_with_concepts(self):
        """Enhance column understanding using concept knowledge"""
        enhanced_columns = {}

        for col in self.unified_dataset.columns:
            try:
                # Find concept for this column
                concept = self.knowledge_framework.find_concept(col, use_apis=False)

                if concept:
                    enhanced_columns[col] = {
                        'original_name': col,
                        'concept': concept.name,
                        'definition': concept.definition,
                        'synonyms': concept.synonyms,
                        'calculation_method': concept.calculation_method
                    }

                    # Log the enhancement
                    self.transformation_log.append({
                        'type': 'concept_enhancement',
                        'column': col,
                        'concept_found': concept.name,
                        'confidence': concept.confidence
                    })
            except Exception as e:
                logger.warning(f"Failed to enhance column {col}: {e}")

        # Store enhanced column information
        self.enhanced_columns = enhanced_columns

    def _perform_intelligent_joins(self):
        """Perform intelligent joins based on concept relationships"""
        if len(self.datasets) <= 1:
            return

        primary_dataset_name = max(self.datasets.items(), key=lambda x: len(x[1]))[0]

        for dataset_name, dataset in self.datasets.items():
            if dataset_name != primary_dataset_name and not dataset.empty:
                try:
                    # Find potential join columns based on concepts
                    join_candidates = self._find_concept_based_join_keys(
                        self.unified_dataset, dataset
                    )

                    if join_candidates:
                        best_join = join_candidates[0]

                        # Perform the join
                        joined_data = self.unified_dataset.merge(
                            dataset,
                            left_on=best_join['left_key'],
                            right_on=best_join['right_key'],
                            how='left',
                            suffixes=('', f'_{dataset_name}')
                        )

                        # Only keep the join if it adds meaningful data
                        if len(joined_data.columns) > len(self.unified_dataset.columns):
                            original_rows = len(self.unified_dataset)
                            self.unified_dataset = joined_data

                            self.transformation_log.append({
                                'type': 'intelligent_join',
                                'dataset': dataset_name,
                                'join_key': best_join['left_key'],
                                'concept_confidence': best_join.get('confidence', 0.8),
                                'rows_before': original_rows,
                                'rows_after': len(joined_data),
                                'columns_added': len(joined_data.columns) - len(self.unified_dataset.columns)
                            })

                            logger.info(f"🔗 Joined {dataset_name} on concept-based key: {best_join['left_key']}")

                except Exception as e:
                    logger.warning(f"Failed to join {dataset_name}: {e}")

    def _find_concept_based_join_keys(self, left_df: pd.DataFrame,
                                      right_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Find join keys based on concept similarity"""
        candidates = []

        # Limit to reasonable number of columns for performance
        left_cols = list(left_df.columns)[:20]
        right_cols = list(right_df.columns)[:20]

        for left_col in left_cols:
            for right_col in right_cols:
                try:
                    # Direct name match
                    if left_col.lower() == right_col.lower():
                        overlap = self._calculate_data_overlap(left_df[left_col], right_df[right_col])
                        if overlap > 0.05:  # At least 5% overlap
                            candidates.append({
                                'left_key': left_col,
                                'right_key': right_col,
                                'left_concept': left_col,
                                'right_concept': right_col,
                                'overlap': overlap,
                                'confidence': 0.9
                            })
                            continue

                    # Concept-based matching
                    left_concept = self.knowledge_framework.find_concept(left_col, use_apis=False)
                    right_concept = self.knowledge_framework.find_concept(right_col, use_apis=False)

                    if left_concept and right_concept:
                        if (left_concept.name == right_concept.name or
                                left_col.lower() in [syn.lower() for syn in right_concept.synonyms] or
                                right_col.lower() in [syn.lower() for syn in left_concept.synonyms]):

                            # Validate data compatibility
                            overlap = self._calculate_data_overlap(left_df[left_col], right_df[right_col])

                            if overlap > 0.05:  # At least 5% overlap
                                candidates.append({
                                    'left_key': left_col,
                                    'right_key': right_col,
                                    'left_concept': left_concept.name,
                                    'right_concept': right_concept.name,
                                    'overlap': overlap,
                                    'confidence': min(left_concept.confidence, right_concept.confidence)
                                })
                except Exception as e:
                    logger.warning(f"Error evaluating join candidate {left_col}-{right_col}: {e}")
                    continue

        # Sort by overlap and confidence
        candidates.sort(key=lambda x: (x['overlap'], x['confidence']), reverse=True)
        return candidates[:3]  # Return top 3 candidates

    def _calculate_data_overlap(self, left_series: pd.Series, right_series: pd.Series) -> float:
        """Calculate overlap between two data series"""
        try:
            left_values = set(left_series.dropna().astype(str).str.lower())
            right_values = set(right_series.dropna().astype(str).str.lower())

            if not left_values or not right_values:
                return 0.0

            intersection = len(left_values & right_values)
            union = len(left_values | right_values)

            return intersection / union if union > 0 else 0.0
        except Exception as e:
            logger.warning(f"Error calculating data overlap: {e}")
            return 0.0

    def _add_concept_driven_fields(self):
        """Add derived fields based on concept knowledge and calculation methods"""

        try:
            # Look for concepts with calculation methods
            for col in self.unified_dataset.columns:
                concept = self.knowledge_framework.find_concept(col, use_apis=False)

                if concept and concept.calculation_method:
                    try:
                        # Attempt to create calculated fields based on concept definitions
                        self._apply_concept_calculations(concept, col)
                    except Exception as e:
                        logger.warning(f"Failed to apply calculation for {concept.name}: {e}")

            # Add time-based derived fields intelligently
            self._add_intelligent_time_fields()

            # Add ratio and percentage fields based on concepts
            self._add_concept_based_ratios()

            # Add statistical derived fields
            self._add_statistical_fields()

        except Exception as e:
            logger.error(f"Error adding concept-driven fields: {e}")

    def _apply_concept_calculations(self, concept, base_column: str):
        """Apply calculations defined in concept knowledge"""
        try:
            calc_method = concept.calculation_method.lower() if concept.calculation_method else ""

            # Insurance-specific calculations
            if concept.name == 'ae_ratio':
                actual_cols = [col for col in self.unified_dataset.columns if 'actual' in col.lower()]
                expected_cols = [col for col in self.unified_dataset.columns if 'expected' in col.lower()]

                if actual_cols and expected_cols:
                    actual_col, expected_col = actual_cols[0], expected_cols[0]
                    self.unified_dataset['ae_ratio_calculated'] = (
                            self.unified_dataset[actual_col] /
                            self.unified_dataset[expected_col].clip(lower=0.001)
                    )

                    self.transformation_log.append({
                        'type': 'concept_calculation',
                        'concept': concept.name,
                        'formula': f'{actual_col} / {expected_col}',
                        'new_column': 'ae_ratio_calculated'
                    })

            elif concept.name == 'lapse_rate':
                lapse_cols = [col for col in self.unified_dataset.columns if 'lapse' in col.lower()]
                exposure_cols = [col for col in self.unified_dataset.columns if 'exposure' in col.lower()]

                if lapse_cols and exposure_cols:
                    lapse_col, exposure_col = lapse_cols[0], exposure_cols[0]
                    self.unified_dataset['lapse_rate_calculated'] = (
                            self.unified_dataset[lapse_col] /
                            self.unified_dataset[exposure_col].clip(lower=1)
                    )

                    self.transformation_log.append({
                        'type': 'concept_calculation',
                        'concept': concept.name,
                        'formula': f'{lapse_col} / {exposure_col}',
                        'new_column': 'lapse_rate_calculated'
                    })

            # Banking-specific calculations
            elif concept.name == 'default_rate':
                default_cols = [col for col in self.unified_dataset.columns if 'default' in col.lower()]
                loan_cols = [col for col in self.unified_dataset.columns if
                             any(term in col.lower() for term in ['loan', 'total', 'account'])]

                if default_cols and loan_cols:
                    default_col, loan_col = default_cols[0], loan_cols[0]
                    self.unified_dataset['default_rate_calculated'] = (
                            self.unified_dataset[default_col] /
                            self.unified_dataset[loan_col].clip(lower=1)
                    )

            # Universal financial calculations
            elif concept.name == 'profit':
                revenue_cols = [col for col in self.unified_dataset.columns if 'revenue' in col.lower()]
                cost_cols = [col for col in self.unified_dataset.columns if 'cost' in col.lower()]

                if revenue_cols and cost_cols:
                    revenue_col = revenue_cols[0]
                    total_costs = self.unified_dataset[cost_cols].sum(axis=1)
                    self.unified_dataset['profit_calculated'] = (
                            self.unified_dataset[revenue_col] - total_costs
                    )

                    self.transformation_log.append({
                        'type': 'concept_calculation',
                        'concept': concept.name,
                        'formula': f'{revenue_col} - sum({cost_cols})',
                        'new_column': 'profit_calculated'
                    })

        except Exception as e:
            logger.warning(f"Error applying concept calculation for {concept.name}: {e}")

    def _add_intelligent_time_fields(self):
        """Add time-based fields with intelligent detection"""
        try:
            date_columns = []

            # Find date columns using multiple methods
            for col in self.unified_dataset.columns:
                if (pd.api.types.is_datetime64_any_dtype(self.unified_dataset[col]) or
                        any(term in col.lower() for term in ['date', 'time', 'created', 'updated', 'issued'])):
                    try:
                        # Try to convert to datetime
                        converted_series = pd.to_datetime(self.unified_dataset[col], errors='coerce')
                        if converted_series.notna().sum() > len(
                                self.unified_dataset) * 0.5:  # More than 50% valid dates
                            self.unified_dataset[col] = converted_series
                            date_columns.append(col)
                    except:
                        continue

            # Add derived time fields for each date column
            current_time = datetime.now()
            for date_col in date_columns:
                try:
                    base_name = date_col.replace('_date', '').replace('_time', '').replace('date_', '').replace('time_',
                                                                                                                '')

                    # Time since calculations
                    self.unified_dataset[f'{base_name}_days_since'] = (
                            current_time - self.unified_dataset[date_col]
                    ).dt.days

                    self.unified_dataset[f'{base_name}_years_since'] = (
                            self.unified_dataset[f'{base_name}_days_since'] / 365.25
                    ).round(2)

                    # Time period groupings
                    self.unified_dataset[f'{base_name}_year'] = self.unified_dataset[date_col].dt.year
                    self.unified_dataset[f'{base_name}_quarter'] = self.unified_dataset[date_col].dt.to_period(
                        'Q').astype(str)
                    self.unified_dataset[f'{base_name}_month'] = self.unified_dataset[date_col].dt.to_period(
                        'M').astype(str)
                    self.unified_dataset[f'{base_name}_day_of_week'] = self.unified_dataset[date_col].dt.day_name()

                    # Age/duration categories
                    days_since = self.unified_dataset[f'{base_name}_days_since']
                    self.unified_dataset[f'{base_name}_age_category'] = pd.cut(
                        days_since,
                        bins=[0, 30, 90, 365, 730, float('inf')],
                        labels=['New (0-30d)', 'Recent (1-3m)', 'Mature (3-12m)', 'Old (1-2y)', 'Very Old (2y+)'],
                        include_lowest=True
                    )

                    self.transformation_log.append({
                        'type': 'time_enhancement',
                        'base_column': date_col,
                        'derived_fields': [f'{base_name}_days_since', f'{base_name}_years_since',
                                           f'{base_name}_year', f'{base_name}_quarter', f'{base_name}_month',
                                           f'{base_name}_day_of_week', f'{base_name}_age_category']
                    })

                except Exception as e:
                    logger.warning(f"Failed to create time fields for {date_col}: {e}")

        except Exception as e:
            logger.error(f"Error adding intelligent time fields: {e}")

    def _add_concept_based_ratios(self):
        """Add ratio fields based on concept relationships"""
        try:
            numeric_cols = self.unified_dataset.select_dtypes(include=[np.number]).columns

            # Look for concept-driven ratio opportunities
            ratio_patterns = [
                (['actual', 'expected'], 'ae_ratio'),
                (['revenue', 'cost'], 'profit_margin'),
                (['claims', 'premium'], 'loss_ratio'),
                (['defaults', 'loans'], 'default_rate'),
                (['success', 'attempts'], 'success_rate'),
                (['sales', 'inventory'], 'turnover_ratio'),
                (['profit', 'revenue'], 'profit_margin_pct'),
                (['current_assets', 'current_liabilities'], 'current_ratio')
            ]

            for pattern_terms, ratio_name in ratio_patterns:
                try:
                    numerator_cols = [col for col in numeric_cols
                                      if any(term in col.lower() for term in pattern_terms[0:1])]
                    denominator_cols = [col for col in numeric_cols
                                        if any(term in col.lower() for term in pattern_terms[1:2])]

                    for num_col in numerator_cols[:2]:  # Limit to avoid too many combinations
                        for den_col in denominator_cols[:2]:
                            if num_col != den_col:
                                try:
                                    ratio_col_name = f'{ratio_name}_{num_col.split("_")[0]}_{den_col.split("_")[0]}'
                                    self.unified_dataset[ratio_col_name] = (
                                            self.unified_dataset[num_col] /
                                            self.unified_dataset[den_col].clip(lower=0.001)
                                    )

                                    self.transformation_log.append({
                                        'type': 'concept_ratio',
                                        'ratio_name': ratio_name,
                                        'numerator': num_col,
                                        'denominator': den_col,
                                        'new_column': ratio_col_name
                                    })

                                except Exception as e:
                                    logger.warning(f"Failed to create ratio {ratio_name}: {e}")

                except Exception as e:
                    logger.warning(f"Error processing ratio pattern {ratio_name}: {e}")

        except Exception as e:
            logger.error(f"Error adding concept-based ratios: {e}")

    def _add_statistical_fields(self):
        """Add statistical derived fields for numeric data"""
        try:
            numeric_cols = self.unified_dataset.select_dtypes(include=[np.number]).columns

            # Add percentile ranks for key metrics
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                try:
                    if self.unified_dataset[col].notna().sum() > 10:  # Enough data for percentiles
                        self.unified_dataset[f'{col}_percentile'] = (
                                self.unified_dataset[col].rank(pct=True) * 100
                        ).round(1)

                        # Add quartile categories
                        self.unified_dataset[f'{col}_quartile'] = pd.qcut(
                            self.unified_dataset[col],
                            q=4,
                            labels=['Q1 (Bottom 25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (Top 25%)'],
                            duplicates='drop'
                        )

                except Exception as e:
                    logger.warning(f"Failed to create statistical fields for {col}: {e}")

        except Exception as e:
            logger.error(f"Error adding statistical fields: {e}")

    def _clean_and_validate_dataset(self):
        """Clean and validate the unified dataset"""
        try:
            # Remove duplicate columns
            self.unified_dataset = self.unified_dataset.loc[:, ~self.unified_dataset.columns.duplicated()]

            # Handle infinite values
            numeric_cols = self.unified_dataset.select_dtypes(include=[np.number]).columns
            self.unified_dataset[numeric_cols] = self.unified_dataset[numeric_cols].replace([np.inf, -np.inf], np.nan)

            # Log data quality metrics
            total_cells = len(self.unified_dataset) * len(self.unified_dataset.columns)
            missing_cells = self.unified_dataset.isnull().sum().sum()
            data_quality = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

            self.transformation_log.append({
                'type': 'data_quality_assessment',
                'total_rows': len(self.unified_dataset),
                'total_columns': len(self.unified_dataset.columns),
                'data_quality_score': data_quality,
                'missing_data_percentage': (missing_cells / total_cells) * 100 if total_cells > 0 else 0
            })

            logger.info(f"📊 Data quality score: {data_quality:.1%}")

        except Exception as e:
            logger.error(f"Error in data cleaning and validation: {e}")

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of data processing operations"""
        try:
            return {
                'datasets_processed': len(self.datasets),
                'unified_dataset_shape': [len(self.unified_dataset), len(self.unified_dataset.columns)],
                'transformations_applied': len(self.transformation_log),
                'transformation_types': list(set(log.get('type', 'unknown') for log in self.transformation_log)),
                'enhanced_columns': len(self.enhanced_columns),
                'processing_log': self.transformation_log
            }
        except Exception as e:
            logger.error(f"Error getting processing summary: {e}")
            return {
                'error': str(e),
                'datasets_processed': 0,
                'unified_dataset_shape': [0, 0]
            }