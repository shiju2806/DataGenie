# backend/enhanced_analytics_part4a.py - Advanced Insight Generation
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import traceback
import warnings
from dataclasses import dataclass, field
import json
import re
from collections import defaultdict

from knowledge_framework import HybridKnowledgeFramework
from mathematical_engine import MathematicalKnowledgeEngine, AnalysisResult
from enhanced_analytics_part1 import EnhancedAnalysisResult, IntelligentDataProcessor
from enhanced_analytics_part3 import IntelligentAnalysisEngine

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class InsightPattern:
    """Pattern for generating insights"""
    name: str
    pattern_type: str
    conditions: Dict[str, Any]
    insight_template: str
    severity: str  # 'high', 'medium', 'low'
    confidence_threshold: float
    domain_specific: bool = False


class AdvancedInsightGenerator:
    """Generates advanced insights using pattern recognition and domain knowledge"""

    def __init__(self, knowledge_framework: HybridKnowledgeFramework):
        self.knowledge_framework = knowledge_framework
        self.insight_patterns = []
        self.generated_insights = []

        self._initialize_insight_patterns()
        logger.info("🔍 Advanced Insight Generator initialized")

    def _initialize_insight_patterns(self):
        """Initialize patterns for insight generation"""

        # Statistical anomaly patterns
        self.insight_patterns.extend([
            InsightPattern(
                name="high_correlation_alert",
                pattern_type="statistical",
                conditions={"correlation_threshold": 0.7, "significance": 0.05},
                insight_template="Strong {direction} correlation detected between {var1} and {var2} (r={correlation:.3f}). This suggests {interpretation}.",
                severity="high",
                confidence_threshold=0.8
            ),
            InsightPattern(
                name="outlier_detection",
                pattern_type="statistical",
                conditions={"outlier_threshold": 3.0},
                insight_template="Significant outliers detected in {variable}. {outlier_count} values exceed {threshold} standard deviations from the mean.",
                severity="medium",
                confidence_threshold=0.7
            ),
            InsightPattern(
                name="distribution_skewness",
                pattern_type="statistical",
                conditions={"skewness_threshold": 1.0},
                insight_template="{variable} shows {skew_type} skewness ({skewness:.2f}), indicating {interpretation}.",
                severity="medium",
                confidence_threshold=0.6
            ),
            InsightPattern(
                name="variance_instability",
                pattern_type="statistical",
                conditions={"cv_threshold": 0.5},
                insight_template="{variable} exhibits high variability (CV={cv:.2f}), suggesting {interpretation}.",
                severity="medium",
                confidence_threshold=0.7
            )
        ])

        # Business performance patterns
        self.insight_patterns.extend([
            InsightPattern(
                name="performance_trend",
                pattern_type="business",
                conditions={"trend_significance": 0.05, "min_periods": 3},
                insight_template="{metric} shows a {trend_direction} trend over {period}. {trend_description}.",
                severity="high",
                confidence_threshold=0.8
            ),
            InsightPattern(
                name="segment_performance_gap",
                pattern_type="business",
                conditions={"performance_gap_threshold": 0.2},
                insight_template="Performance gap identified: {top_segment} outperforms {bottom_segment} by {gap_percentage:.1f}% in {metric}.",
                severity="high",
                confidence_threshold=0.9
            ),
            InsightPattern(
                name="concentration_risk",
                pattern_type="business",
                conditions={"concentration_threshold": 0.8},
                insight_template="High concentration detected: Top {top_n} categories account for {concentration:.1f}% of {metric}.",
                severity="medium",
                confidence_threshold=0.8
            ),
            InsightPattern(
                name="seasonal_pattern",
                pattern_type="business",
                conditions={"seasonality_strength": 0.3},
                insight_template="Seasonal pattern detected in {metric}: {pattern_description}.",
                severity="medium",
                confidence_threshold=0.7
            )
        ])

        # Domain-specific patterns
        self._add_insurance_patterns()
        self._add_banking_patterns()
        self._add_technology_patterns()

    def _add_insurance_patterns(self):
        """Add insurance-specific insight patterns"""

        insurance_patterns = [
            InsightPattern(
                name="mortality_experience_variance",
                pattern_type="insurance",
                conditions={"ae_ratio_threshold": [0.8, 1.2]},
                insight_template="Mortality experience shows {variance_type} with A/E ratio of {ae_ratio:.3f}. {regulatory_implication}.",
                severity="high",
                confidence_threshold=0.9,
                domain_specific=True
            ),
            InsightPattern(
                name="lapse_rate_alert",
                pattern_type="insurance",
                conditions={"lapse_rate_threshold": 0.15},
                insight_template="Elevated lapse rate detected: {lapse_rate:.2f}% in {segment}. Consider retention strategies.",
                severity="high",
                confidence_threshold=0.8,
                domain_specific=True
            ),
            InsightPattern(
                name="reserve_adequacy_concern",
                pattern_type="insurance",
                conditions={"reserve_ratio_threshold": 0.95},
                insight_template="Reserve adequacy concern: {reserve_type} reserves at {adequacy:.1f}% of required level.",
                severity="high",
                confidence_threshold=0.9,
                domain_specific=True
            ),
            InsightPattern(
                name="claim_frequency_spike",
                pattern_type="insurance",
                conditions={"frequency_increase_threshold": 0.2},
                insight_template="Claim frequency spike detected: {increase:.1f}% increase in {time_period}.",
                severity="high",
                confidence_threshold=0.8,
                domain_specific=True
            ),
            InsightPattern(
                name="premium_adequacy_warning",
                pattern_type="insurance",
                conditions={"loss_ratio_threshold": 0.85},
                insight_template="Premium adequacy warning: Loss ratio at {loss_ratio:.2f}%, approaching break-even.",
                severity="medium",
                confidence_threshold=0.8,
                domain_specific=True
            )
        ]

        self.insight_patterns.extend(insurance_patterns)

    def _add_banking_patterns(self):
        """Add banking-specific insight patterns"""

        banking_patterns = [
            InsightPattern(
                name="credit_risk_concentration",
                pattern_type="banking",
                conditions={"default_rate_threshold": 0.05},
                insight_template="Credit risk concentration in {segment}: Default rate of {default_rate:.2f}% exceeds threshold.",
                severity="high",
                confidence_threshold=0.9,
                domain_specific=True
            ),
            InsightPattern(
                name="capital_adequacy_warning",
                pattern_type="banking",
                conditions={"capital_ratio_threshold": 0.08},
                insight_template="Capital adequacy concern: {capital_type} ratio at {ratio:.2f}%, approaching regulatory minimum.",
                severity="high",
                confidence_threshold=0.95,
                domain_specific=True
            ),
            InsightPattern(
                name="loan_portfolio_diversification",
                pattern_type="banking",
                conditions={"concentration_limit": 0.25},
                insight_template="Portfolio concentration: {loan_type} represents {concentration:.1f}% of total portfolio.",
                severity="medium",
                confidence_threshold=0.8,
                domain_specific=True
            ),
            InsightPattern(
                name="liquidity_stress_indicator",
                pattern_type="banking",
                conditions={"liquidity_ratio_threshold": 0.1},
                insight_template="Liquidity stress indicator: {liquidity_metric} at {ratio:.2f}%, requires monitoring.",
                severity="high",
                confidence_threshold=0.9,
                domain_specific=True
            ),
            InsightPattern(
                name="interest_rate_sensitivity",
                pattern_type="banking",
                conditions={"duration_gap_threshold": 2.0},
                insight_template="Interest rate sensitivity: Duration gap of {gap:.1f} years indicates {risk_direction} exposure.",
                severity="medium",
                confidence_threshold=0.8,
                domain_specific=True
            )
        ]

        self.insight_patterns.extend(banking_patterns)

    def _add_technology_patterns(self):
        """Add technology-specific insight patterns"""

        tech_patterns = [
            InsightPattern(
                name="performance_degradation",
                pattern_type="technology",
                conditions={"latency_threshold": 500, "uptime_threshold": 0.99},
                insight_template="Performance degradation detected: {metric} at {value}{unit}, below SLA requirement.",
                severity="high",
                confidence_threshold=0.9,
                domain_specific=True
            ),
            InsightPattern(
                name="user_engagement_drop",
                pattern_type="technology",
                conditions={"engagement_drop_threshold": 0.15},
                insight_template="User engagement decline: {metric} decreased by {drop_percentage:.1f}% over {period}.",
                severity="medium",
                confidence_threshold=0.8,
                domain_specific=True
            ),
            InsightPattern(
                name="scaling_bottleneck",
                pattern_type="technology",
                conditions={"utilization_threshold": 0.85},
                insight_template="Scaling bottleneck identified: {resource} utilization at {utilization:.1f}%.",
                severity="medium",
                confidence_threshold=0.8,
                domain_specific=True
            ),
            InsightPattern(
                name="error_rate_spike",
                pattern_type="technology",
                conditions={"error_rate_threshold": 0.05},
                insight_template="Error rate spike detected: {error_rate:.2f}% error rate in {system_component}.",
                severity="high",
                confidence_threshold=0.9,
                domain_specific=True
            ),
            InsightPattern(
                name="conversion_funnel_drop",
                pattern_type="technology",
                conditions={"conversion_drop_threshold": 0.1},
                insight_template="Conversion funnel drop: {drop:.1f}% decrease in {funnel_stage} conversion rate.",
                severity="medium",
                confidence_threshold=0.8,
                domain_specific=True
            )
        ]

        self.insight_patterns.extend(tech_patterns)

    def generate_insights(self, analysis_result: EnhancedAnalysisResult,
                          data: pd.DataFrame, domain: str = None) -> List[str]:
        """Generate advanced insights from analysis results"""

        insights = []

        try:
            # Start with existing insights
            insights.extend(analysis_result.insights)

            # Generate pattern-based insights
            pattern_insights = self._generate_pattern_insights(analysis_result, data, domain)
            insights.extend(pattern_insights)

            # Generate comparative insights
            comparative_insights = self._generate_comparative_insights(analysis_result, data)
            insights.extend(comparative_insights)

            # Generate predictive insights
            predictive_insights = self._generate_predictive_insights(analysis_result, data, domain)
            insights.extend(predictive_insights)

            # Generate actionable recommendations
            recommendations = self._generate_actionable_recommendations(analysis_result, data, domain)
            insights.extend(recommendations)

            # Generate quality and confidence insights
            quality_insights = self._generate_quality_insights(analysis_result, data)
            insights.extend(quality_insights)

            # Store generated insights for learning
            self.generated_insights.extend(insights)

            # Remove duplicates while preserving order
            unique_insights = []
            seen = set()
            for insight in insights:
                if insight not in seen:
                    unique_insights.append(insight)
                    seen.add(insight)

            logger.info(f"🔍 Generated {len(unique_insights)} unique insights")
            return unique_insights

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return analysis_result.insights

    def _generate_pattern_insights(self, analysis_result: EnhancedAnalysisResult,
                                   data: pd.DataFrame, domain: str = None) -> List[str]:
        """Generate insights based on predefined patterns"""

        insights = []

        try:
            for pattern in self.insight_patterns:
                # Skip domain-specific patterns if domain doesn't match
                if pattern.domain_specific and pattern.pattern_type != domain:
                    continue

                # Check if pattern applies to current analysis
                if self._pattern_applies(pattern, analysis_result, data):
                    insight = self._generate_pattern_insight(pattern, analysis_result, data)
                    if insight:
                        insights.append(f"🔍 {insight}")

        except Exception as e:
            logger.warning(f"Error in pattern insight generation: {e}")

        return insights

    def _pattern_applies(self, pattern: InsightPattern,
                         analysis_result: EnhancedAnalysisResult,
                         data: pd.DataFrame) -> bool:
        """Check if a pattern applies to the current analysis"""

        try:
            # Statistical patterns
            if pattern.pattern_type == "statistical":
                if pattern.name == "high_correlation_alert":
                    math_result = analysis_result.mathematical_analysis
                    if math_result and 'correlation' in math_result.method_used:
                        correlation = math_result.results.get('statistic', 0)
                        return abs(correlation) >= pattern.conditions['correlation_threshold']

                elif pattern.name == "outlier_detection":
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        for col in numeric_cols[:3]:  # Check first 3 numeric columns
                            if data[col].notna().sum() > 3:  # Need minimum data
                                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                                outliers = z_scores > pattern.conditions['outlier_threshold']
                                if outliers.sum() > 0:
                                    return True

                elif pattern.name == "distribution_skewness":
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols[:3]:
                        if data[col].notna().sum() > 3:
                            skewness = data[col].skew()
                            if abs(skewness) >= pattern.conditions['skewness_threshold']:
                                return True

                elif pattern.name == "variance_instability":
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols[:3]:
                        if data[col].notna().sum() > 3:
                            cv = data[col].std() / data[col].mean() if data[col].mean() != 0 else 0
                            if cv >= pattern.conditions['cv_threshold']:
                                return True

            # Business patterns
            elif pattern.pattern_type == "business":
                if pattern.name == "segment_performance_gap":
                    # Check if we have grouped data with performance metrics
                    if len(analysis_result.data) > 1:
                        numeric_fields = [k for k in analysis_result.data[0].keys()
                                          if isinstance(analysis_result.data[0].get(k), (int, float))]
                        return len(numeric_fields) > 0

                elif pattern.name == "concentration_risk":
                    # Check for concentration in categorical data
                    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
                    for col in categorical_cols[:2]:
                        value_counts = data[col].value_counts(normalize=True)
                        if len(value_counts) > 0 and value_counts.iloc[0] >= pattern.conditions[
                            'concentration_threshold']:
                            return True

            # Domain-specific patterns
            elif pattern.domain_specific:
                return self._check_domain_pattern(pattern, analysis_result, data)

        except Exception as e:
            logger.warning(f"Error checking pattern applicability for {pattern.name}: {e}")

        return False

    def _check_domain_pattern(self, pattern: InsightPattern,
                              analysis_result: EnhancedAnalysisResult,
                              data: pd.DataFrame) -> bool:
        """Check domain-specific pattern applicability"""

        try:
            if pattern.pattern_type == "insurance":
                if pattern.name == "mortality_experience_variance":
                    # Check for A/E ratio columns
                    ae_cols = [col for col in data.columns if 'ae_ratio' in col.lower() or 'a/e' in col.lower()]
                    return len(ae_cols) > 0

                elif pattern.name == "lapse_rate_alert":
                    lapse_cols = [col for col in data.columns if 'lapse' in col.lower()]
                    if lapse_cols:
                        lapse_values = data[lapse_cols[0]].dropna()
                        return len(lapse_values) > 0 and lapse_values.mean() > pattern.conditions[
                            'lapse_rate_threshold']

                elif pattern.name == "claim_frequency_spike":
                    claim_cols = [col for col in data.columns if 'claim' in col.lower() and 'freq' in col.lower()]
                    return len(claim_cols) > 0

                elif pattern.name == "premium_adequacy_warning":
                    loss_ratio_cols = [col for col in data.columns if 'loss' in col.lower() and 'ratio' in col.lower()]
                    return len(loss_ratio_cols) > 0

            elif pattern.pattern_type == "banking":
                if pattern.name == "credit_risk_concentration":
                    default_cols = [col for col in data.columns if 'default' in col.lower()]
                    if default_cols:
                        default_values = data[default_cols[0]].dropna()
                        return len(default_values) > 0 and default_values.mean() > pattern.conditions[
                            'default_rate_threshold']

                elif pattern.name == "capital_adequacy_warning":
                    capital_cols = [col for col in data.columns if 'capital' in col.lower()]
                    return len(capital_cols) > 0

                elif pattern.name == "liquidity_stress_indicator":
                    liquidity_cols = [col for col in data.columns if 'liquidity' in col.lower()]
                    return len(liquidity_cols) > 0

            elif pattern.pattern_type == "technology":
                if pattern.name == "performance_degradation":
                    perf_cols = [col for col in data.columns
                                 if any(term in col.lower() for term in ['latency', 'response', 'uptime'])]
                    return len(perf_cols) > 0

                elif pattern.name == "error_rate_spike":
                    error_cols = [col for col in data.columns if 'error' in col.lower()]
                    return len(error_cols) > 0

                elif pattern.name == "user_engagement_drop":
                    engagement_cols = [col for col in data.columns
                                       if any(term in col.lower() for term in ['engagement', 'active', 'session'])]
                    return len(engagement_cols) > 0

        except Exception as e:
            logger.warning(f"Error checking domain pattern {pattern.name}: {e}")

        return False

    def _generate_pattern_insight(self, pattern: InsightPattern,
                                  analysis_result: EnhancedAnalysisResult,
                                  data: pd.DataFrame) -> Optional[str]:
        """Generate insight text based on pattern template"""

        try:
            template_vars = {}

            # Statistical pattern insights
            if pattern.name == "high_correlation_alert":
                math_result = analysis_result.mathematical_analysis
                correlation = math_result.results.get('statistic', 0)
                variables = analysis_result.metadata.get('variables_analyzed', ['var1', 'var2'])

                template_vars = {
                    'direction': 'positive' if correlation > 0 else 'negative',
                    'var1': variables[0] if len(variables) > 0 else 'variable1',
                    'var2': variables[1] if len(variables) > 1 else 'variable2',
                    'correlation': correlation,
                    'interpretation': 'a meaningful relationship that warrants further investigation'
                }

            elif pattern.name == "outlier_detection":
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:1]:  # Take first numeric column with outliers
                    if data[col].notna().sum() > 3:
                        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                        outliers = z_scores > pattern.conditions['outlier_threshold']
                        outlier_count = outliers.sum()

                        if outlier_count > 0:
                            template_vars = {
                                'variable': col,
                                'outlier_count': outlier_count,
                                'threshold': pattern.conditions['outlier_threshold']
                            }
                            break

            elif pattern.name == "distribution_skewness":
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:1]:
                    if data[col].notna().sum() > 3:
                        skewness = data[col].skew()
                        if abs(skewness) >= pattern.conditions['skewness_threshold']:
                            skew_type = "right" if skewness > 0 else "left"
                            interpretation = "data concentrated towards lower values" if skewness > 0 else "data concentrated towards higher values"

                            template_vars = {
                                'variable': col,
                                'skew_type': skew_type,
                                'skewness': skewness,
                                'interpretation': interpretation
                            }
                            break

            elif pattern.name == "variance_instability":
                numeric_cols = data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols[:1]:
                    if data[col].notna().sum() > 3:
                        mean_val = data[col].mean()
                        cv = data[col].std() / mean_val if mean_val != 0 else 0
                        if cv >= pattern.conditions['cv_threshold']:
                            interpretation = "unstable patterns requiring investigation"

                            template_vars = {
                                'variable': col,
                                'cv': cv,
                                'interpretation': interpretation
                            }
                            break

            elif pattern.name == "segment_performance_gap":
                if len(analysis_result.data) > 1:
                    # Find numeric metric with largest variation
                    numeric_fields = [k for k in analysis_result.data[0].keys()
                                      if isinstance(analysis_result.data[0].get(k), (int, float))]

                    if numeric_fields:
                        metric = numeric_fields[0]
                        values = [item.get(metric, 0) for item in analysis_result.data]

                        if values:
                            max_val, min_val = max(values), min(values)
                            gap_percentage = ((max_val - min_val) / min_val) * 100 if min_val > 0 else 0

                            template_vars = {
                                'top_segment': 'top performer',
                                'bottom_segment': 'lowest performer',
                                'gap_percentage': gap_percentage,
                                'metric': metric
                            }

            # Domain-specific insights
            elif pattern.name == "mortality_experience_variance":
                ae_cols = [col for col in data.columns if 'ae_ratio' in col.lower() or 'a/e' in col.lower()]
                if ae_cols:
                    ae_values = data[ae_cols[0]].dropna()
                    if len(ae_values) > 0:
                        ae_ratio = ae_values.mean()
                        variance_type = "favorable" if ae_ratio < 1.0 else "adverse"

                        template_vars = {
                            'variance_type': variance_type,
                            'ae_ratio': ae_ratio,
                            'regulatory_implication': 'Consider impact on reserves and pricing'
                        }

            elif pattern.name == "lapse_rate_alert":
                lapse_cols = [col for col in data.columns if 'lapse' in col.lower()]
                if lapse_cols:
                    lapse_values = data[lapse_cols[0]].dropna()
                    if len(lapse_values) > 0:
                        lapse_rate = lapse_values.mean()

                        template_vars = {
                            'lapse_rate': lapse_rate * 100,  # Convert to percentage
                            'segment': 'overall portfolio'
                        }

            elif pattern.name == "credit_risk_concentration":
                default_cols = [col for col in data.columns if 'default' in col.lower()]
                if default_cols:
                    default_values = data[default_cols[0]].dropna()
                    if len(default_values) > 0:
                        default_rate = default_values.mean()

                        template_vars = {
                            'segment': 'analyzed portfolio',
                            'default_rate': default_rate * 100  # Convert to percentage
                        }

            # Apply template
            if template_vars:
                try:
                    return pattern.insight_template.format(**template_vars)
                except KeyError as e:
                    logger.warning(f"Missing template variable for {pattern.name}: {e}")
                    return None

        except Exception as e:
            logger.warning(f"Error generating pattern insight for {pattern.name}: {e}")

        return None

    def _generate_comparative_insights(self, analysis_result: EnhancedAnalysisResult,
                                       data: pd.DataFrame) -> List[str]:
        """Generate comparative insights across segments or time periods"""

        insights = []

        try:
            # Compare across categorical segments
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns

            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]

                # Get top categories by volume
                top_categories = data[cat_col].value_counts().head(5).index

                if len(top_categories) >= 2:
                    category_means = data[data[cat_col].isin(top_categories)].groupby(cat_col)[num_col].mean()

                    if len(category_means) > 1:
                        best_category = category_means.idxmax()
                        worst_category = category_means.idxmin()

                        improvement_potential = ((category_means.max() - category_means.min()) /
                                                 category_means.min()) * 100 if category_means.min() != 0 else 0

                        insights.append(
                            f"📊 Performance comparison: {best_category} leads with {category_means.max():.2f} "
                            f"average {num_col}, while {worst_category} shows {improvement_potential:.1f}% "
                            f"improvement potential"
                        )

            # Time-based insights if date columns exist
            date_cols = [col for col in data.columns if
                         any(term in col.lower() for term in ['date', 'time', 'created', 'issued'])]
            if date_cols and len(numeric_cols) > 0:
                date_col = date_cols[0]
                num_col = numeric_cols[0]

                try:
                    data_copy = data.copy()
                    data_copy[date_col] = pd.to_datetime(data_copy[date_col], errors='coerce')
                    recent_data = data_copy[data_copy[date_col].notna()].sort_values(date_col)

                    if len(recent_data) > 10:
                        # Compare recent vs earlier periods
                        split_point = len(recent_data) // 2
                        earlier_data = recent_data.iloc[:split_point]
                        recent_period = recent_data.iloc[split_point:]

                        if len(earlier_data) > 0 and len(recent_period) > 0:
                            earlier_mean = earlier_data[num_col].mean()
                            recent_mean = recent_period[num_col].mean()

                            if earlier_mean != 0:
                                change_pct = ((recent_mean - earlier_mean) / earlier_mean) * 100
                                trend_direction = "increased" if change_pct > 0 else "decreased"

                                insights.append(
                                    f"📈 Temporal trend: {num_col} has {trend_direction} by {abs(change_pct):.1f}% "
                                    f"in recent period compared to earlier data"
                                )

                except Exception:
                    pass  # Skip time-based analysis if date parsing fails

            # Distribution comparisons
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[0], numeric_cols[1]

                # Compare variability
                cv1 = data[col1].std() / data[col1].mean() if data[col1].mean() != 0 else 0
                cv2 = data[col2].std() / data[col2].mean() if data[col2].mean() != 0 else 0

                if abs(cv1 - cv2) > 0.2:  # Significant difference in variability
                    more_variable = col1 if cv1 > cv2 else col2
                    less_variable = col2 if cv1 > cv2 else col1

                    insights.append(
                        f"📊 Variability comparison: {more_variable} shows significantly higher variability "
                        f"(CV={max(cv1, cv2):.2f}) compared to {less_variable} (CV={min(cv1, cv2):.2f})"
                    )

        except Exception as e:
            logger.warning(f"Error generating comparative insights: {e}")

        return insights

    def _generate_predictive_insights(self, analysis_result: EnhancedAnalysisResult,
                                      data: pd.DataFrame, domain: str = None) -> List[str]:
        """Generate predictive insights and forecasts"""

        insights = []

        try:
            # Regression-based predictions
            if (analysis_result.mathematical_analysis and
                    'regression' in analysis_result.mathematical_analysis.method_used):

                math_result = analysis_result.mathematical_analysis
                r_squared = math_result.results.get('r_squared', 0)

                if r_squared > 0.5:
                    prediction_quality = "strong" if r_squared > 0.8 else "moderate"
                    insights.append(
                        f"🔮 Predictive model shows {prediction_quality} explanatory power "
                        f"(R² = {r_squared:.3f}), suitable for forecasting scenarios"
                    )

                    # Add domain-specific prediction insights
                    if domain == 'insurance':
                        insights.append(
                            "💡 Model can be used for experience rating and reserve projections"
                        )
                    elif domain == 'banking':
                        insights.append(
                            "💡 Model suitable for credit risk assessment and capital planning"
                        )
                    elif domain == 'technology':
                        insights.append(
                            "💡 Model applicable for capacity planning and performance optimization"
                        )

            # Trend-based predictions
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                for col in numeric_cols[:2]:  # Check first 2 numeric columns
                    if len(data) > 5 and data[col].notna().sum() > 5:
                        # Simple trend analysis
                        valid_data = data[col].dropna()
                        x = np.arange(len(valid_data))
                        y = valid_data.values

                        if len(y) > 3:
                            correlation = np.corrcoef(x, y)[0, 1] if len(y) > 1 else 0

                            if abs(correlation) > 0.3:
                                trend_direction = "upward" if correlation > 0 else "downward"
                                trend_strength = "strong" if abs(correlation) > 0.7 else "moderate"

                                # Calculate potential future value
                                if abs(correlation) > 0.5:
                                    slope = np.polyfit(x, y, 1)[0]
                                    current_value = y[-1] if len(y) > 0 else 0
                                    projected_change = slope * 5  # Project 5 periods ahead
                                    change_pct = (projected_change / current_value) * 100 if current_value != 0 else 0

                                    insights.append(
                                        f"📊 {col} exhibits a {trend_strength} {trend_direction} trend "
                                        f"(correlation: {correlation:.3f}). Projected {abs(change_pct):.1f}% "
                                        f"{'increase' if change_pct > 0 else 'decrease'} over next 5 periods"
                                    )
                                else:
                                    insights.append(
                                        f"📊 {col} exhibits a {trend_strength} {trend_direction} trend "
                                        f"(correlation: {correlation:.3f})"
                                    )

            # Seasonal pattern predictions
            date_cols = [col for col in data.columns if any(term in col.lower() for term in ['date', 'time'])]
            if date_cols and len(numeric_cols) > 0:
                try:
                    date_col = date_cols[0]
                    num_col = numeric_cols[0]

                    data_copy = data.copy()
                    data_copy[date_col] = pd.to_datetime(data_copy[date_col], errors='coerce')
                    time_data = data_copy[data_copy[date_col].notna()].sort_values(date_col)

                    if len(time_data) > 12:  # Need sufficient data for seasonality
                        # Extract month and check for seasonal patterns
                        time_data['month'] = time_data[date_col].dt.month
                        monthly_means = time_data.groupby('month')[num_col].mean()

                        if len(monthly_means) > 6:  # Need at least 6 months
                            cv_monthly = monthly_means.std() / monthly_means.mean() if monthly_means.mean() != 0 else 0

                            if cv_monthly > 0.2:  # Significant seasonal variation
                                peak_month = monthly_means.idxmax()
                                low_month = monthly_means.idxmin()

                                month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April',
                                               5: 'May', 6: 'June', 7: 'July', 8: 'August',
                                               9: 'September', 10: 'October', 11: 'November', 12: 'December'}

                                insights.append(
                                    f"📅 Seasonal pattern detected in {num_col}: Peak in "
                                    f"{month_names.get(peak_month, peak_month)}, low in "
                                    f"{month_names.get(low_month, low_month)}"
                                )

                except Exception:
                    pass  # Skip seasonal analysis if it fails

        except Exception as e:
            logger.warning(f"Error generating predictive insights: {e}")

        return insights

    def _generate_actionable_recommendations(self, analysis_result: EnhancedAnalysisResult,
                                             data: pd.DataFrame, domain: str = None) -> List[str]:
        """Generate actionable business recommendations"""

        recommendations = []

        try:
            # Data quality recommendations
            missing_data_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100

            if missing_data_pct > 15:
                recommendations.append(
                    f"💼 Data Quality: Critical - Address missing data ({missing_data_pct:.1f}%) through "
                    f"improved collection processes or systematic imputation strategies"
                )
            elif missing_data_pct > 5:
                recommendations.append(
                    f"💼 Data Quality: Moderate - Improve data completeness ({missing_data_pct:.1f}% missing) "
                    f"for more reliable analysis"
                )

            # Statistical recommendations
            if analysis_result.mathematical_analysis:
                violated_assumptions = [
                    assumption for assumption, met in
                    analysis_result.mathematical_analysis.assumptions_met.items()
                    if not met
                ]

                if violated_assumptions:
                    recommendations.append(
                        f"📋 Methodology: Consider alternative analysis methods due to "
                        f"assumption violations: {', '.join(violated_assumptions[:2])}"
                    )

                # Confidence-based recommendations
                confidence = analysis_result.mathematical_analysis.confidence
                if confidence < 0.7:
                    recommendations.append(
                        f"⚠️ Analysis Confidence: Low confidence ({confidence:.2f}) suggests need for "
                        f"additional data or alternative analytical approaches"
                    )

            # Performance recommendations based on analysis results
            if len(analysis_result.data) > 0:
                # Find underperforming segments
                numeric_fields = [k for k in analysis_result.data[0].keys()
                                  if isinstance(analysis_result.data[0].get(k), (int, float))]

                if numeric_fields and len(analysis_result.data) > 1:
                    for metric in numeric_fields[:2]:  # Check top 2 metrics
                        values = [item.get(metric, 0) for item in analysis_result.data]

                        if values and len(values) > 2:
                            avg_value = np.mean(values)
                            std_value = np.std(values)

                            # Identify underperformers (below 1 std dev)
                            underperformer_threshold = avg_value - std_value
                            underperformers = [
                                item for item in analysis_result.data
                                if item.get(metric, 0) < underperformer_threshold
                            ]

                            if underperformers and len(underperformers) > 0:
                                recommendations.append(
                                    f"🎯 Performance: Focus improvement efforts on "
                                    f"{len(underperformers)} underperforming segments in {metric}"
                                )

                            # Identify top performers for best practice sharing
                            top_performer_threshold = avg_value + std_value
                            top_performers = [
                                item for item in analysis_result.data
                                if item.get(metric, 0) > top_performer_threshold
                            ]

                            if top_performers and len(top_performers) > 0:
                                recommendations.append(
                                    f"✅ Best Practices: Analyze and replicate strategies from "
                                    f"{len(top_performers)} top-performing segments in {metric}"
                                )

            # Variance and stability recommendations
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:  # Check first 3 numeric columns
                if data[col].notna().sum() > 3:
                    cv = data[col].std() / data[col].mean() if data[col].mean() != 0 else 0

                    if cv > 0.5:  # High variability
                        recommendations.append(
                            f"📊 Stability: {col} shows high variability (CV={cv:.2f}). "
                            f"Investigate causes and implement stabilization measures"
                        )
                    elif cv < 0.1:  # Very low variability
                        recommendations.append(
                            f"🔍 Analysis Depth: {col} shows low variability (CV={cv:.2f}). "
                            f"Consider segmentation or additional dimensions for deeper insights"
                        )

            # Domain-specific recommendations
            if domain == 'insurance':
                recommendations.extend(self._get_insurance_recommendations(analysis_result, data))
            elif domain == 'banking':
                recommendations.extend(self._get_banking_recommendations(analysis_result, data))
            elif domain == 'technology':
                recommendations.extend(self._get_technology_recommendations(analysis_result, data))

            # General business recommendations
            self._add_general_business_recommendations(recommendations, analysis_result, data)

        except Exception as e:
            logger.warning(f"Error generating actionable recommendations: {e}")

        return recommendations

    def _get_insurance_recommendations(self, analysis_result: EnhancedAnalysisResult,
                                       data: pd.DataFrame) -> List[str]:
        """Generate insurance-specific recommendations"""

        recommendations = []

        try:
            # A/E ratio recommendations
            ae_cols = [col for col in data.columns if 'ae_ratio' in col.lower() or 'a/e' in col.lower()]
            if ae_cols:
                ae_values = data[ae_cols[0]].dropna()
                if len(ae_values) > 0:
                    avg_ae = ae_values.mean()

                    if avg_ae > 1.15:
                        recommendations.append(
                            "⚠️ Insurance: Adverse mortality experience (A/E > 1.15) requires immediate "
                            "review of underwriting guidelines and pricing adequacy"
                        )
                    elif avg_ae > 1.05:
                        recommendations.append(
                            "📋 Insurance: Monitor mortality experience closely and consider "
                            "gradual pricing adjustments"
                        )
                    elif avg_ae < 0.85:
                        recommendations.append(
                            "✅ Insurance: Favorable mortality experience may support reserve releases "
                            "and competitive pricing strategies"
                        )

            # Lapse rate recommendations
            lapse_cols = [col for col in data.columns if 'lapse' in col.lower()]
            if lapse_cols:
                lapse_values = data[lapse_cols[0]].dropna()
                if len(lapse_values) > 0:
                    avg_lapse = lapse_values.mean()

                    if avg_lapse > 0.15:  # 15% lapse rate
                        recommendations.append(
                            "🔄 Insurance: High lapse rates (>15%) indicate urgent need for "
                            "policyholder retention program and competitive review"
                        )
                    elif avg_lapse > 0.10:
                        recommendations.append(
                            "📞 Insurance: Elevated lapse rates suggest implementing "
                            "customer engagement and retention initiatives"
                        )

            # Premium and claims analysis
            premium_cols = [col for col in data.columns if 'premium' in col.lower()]
            claim_cols = [col for col in data.columns if 'claim' in col.lower()]

            if premium_cols and claim_cols:
                premium_total = data[premium_cols[0]].sum()
                claim_total = data[claim_cols[0]].sum()

                if premium_total > 0:
                    loss_ratio = claim_total / premium_total

                    if loss_ratio > 0.85:
                        recommendations.append(
                            f"⚠️ Insurance: High loss ratio ({loss_ratio:.2f}) indicates "
                            "need for pricing review and claims management enhancement"
                        )
                    elif loss_ratio < 0.60:
                        recommendations.append(
                            f"💰 Insurance: Low loss ratio ({loss_ratio:.2f}) presents "
                            "opportunity for competitive pricing or product expansion"
                        )

        except Exception as e:
            logger.warning(f"Error generating insurance recommendations: {e}")

        return recommendations

    def _get_banking_recommendations(self, analysis_result: EnhancedAnalysisResult,
                                     data: pd.DataFrame) -> List[str]:
        """Generate banking-specific recommendations"""

        recommendations = []

        try:
            # Credit risk recommendations
            default_cols = [col for col in data.columns if 'default' in col.lower()]
            if default_cols:
                default_rates = data[default_cols[0]].dropna()
                if len(default_rates) > 0:
                    avg_default_rate = default_rates.mean()

                    if avg_default_rate > 0.05:  # 5% default rate
                        recommendations.append(
                            f"⚠️ Banking: Elevated default rates ({avg_default_rate:.2f}%) require "
                            "enhanced credit monitoring and portfolio risk management"
                        )
                    elif avg_default_rate > 0.03:
                        recommendations.append(
                            f"📊 Banking: Monitor default trends ({avg_default_rate:.2f}%) and "
                            "consider proactive risk mitigation strategies"
                        )

            # Capital adequacy recommendations
            capital_cols = [col for col in data.columns if 'capital' in col.lower()]
            if capital_cols:
                capital_ratios = data[capital_cols[0]].dropna()
                if len(capital_ratios) > 0:
                    avg_capital_ratio = capital_ratios.mean()

                    if avg_capital_ratio < 0.10:  # Below 10%
                        recommendations.append(
                            f"⚠️ Banking: Capital ratio ({avg_capital_ratio:.2f}) approaching "
                            "regulatory minimums - consider capital raising initiatives"
                        )
                    elif avg_capital_ratio > 0.15:  # Above 15%
                        recommendations.append(
                            f"💼 Banking: Strong capital position ({avg_capital_ratio:.2f}) "
                            "supports growth opportunities and dividend considerations"
                        )

            # Loan portfolio diversification
            loan_type_cols = [col for col in data.columns if 'loan' in col.lower() and 'type' in col.lower()]
            if loan_type_cols:
                loan_distribution = data[loan_type_cols[0]].value_counts(normalize=True)
                max_concentration = loan_distribution.max()

                if max_concentration > 0.4:  # More than 40% in one category
                    recommendations.append(
                        f"📊 Banking: High loan concentration ({max_concentration:.1%}) in "
                        f"{loan_distribution.idxmax()} - consider portfolio diversification"
                    )

            # Interest rate risk
            duration_cols = [col for col in data.columns if 'duration' in col.lower()]
            if duration_cols:
                duration_values = data[duration_cols[0]].dropna()
                if len(duration_values) > 0:
                    avg_duration = duration_values.mean()

                    if avg_duration > 5.0:  # High duration
                        recommendations.append(
                            f"📈 Banking: High portfolio duration ({avg_duration:.1f} years) "
                            "indicates significant interest rate risk exposure"
                        )

        except Exception as e:
            logger.warning(f"Error generating banking recommendations: {e}")

        return recommendations

    def _get_technology_recommendations(self, analysis_result: EnhancedAnalysisResult,
                                        data: pd.DataFrame) -> List[str]:
        """Generate technology-specific recommendations"""

        recommendations = []

        try:
            # Performance recommendations
            latency_cols = [col for col in data.columns if 'latency' in col.lower() or 'response' in col.lower()]
            if latency_cols:
                latency_values = data[latency_cols[0]].dropna()
                if len(latency_values) > 0:
                    avg_latency = latency_values.mean()

                    if avg_latency > 1000:  # Over 1 second
                        recommendations.append(
                            f"⚡ Technology: Critical latency issues ({avg_latency:.0f}ms) require "
                            "immediate performance optimization and infrastructure review"
                        )
                    elif avg_latency > 500:  # Over 500ms
                        recommendations.append(
                            f"📈 Technology: Elevated latency ({avg_latency:.0f}ms) suggests "
                            "need for performance tuning and capacity planning"
                        )

            # Uptime recommendations
            uptime_cols = [col for col in data.columns if 'uptime' in col.lower() or 'availability' in col.lower()]
            if uptime_cols:
                uptime_values = data[uptime_cols[0]].dropna()
                if len(uptime_values) > 0:
                    avg_uptime = uptime_values.mean()

                    if avg_uptime < 0.99:  # Below 99%
                        recommendations.append(
                            f"🔧 Technology: Uptime below SLA ({avg_uptime:.3f}) requires "
                            "infrastructure reliability improvements and redundancy planning"
                        )
                    elif avg_uptime < 0.995:  # Below 99.5%
                        recommendations.append(
                            f"📊 Technology: Monitor uptime trends ({avg_uptime:.3f}) and "
                            "proactively address reliability concerns"
                        )

            # User engagement recommendations
            engagement_cols = [col for col in data.columns
                               if any(term in col.lower() for term in ['active', 'engagement', 'session'])]
            if engagement_cols:
                engagement_values = data[engagement_cols[0]].dropna()
                if len(engagement_values) > 1:
                    # Check for declining trends
                    engagement_trend = np.corrcoef(range(len(engagement_values)), engagement_values)[0, 1]

                    if engagement_trend < -0.3:  # Declining trend
                        recommendations.append(
                            "📱 Technology: Declining user engagement trends require "
                            "UX/UI improvements and feature enhancement initiatives"
                        )

            # Error rate recommendations
            error_cols = [col for col in data.columns if 'error' in col.lower()]
            if error_cols:
                error_values = data[error_cols[0]].dropna()
                if len(error_values) > 0:
                    avg_error_rate = error_values.mean()

                    if avg_error_rate > 0.05:  # Above 5%
                        recommendations.append(
                            f"🚨 Technology: High error rate ({avg_error_rate:.2%}) indicates "
                            "critical system stability issues requiring immediate attention"
                        )
                    elif avg_error_rate > 0.01:  # Above 1%
                        recommendations.append(
                            f"⚠️ Technology: Elevated error rate ({avg_error_rate:.2%}) suggests "
                            "need for debugging and quality assurance improvements"
                        )

        except Exception as e:
            logger.warning(f"Error generating technology recommendations: {e}")

        return recommendations

    def _add_general_business_recommendations(self, recommendations: List[str],
                                              analysis_result: EnhancedAnalysisResult,
                                              data: pd.DataFrame):
        """Add general business recommendations applicable across domains"""

        try:
            # Sample size recommendations
            sample_size = len(data)
            if sample_size < 30:
                recommendations.append(
                    f"📊 Sample Size: Small sample ({sample_size} records) may limit "
                    "statistical reliability - consider collecting additional data"
                )
            elif sample_size < 100:
                recommendations.append(
                    f"📈 Data Collection: Moderate sample size ({sample_size}) - "
                    "additional data would improve analytical confidence"
                )

            # Correlation insights for business action
            if (analysis_result.mathematical_analysis and
                    'correlation' in analysis_result.mathematical_analysis.method_used):

                correlation = analysis_result.mathematical_analysis.results.get('statistic', 0)
                if abs(correlation) > 0.7:
                    recommendations.append(
                        "🔍 Strategic Focus: Strong correlations identified - leverage these "
                        "relationships for predictive modeling and strategic planning"
                    )
                elif abs(correlation) < 0.3:
                    recommendations.append(
                        "🔄 Analysis Expansion: Weak correlations suggest exploring "
                        "additional variables or alternative analytical approaches"
                    )

            # Time-based recommendations
            date_cols = [col for col in data.columns if any(term in col.lower() for term in ['date', 'time'])]
            if date_cols:
                recommendations.append(
                    "📅 Temporal Analysis: Time dimension available - consider "
                    "trend analysis and seasonal pattern investigation for strategic planning"
                )

            # Segmentation recommendations
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 2:
                recommendations.append(
                    f"🎯 Segmentation Opportunity: Multiple categorical dimensions ({len(categorical_cols)}) "
                    "available - explore customer/product segmentation strategies"
                )

        except Exception as e:
            logger.warning(f"Error adding general business recommendations: {e}")

    def _generate_quality_insights(self, analysis_result: EnhancedAnalysisResult,
                                   data: pd.DataFrame) -> List[str]:
        """Generate insights about data quality and analysis reliability"""

        insights = []

        try:
            # Data completeness insights
            total_cells = len(data) * len(data.columns)
            missing_cells = data.isnull().sum().sum()
            completeness = 1 - (missing_cells / total_cells) if total_cells > 0 else 0

            if completeness > 0.95:
                insights.append(
                    f"✅ Data Quality: Excellent completeness ({completeness:.1%}) provides "
                    "strong foundation for reliable analysis"
                )
            elif completeness > 0.85:
                insights.append(
                    f"📊 Data Quality: Good completeness ({completeness:.1%}) supports "
                    "reliable analysis with minor limitations"
                )
            elif completeness > 0.70:
                insights.append(
                    f"⚠️ Data Quality: Moderate completeness ({completeness:.1%}) may "
                    "impact analysis reliability - consider data quality improvements"
                )
            else:
                insights.append(
                    f"🚨 Data Quality: Poor completeness ({completeness:.1%}) significantly "
                    "limits analysis reliability - prioritize data quality initiatives"
                )

            # Sample size adequacy
            sample_size = len(data)
            if sample_size > 1000:
                insights.append(
                    f"💪 Statistical Power: Large sample size ({sample_size:,}) provides "
                    "excellent statistical power for reliable conclusions"
                )
            elif sample_size > 100:
                insights.append(
                    f"📈 Statistical Power: Adequate sample size ({sample_size:,}) supports "
                    "reasonable statistical conclusions"
                )
            else:
                insights.append(
                    f"⚠️ Statistical Power: Small sample size ({sample_size:,}) limits "
                    "statistical power - interpret results with caution"
                )

            # Variable diversity
            numeric_count = len(data.select_dtypes(include=[np.number]).columns)
            categorical_count = len(data.select_dtypes(include=['object', 'category']).columns)

            if numeric_count > 5 and categorical_count > 2:
                insights.append(
                    f"🔬 Analytical Scope: Rich dataset ({numeric_count} numeric, "
                    f"{categorical_count} categorical variables) enables comprehensive analysis"
                )
            elif numeric_count > 2 or categorical_count > 1:
                insights.append(
                    f"📊 Analytical Scope: Moderate dataset complexity enables "
                    f"focused analysis across {numeric_count + categorical_count} variables"
                )

            # Analysis confidence assessment
            if analysis_result.confidence_assessment:
                overall_confidence = analysis_result.confidence_assessment.get('overall_confidence', 0.5)

                if overall_confidence > 0.8:
                    insights.append(
                        f"🎯 Analysis Confidence: High confidence ({overall_confidence:.2f}) "
                        "in results supports strategic decision-making"
                    )
                elif overall_confidence > 0.6:
                    insights.append(
                        f"📋 Analysis Confidence: Moderate confidence ({overall_confidence:.2f}) "
                        "suggests results are reliable with some limitations"
                    )
                else:
                    insights.append(
                        f"⚠️ Analysis Confidence: Lower confidence ({overall_confidence:.2f}) "
                        "indicates results should be interpreted cautiously"
                    )

        except Exception as e:
            logger.warning(f"Error generating quality insights: {e}")

        return insights

    def get_insight_patterns_summary(self) -> Dict[str, Any]:
        """Get summary of available insight patterns"""

        pattern_summary = {
            'total_patterns': len(self.insight_patterns),
            'pattern_types': {},
            'domain_specific_patterns': 0,
            'severity_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }

        for pattern in self.insight_patterns:
            # Count by type
            pattern_type = pattern.pattern_type
            if pattern_type not in pattern_summary['pattern_types']:
                pattern_summary['pattern_types'][pattern_type] = 0
            pattern_summary['pattern_types'][pattern_type] += 1

            # Count domain-specific
            if pattern.domain_specific:
                pattern_summary['domain_specific_patterns'] += 1

            # Count by severity
            severity = pattern.severity
            if severity in pattern_summary['severity_distribution']:
                pattern_summary['severity_distribution'][severity] += 1

        return pattern_summary

    def get_generated_insights_history(self) -> List[str]:
        """Get history of generated insights for learning and improvement"""
        return self.generated_insights.copy()

    def clear_insights_history(self):
        """Clear the insights history"""
        self.generated_insights.clear()
        logger.info("Insights history cleared")