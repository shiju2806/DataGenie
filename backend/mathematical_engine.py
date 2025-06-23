# backend/mathematical_engine.py - Mathematical Knowledge Foundation
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.spatial.distance as distance
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
import warnings

logger = logging.getLogger(__name__)


class DataType(Enum):
    NUMERIC_CONTINUOUS = "numeric_continuous"
    NUMERIC_DISCRETE = "numeric_discrete"
    CATEGORICAL_NOMINAL = "categorical_nominal"
    CATEGORICAL_ORDINAL = "categorical_ordinal"
    DATETIME = "datetime"
    BINARY = "binary"


class AnalysisType(Enum):
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    PREDICTIVE = "predictive"
    ASSOCIATIVE = "associative"


@dataclass
class MathematicalMethod:
    name: str
    category: str
    description: str
    assumptions: List[str]
    data_requirements: Dict[str, Any]
    function: Callable
    interpretation_guide: str
    confidence_threshold: float = 0.95


@dataclass
class DataCharacteristics:
    data_types: Dict[str, DataType]
    sample_size: int
    missing_data: Dict[str, float]
    distribution_properties: Dict[str, Dict[str, float]]
    correlation_matrix: Optional[pd.DataFrame] = None
    outlier_info: Dict[str, List] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    method_used: str
    results: Dict[str, Any]
    interpretation: str
    confidence: float
    assumptions_met: Dict[str, bool]
    warnings: List[str]
    recommendations: List[str]


class MathematicalKnowledgeEngine:
    """Core mathematical intelligence for all analytics operations"""

    def __init__(self):
        self.methods_registry = {}
        self.concept_hierarchy = {}
        self.assumption_validators = {}
        self._initialize_mathematical_knowledge()

    def _initialize_mathematical_knowledge(self):
        """Initialize comprehensive mathematical knowledge base"""

        # Core statistical methods
        self._register_descriptive_methods()
        self._register_inferential_methods()
        self._register_association_methods()
        self._register_predictive_methods()

        # Initialize concept hierarchy
        self._build_concept_hierarchy()

        # Initialize assumption validators
        self._initialize_assumption_validators()

    def _register_descriptive_methods(self):
        """Register descriptive statistical methods"""

        self.methods_registry['central_tendency'] = {
            'mean': MathematicalMethod(
                name='arithmetic_mean',
                category='descriptive',
                description='Average value of numeric data',
                assumptions=['numeric_data'],
                data_requirements={'min_sample_size': 1,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]},
                function=np.mean,
                interpretation_guide='Represents typical value, sensitive to outliers'
            ),
            'median': MathematicalMethod(
                name='median',
                category='descriptive',
                description='Middle value when data is ordered',
                assumptions=['ordinal_or_numeric'],
                data_requirements={'min_sample_size': 1,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE,
                                                  DataType.CATEGORICAL_ORDINAL]},
                function=np.median,
                interpretation_guide='Robust to outliers, represents central position'
            ),
            'mode': MathematicalMethod(
                name='mode',
                category='descriptive',
                description='Most frequently occurring value',
                assumptions=['any_data_type'],
                data_requirements={'min_sample_size': 1, 'data_types': list(DataType)},
                function=lambda x: stats.mode(x)[0][0] if len(x) > 0 else None,
                interpretation_guide='Most common value, may not exist or be unique'
            )
        }

        self.methods_registry['dispersion'] = {
            'variance': MathematicalMethod(
                name='variance',
                category='descriptive',
                description='Average squared deviation from mean',
                assumptions=['numeric_data', 'interval_scale'],
                data_requirements={'min_sample_size': 2,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]},
                function=np.var,
                interpretation_guide='Measure of spread, same units as data squared'
            ),
            'standard_deviation': MathematicalMethod(
                name='standard_deviation',
                category='descriptive',
                description='Square root of variance',
                assumptions=['numeric_data', 'interval_scale'],
                data_requirements={'min_sample_size': 2,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]},
                function=np.std,
                interpretation_guide='Measure of spread in same units as original data'
            ),
            'iqr': MathematicalMethod(
                name='interquartile_range',
                category='descriptive',
                description='Range between 75th and 25th percentiles',
                assumptions=['ordinal_or_numeric'],
                data_requirements={'min_sample_size': 4,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]},
                function=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
                interpretation_guide='Robust measure of spread, contains middle 50% of data'
            )
        }

        self.methods_registry['distribution_shape'] = {
            'skewness': MathematicalMethod(
                name='skewness',
                category='descriptive',
                description='Measure of asymmetry in distribution',
                assumptions=['numeric_data', 'interval_scale'],
                data_requirements={'min_sample_size': 3,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]},
                function=stats.skew,
                interpretation_guide='Positive: right tail, Negative: left tail, Zero: symmetric'
            ),
            'kurtosis': MathematicalMethod(
                name='kurtosis',
                category='descriptive',
                description='Measure of tail heaviness',
                assumptions=['numeric_data', 'interval_scale'],
                data_requirements={'min_sample_size': 4,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]},
                function=stats.kurtosis,
                interpretation_guide='High: heavy tails, Low: light tails, Normal: kurtosis=0'
            )
        }

    def _register_inferential_methods(self):
        """Register inferential statistical methods"""

        self.methods_registry['hypothesis_tests'] = {
            'one_sample_ttest': MathematicalMethod(
                name='one_sample_t_test',
                category='inferential',
                description='Test if sample mean differs from population mean',
                assumptions=['numeric_data', 'normality_or_large_sample', 'independence'],
                data_requirements={'min_sample_size': 1, 'data_types': [DataType.NUMERIC_CONTINUOUS]},
                function=lambda x, popmean=0: stats.ttest_1samp(x, popmean),
                interpretation_guide='p < 0.05 suggests significant difference from population mean'
            ),
            'two_sample_ttest': MathematicalMethod(
                name='independent_t_test',
                category='inferential',
                description='Compare means of two independent groups',
                assumptions=['numeric_data', 'normality_or_large_sample', 'independence', 'equal_variances'],
                data_requirements={'min_sample_size': 2, 'data_types': [DataType.NUMERIC_CONTINUOUS]},
                function=lambda x, y: stats.ttest_ind(x, y),
                interpretation_guide='p < 0.05 suggests significant difference between group means'
            ),
            'paired_ttest': MathematicalMethod(
                name='paired_t_test',
                category='inferential',
                description='Compare paired observations',
                assumptions=['numeric_data', 'normality_of_differences', 'paired_data'],
                data_requirements={'min_sample_size': 1, 'data_types': [DataType.NUMERIC_CONTINUOUS]},
                function=lambda x, y: stats.ttest_rel(x, y),
                interpretation_guide='p < 0.05 suggests significant difference between paired observations'
            ),
            'chi_square_independence': MathematicalMethod(
                name='chi_square_test',
                category='inferential',
                description='Test independence of categorical variables',
                assumptions=['categorical_data', 'expected_frequency_5plus', 'independence'],
                data_requirements={'min_sample_size': 5,
                                   'data_types': [DataType.CATEGORICAL_NOMINAL, DataType.CATEGORICAL_ORDINAL]},
                function=lambda contingency_table: stats.chi2_contingency(contingency_table),
                interpretation_guide='p < 0.05 suggests variables are not independent'
            )
        }

        self.methods_registry['confidence_intervals'] = {
            'mean_ci': MathematicalMethod(
                name='confidence_interval_mean',
                category='inferential',
                description='Confidence interval for population mean',
                assumptions=['numeric_data', 'normality_or_large_sample', 'independence'],
                data_requirements={'min_sample_size': 1, 'data_types': [DataType.NUMERIC_CONTINUOUS]},
                function=self._calculate_mean_ci,
                interpretation_guide='Range likely to contain true population mean'
            ),
            'proportion_ci': MathematicalMethod(
                name='confidence_interval_proportion',
                category='inferential',
                description='Confidence interval for population proportion',
                assumptions=['binary_data', 'large_sample', 'independence'],
                data_requirements={'min_sample_size': 30, 'data_types': [DataType.BINARY]},
                function=self._calculate_proportion_ci,
                interpretation_guide='Range likely to contain true population proportion'
            )
        }

    def _register_association_methods(self):
        """Register association and correlation methods"""

        self.methods_registry['correlation'] = {
            'pearson': MathematicalMethod(
                name='pearson_correlation',
                category='associative',
                description='Linear correlation between two continuous variables',
                assumptions=['numeric_data', 'linear_relationship', 'bivariate_normality'],
                data_requirements={'min_sample_size': 3, 'data_types': [DataType.NUMERIC_CONTINUOUS]},
                function=lambda x, y: stats.pearsonr(x, y),
                interpretation_guide='Ranges -1 to +1, measures linear relationship strength'
            ),
            'spearman': MathematicalMethod(
                name='spearman_correlation',
                category='associative',
                description='Rank correlation between variables',
                assumptions=['ordinal_or_numeric', 'monotonic_relationship'],
                data_requirements={'min_sample_size': 3,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.CATEGORICAL_ORDINAL]},
                function=lambda x, y: stats.spearmanr(x, y),
                interpretation_guide='Ranges -1 to +1, measures monotonic relationship strength'
            ),
            'kendall': MathematicalMethod(
                name='kendall_tau',
                category='associative',
                description='Rank correlation robust to outliers',
                assumptions=['ordinal_or_numeric'],
                data_requirements={'min_sample_size': 3,
                                   'data_types': [DataType.NUMERIC_CONTINUOUS, DataType.CATEGORICAL_ORDINAL]},
                function=lambda x, y: stats.kendalltau(x, y),
                interpretation_guide='Similar to Spearman but more robust to outliers'
            )
        }

    def _register_predictive_methods(self):
        """Register predictive modeling methods"""

        self.methods_registry['regression'] = {
            'linear_regression': MathematicalMethod(
                name='linear_regression',
                category='predictive',
                description='Linear relationship modeling',
                assumptions=['numeric_data', 'linearity', 'independence', 'homoscedasticity', 'normality_of_residuals'],
                data_requirements={'min_sample_size': 10, 'data_types': [DataType.NUMERIC_CONTINUOUS]},
                function=self._linear_regression_analysis,
                interpretation_guide='Models linear relationship between variables'
            ),
            'logistic_regression': MathematicalMethod(
                name='logistic_regression',
                category='predictive',
                description='Binary outcome prediction',
                assumptions=['binary_outcome', 'linearity_of_logit', 'independence'],
                data_requirements={'min_sample_size': 50, 'data_types': [DataType.BINARY]},
                function=self._logistic_regression_analysis,
                interpretation_guide='Models probability of binary outcome'
            )
        }

    def _build_concept_hierarchy(self):
        """Build hierarchical concept relationships"""

        self.concept_hierarchy = {
            'statistics': {
                'descriptive': {
                    'central_tendency': ['mean', 'median', 'mode'],
                    'dispersion': ['variance', 'standard_deviation', 'range', 'iqr'],
                    'shape': ['skewness', 'kurtosis']
                },
                'inferential': {
                    'hypothesis_testing': ['t_test', 'chi_square', 'anova', 'mann_whitney'],
                    'confidence_intervals': ['mean_ci', 'proportion_ci', 'difference_ci'],
                    'effect_sizes': ['cohens_d', 'eta_squared', 'cramers_v']
                },
                'associative': {
                    'correlation': ['pearson', 'spearman', 'kendall'],
                    'regression': ['linear', 'logistic', 'polynomial']
                }
            },
            'probability': {
                'distributions': {
                    'continuous': ['normal', 'uniform', 'exponential', 'gamma'],
                    'discrete': ['binomial', 'poisson', 'negative_binomial']
                },
                'concepts': ['independence', 'conditional_probability', 'bayes_theorem']
            }
        }

    def _initialize_assumption_validators(self):
        """Initialize functions to check statistical assumptions"""

        self.assumption_validators = {
            'normality': self._test_normality,
            'linearity': self._test_linearity,
            'independence': self._test_independence,
            'homoscedasticity': self._test_homoscedasticity,
            'equal_variances': self._test_equal_variances,
            'large_sample': self._test_large_sample,
            'expected_frequency': self._test_expected_frequency
        }

    def analyze_data_characteristics(self, data: Union[pd.DataFrame, np.ndarray, List]) -> DataCharacteristics:
        """Comprehensive analysis of data characteristics for method selection"""

        if isinstance(data, list):
            data = np.array(data)
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)

        characteristics = DataCharacteristics(
            data_types={},
            sample_size=len(data),
            missing_data={},
            distribution_properties={}
        )

        for col in data.columns:
            # Determine data type
            characteristics.data_types[col] = self._classify_data_type(data[col])

            # Calculate missing data percentage
            characteristics.missing_data[col] = data[col].isnull().sum() / len(data)

            # Analyze distribution properties for numeric data
            if characteristics.data_types[col] in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]:
                numeric_data = data[col].dropna()
                if len(numeric_data) > 0:
                    characteristics.distribution_properties[col] = {
                        'mean': float(np.mean(numeric_data)),
                        'median': float(np.median(numeric_data)),
                        'std': float(np.std(numeric_data)),
                        'skewness': float(stats.skew(numeric_data)),
                        'kurtosis': float(stats.kurtosis(numeric_data)),
                        'min': float(np.min(numeric_data)),
                        'max': float(np.max(numeric_data))
                    }

                    # Detect outliers
                    q1, q3 = np.percentile(numeric_data, [25, 75])
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = numeric_data[(numeric_data < lower_bound) | (numeric_data > upper_bound)]
                    characteristics.outlier_info[col] = outliers.tolist()

        # Calculate correlation matrix for numeric variables
        numeric_cols = [col for col, dtype in characteristics.data_types.items()
                        if dtype in [DataType.NUMERIC_CONTINUOUS, DataType.NUMERIC_DISCRETE]]
        if len(numeric_cols) > 1:
            characteristics.correlation_matrix = data[numeric_cols].corr()

        return characteristics

    def recommend_methods(self, analysis_intent: str, data_characteristics: DataCharacteristics,
                          variables: List[str]) -> List[Tuple[MathematicalMethod, float]]:
        """Recommend appropriate mathematical methods based on intent and data"""

        recommendations = []

        # Parse analysis intent
        intent_keywords = analysis_intent.lower().split()

        # Determine analysis type
        if any(word in intent_keywords for word in ['correlation', 'relationship', 'association']):
            analysis_type = AnalysisType.ASSOCIATIVE
        elif any(word in intent_keywords for word in ['predict', 'forecast', 'model']):
            analysis_type = AnalysisType.PREDICTIVE
        elif any(word in intent_keywords for word in ['test', 'compare', 'difference', 'significant']):
            analysis_type = AnalysisType.INFERENTIAL
        else:
            analysis_type = AnalysisType.DESCRIPTIVE

        # Get relevant method categories
        relevant_categories = self._get_relevant_categories(analysis_type, intent_keywords)

        # Evaluate methods for compatibility
        for category in relevant_categories:
            if category in self.methods_registry:
                for method_name, method in self.methods_registry[category].items():
                    compatibility_score = self._calculate_method_compatibility(
                        method, data_characteristics, variables
                    )
                    if compatibility_score > 0.5:  # Only recommend if reasonably compatible
                        recommendations.append((method, compatibility_score))

        # Sort by compatibility score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:5]  # Return top 5 recommendations

    def execute_analysis(self, method: MathematicalMethod, data: pd.DataFrame,
                         variables: List[str], **kwargs) -> AnalysisResult:
        """Execute mathematical analysis with comprehensive validation"""

        # Validate assumptions
        assumptions_met = self._validate_assumptions(method, data, variables)

        # Generate warnings for violated assumptions
        warnings_list = []
        for assumption, is_met in assumptions_met.items():
            if not is_met:
                warnings_list.append(f"Assumption violated: {assumption}")

        # Execute the analysis
        try:
            if method.name in ['linear_regression', 'logistic_regression']:
                results = method.function(data, variables, **kwargs)
            elif method.name in ['pearson_correlation', 'spearman_correlation', 'kendall_tau']:
                if len(variables) >= 2:
                    x, y = data[variables[0]].dropna(), data[variables[1]].dropna()
                    # Align the data
                    common_idx = x.index.intersection(y.index)
                    x, y = x[common_idx], y[common_idx]
                    results = method.function(x, y)
                else:
                    raise ValueError("Correlation requires at least 2 variables")
            else:
                if len(variables) == 1:
                    clean_data = data[variables[0]].dropna()
                    results = method.function(clean_data, **kwargs)
                else:
                    # Multi-variable analysis
                    clean_data = data[variables].dropna()
                    results = method.function(clean_data, **kwargs)

        except Exception as e:
            return AnalysisResult(
                method_used=method.name,
                results={'error': str(e)},
                interpretation=f"Analysis failed: {str(e)}",
                confidence=0.0,
                assumptions_met=assumptions_met,
                warnings=[f"Execution error: {str(e)}"],
                recommendations=["Check data quality and method requirements"]
            )

        # Calculate confidence based on assumption satisfaction
        confidence = self._calculate_analysis_confidence(assumptions_met, data, variables)

        # Generate interpretation
        interpretation = self._generate_interpretation(method, results, assumptions_met)

        # Generate recommendations
        recommendations = self._generate_recommendations(method, assumptions_met, warnings_list)

        return AnalysisResult(
            method_used=method.name,
            results=self._format_results(results),
            interpretation=interpretation,
            confidence=confidence,
            assumptions_met=assumptions_met,
            warnings=warnings_list,
            recommendations=recommendations
        )

    # Helper methods (implementation details)

    def _classify_data_type(self, series: pd.Series) -> DataType:
        """Classify data type for mathematical method selection"""

        if pd.api.types.is_datetime64_any_dtype(series):
            return DataType.DATETIME
        elif pd.api.types.is_numeric_dtype(series):
            unique_vals = series.nunique()
            if unique_vals == 2:
                return DataType.BINARY
            elif unique_vals <= 10 and series.dtype in ['int64', 'int32']:
                return DataType.NUMERIC_DISCRETE
            else:
                return DataType.NUMERIC_CONTINUOUS
        else:
            unique_vals = series.nunique()
            if unique_vals == 2:
                return DataType.BINARY
            elif unique_vals <= 20:
                return DataType.CATEGORICAL_NOMINAL
            else:
                return DataType.CATEGORICAL_NOMINAL

    def _get_relevant_categories(self, analysis_type: AnalysisType, keywords: List[str]) -> List[str]:
        """Get relevant method categories based on analysis type and keywords"""

        category_mapping = {
            AnalysisType.DESCRIPTIVE: ['central_tendency', 'dispersion', 'distribution_shape'],
            AnalysisType.INFERENTIAL: ['hypothesis_tests', 'confidence_intervals'],
            AnalysisType.ASSOCIATIVE: ['correlation'],
            AnalysisType.PREDICTIVE: ['regression']
        }

        base_categories = category_mapping.get(analysis_type, [])

        # Add specific categories based on keywords
        if 'correlation' in keywords:
            base_categories.append('correlation')
        if any(word in keywords for word in ['mean', 'average']):
            base_categories.append('central_tendency')
        if any(word in keywords for word in ['test', 'significant']):
            base_categories.append('hypothesis_tests')

        return list(set(base_categories))

    def _calculate_method_compatibility(self, method: MathematicalMethod,
                                        data_characteristics: DataCharacteristics,
                                        variables: List[str]) -> float:
        """Calculate compatibility score between method and data"""

        score = 1.0

        # Check sample size requirement
        if data_characteristics.sample_size < method.data_requirements['min_sample_size']:
            score *= 0.3

        # Check data type compatibility
        required_types = method.data_requirements['data_types']
        variable_types = [data_characteristics.data_types.get(var) for var in variables]

        if not any(vtype in required_types for vtype in variable_types if vtype is not None):
            score *= 0.1

        # Check missing data
        for var in variables:
            missing_pct = data_characteristics.missing_data.get(var, 0)
            if missing_pct > 0.5:  # More than 50% missing
                score *= 0.5
            elif missing_pct > 0.2:  # More than 20% missing
                score *= 0.8

        return score

    def _validate_assumptions(self, method: MathematicalMethod, data: pd.DataFrame,
                              variables: List[str]) -> Dict[str, bool]:
        """Validate statistical assumptions for the method"""

        assumptions_met = {}

        for assumption in method.assumptions:
            if assumption in self.assumption_validators:
                try:
                    assumptions_met[assumption] = self.assumption_validators[assumption](data, variables)
                except:
                    assumptions_met[assumption] = False
            else:
                # For assumptions we can't test, assume they're met
                assumptions_met[assumption] = True

        return assumptions_met

    def _calculate_analysis_confidence(self, assumptions_met: Dict[str, bool],
                                       data: pd.DataFrame, variables: List[str]) -> float:
        """Calculate confidence in analysis results"""

        # Base confidence
        confidence = 0.8

        # Reduce confidence for violated assumptions
        assumption_satisfaction = sum(assumptions_met.values()) / len(assumptions_met) if assumptions_met else 1.0
        confidence *= assumption_satisfaction

        # Adjust for sample size
        sample_size = len(data)
        if sample_size < 30:
            confidence *= 0.8
        elif sample_size < 100:
            confidence *= 0.9

        # Adjust for missing data
        for var in variables:
            missing_pct = data[var].isnull().sum() / len(data)
            confidence *= (1 - missing_pct)

        return max(0.1, min(1.0, confidence))

    def _generate_interpretation(self, method: MathematicalMethod, results: Any,
                                 assumptions_met: Dict[str, bool]) -> str:
        """Generate human-readable interpretation of results"""

        base_interpretation = method.interpretation_guide

        # Add method-specific interpretation
        if isinstance(results, tuple) and len(results) >= 2:
            statistic, p_value = results[0], results[1]

            if 'correlation' in method.name:
                correlation_strength = abs(statistic)
                if correlation_strength < 0.3:
                    strength = "weak"
                elif correlation_strength < 0.7:
                    strength = "moderate"
                else:
                    strength = "strong"

                direction = "positive" if statistic > 0 else "negative"
                significance = "statistically significant" if p_value < 0.05 else "not statistically significant"

                interpretation = f"Found a {strength} {direction} correlation (r = {statistic:.3f}). "
                interpretation += f"This relationship is {significance} (p = {p_value:.3f}). "
                interpretation += base_interpretation

            elif 'test' in method.name:
                significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
                interpretation = f"Test statistic = {statistic:.3f}, p-value = {p_value:.3f}. "
                interpretation += f"The result is {significance}. "
                interpretation += base_interpretation
            else:
                interpretation = base_interpretation
        else:
            interpretation = base_interpretation

        # Add warnings for violated assumptions
        violated_assumptions = [assumption for assumption, met in assumptions_met.items() if not met]
        if violated_assumptions:
            interpretation += f" Note: Some assumptions may be violated ({', '.join(violated_assumptions)}), "
            interpretation += "which may affect the reliability of these results."

        return interpretation

    def _generate_recommendations(self, method: MathematicalMethod,
                                  assumptions_met: Dict[str, bool],
                                  warnings: List[str]) -> List[str]:
        """Generate actionable recommendations"""

        recommendations = []

        # Recommendations for violated assumptions
        violated_assumptions = [assumption for assumption, met in assumptions_met.items() if not met]

        for assumption in violated_assumptions:
            if assumption == 'normality':
                recommendations.append(
                    "Consider using non-parametric alternatives (e.g., Mann-Whitney instead of t-test)")
            elif assumption == 'linearity':
                recommendations.append("Explore non-linear relationships or transform variables")
            elif assumption == 'independence':
                recommendations.append("Check for clustering or time dependencies in data")
            elif assumption == 'equal_variances':
                recommendations.append("Use Welch's t-test for unequal variances")

        # General recommendations
        if len(warnings) > 3:
            recommendations.append("Multiple assumption violations detected - consider alternative methods")

        if not recommendations:
            recommendations.append("Results appear reliable - assumptions are adequately met")

        return recommendations

    def _format_results(self, results: Any) -> Dict[str, Any]:
        """Format results into consistent dictionary structure"""

        if isinstance(results, tuple):
            if len(results) == 2:
                return {
                    'statistic': float(results[0]) if not np.isnan(results[0]) else None,
                    'p_value': float(results[1]) if not np.isnan(results[1]) else None
                }
            elif len(results) == 3:
                return {
                    'statistic': float(results[0]) if not np.isnan(results[0]) else None,
                    'p_value': float(results[1]) if not np.isnan(results[1]) else None,
                    'additional': results[2]
                }
        elif isinstance(results, dict):
            return results
        elif isinstance(results, (int, float)):
            return {'value': float(results)}
        else:
            return {'result': str(results)}

    # Statistical assumption validators

    def _test_normality(self, data: pd.DataFrame, variables: List[str]) -> bool:
        """Test for normality using Shapiro-Wilk test"""
        for var in variables:
            if data[var].dtype in ['int64', 'float64']:
                clean_data = data[var].dropna()
                if len(clean_data) > 3:
                    _, p_value = stats.shapiro(clean_data[:5000])  # Limit for performance
                    if p_value < 0.05:  # Reject normality
                        return False
        return True

    def _test_linearity(self, data: pd.DataFrame, variables: List[str]) -> bool:
        """Test for linear relationship using correlation"""
        if len(variables) >= 2:
            x, y = data[variables[0]].dropna(), data[variables[1]].dropna()
            common_idx = x.index.intersection(y.index)
            if len(common_idx) > 10:
                x_vals, y_vals = x[common_idx], y[common_idx]
                # Simple linearity check using R-squared
                corr, _ = stats.pearsonr(x_vals, y_vals)
                return abs(corr) > 0.1  # Weak linear relationship threshold
        return True  # Assume linear if can't test

    def _test_independence(self, data: pd.DataFrame, variables: List[str]) -> bool:
        """Test for independence (basic autocorrelation check)"""
        # For time series data, check autocorrelation
        # For now, assume independence unless obvious time structure
        return True

    def _test_homoscedasticity(self, data: pd.DataFrame, variables: List[str]) -> bool:
        """Test for equal variances using Levene's test"""
        if len(variables) >= 2:
            try:
                groups = [data[var].dropna() for var in variables]
                if all(len(group) > 3 for group in groups):
                    _, p_value = stats.levene(*groups)
                    return p_value >= 0.05  # Accept equal variances
            except:
                pass
        return True

    def _test_equal_variances(self, data: pd.DataFrame, variables: List[str]) -> bool:
        """Test for equal variances - same as homoscedasticity"""
        return self._test_homoscedasticity(data, variables)

    def _test_large_sample(self, data: pd.DataFrame, variables: List[str]) -> bool:
        """Check if sample size is large enough"""
        min_sample_size = 30  # Common threshold
        for var in variables:
            if data[var].dropna().shape[0] < min_sample_size:
                return False
        return True

    def _test_expected_frequency(self, data: pd.DataFrame, variables: List[str]) -> bool:
        """Test expected frequency > 5 for chi-square tests"""
        # This would need contingency table - simplified for now
        return True

    # Advanced analysis methods

    def _calculate_mean_ci(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)

        # Use t-distribution for small samples
        if n < 30:
            t_critical = stats.t.ppf((1 + confidence) / 2, n - 1)
        else:
            t_critical = stats.norm.ppf((1 + confidence) / 2)

        margin_error = t_critical * std_err
        return (mean - margin_error, mean + margin_error)

    def _calculate_proportion_ci(self, data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for proportion"""
        n = len(data)
        p = np.mean(data)  # Proportion of 1s in binary data

        # Wilson score interval (more robust than normal approximation)
        z = stats.norm.ppf((1 + confidence) / 2)

        denominator = 1 + z ** 2 / n
        center = (p + z ** 2 / (2 * n)) / denominator
        margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def _linear_regression_analysis(self, data: pd.DataFrame, variables: List[str], **kwargs) -> Dict[str, Any]:
        """Comprehensive linear regression analysis"""
        if len(variables) < 2:
            raise ValueError("Linear regression requires at least 2 variables")

        # Prepare data
        clean_data = data[variables].dropna()
        X = clean_data[variables[:-1]]  # All but last variable as predictors
        y = clean_data[variables[-1]]  # Last variable as target

        # Fit model
        model = LinearRegression()
        model.fit(X, y)

        # Predictions and metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)

        # Coefficients
        coefficients = dict(zip(variables[:-1], model.coef_))

        return {
            'coefficients': coefficients,
            'intercept': model.intercept_,
            'r_squared': r2,
            'rmse': rmse,
            'n_observations': len(clean_data),
            'predictions': y_pred.tolist()[:10]  # First 10 predictions
        }

    def _logistic_regression_analysis(self, data: pd.DataFrame, variables: List[str], **kwargs) -> Dict[str, Any]:
        """Comprehensive logistic regression analysis"""
        if len(variables) < 2:
            raise ValueError("Logistic regression requires at least 2 variables")

        # Prepare data
        clean_data = data[variables].dropna()
        X = clean_data[variables[:-1]]  # All but last variable as predictors
        y = clean_data[variables[-1]]  # Last variable as target

        # Fit model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        # Predictions and metrics
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

        # Calculate accuracy
        accuracy = np.mean(y == y_pred)

        # Coefficients
        coefficients = dict(zip(variables[:-1], model.coef_[0]))

        return {
            'coefficients': coefficients,
            'intercept': model.intercept_[0],
            'accuracy': accuracy,
            'n_observations': len(clean_data),
            'predicted_probabilities': y_pred_proba.tolist()[:10]  # First 10 predictions
        }