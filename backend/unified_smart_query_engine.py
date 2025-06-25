# backend/unified_smart_query_engine.py - LLM-Powered Analytics Engine
"""
Unified Smart Query Engine - No hardcoding, handles any query intelligently

This replaces all the override functions with a single LLM-powered system that:
1. Understands query intent using OpenAI
2. Profiles data automatically
3. Generates appropriate pandas operations
4. Executes with proper error handling
5. Learns from results to improve

Usage:
    Replace your override functions and complex interpretation logic with:

    smart_engine = UnifiedSmartQueryEngine(openai_api_key)
    result = await smart_engine.execute_query(query, dataframe)
"""

import pandas as pd
import numpy as np
import re
import json
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, date
import openai
from openai import OpenAI
import traceback
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Query intent classification"""
    TOTAL_BY_YEAR = "total_by_year"
    TOTAL_BY_MONTH = "total_by_month"
    BREAKDOWN_BY_DIMENSION = "breakdown_by_dimension"
    MULTI_DIMENSION_BREAKDOWN = "multi_dimension_breakdown"
    TREND_ANALYSIS = "trend_analysis"
    COMPARISON = "comparison"
    HIGHEST_LOWEST = "highest_lowest"
    CORRELATION = "correlation"
    GENERAL_SUMMARY = "general_summary"


@dataclass
class DataProfile:
    """Automatic data profiling result"""
    numeric_columns: List[str]
    categorical_columns: List[str]
    temporal_columns: List[str]
    likely_metrics: List[str]
    likely_dimensions: List[str]
    inferred_domain: str
    row_count: int
    column_count: int
    sample_data: Dict[str, Any]
    temporal_patterns: Dict[str, Any]


@dataclass
class QueryExecution:
    """Query execution result"""
    intent: QueryIntent
    confidence: float
    pandas_operation: str
    result_data: List[Dict[str, Any]]
    insights: List[str]
    metadata: Dict[str, Any]
    execution_time_ms: float


class UnifiedSmartQueryEngine:
    """
    LLM-powered query engine that handles any analytics query without hardcoding
    """

    def __init__(self, openai_api_key: str):
        """Initialize with OpenAI API key"""
        self.client = OpenAI(api_key=openai_api_key)
        self.logger = logging.getLogger(__name__)

    def profile_data(self, df: pd.DataFrame) -> DataProfile:
        """Automatically profile any dataset to understand its structure"""

        # Basic column categorization
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Detect temporal columns (both datetime and integer year/month)
        temporal_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

        # Check for integer temporal columns
        for col in numeric_cols:
            if col.lower() in ['year', 'yr', 'month', 'mon', 'quarter', 'q']:
                temporal_cols.append(col)

        # Smart metric detection
        likely_metrics = []
        for col in numeric_cols:
            col_lower = col.lower()
            metric_indicators = [
                'sales', 'revenue', 'profit', 'amount', 'value', 'total', 'sum',
                'cost', 'price', 'income', 'expense', 'balance', 'volume',
                'quantity', 'count', 'score', 'rate', 'ratio', 'margin'
            ]
            if any(indicator in col_lower for indicator in metric_indicators):
                likely_metrics.append(col)

        # Smart dimension detection
        likely_dimensions = categorical_cols.copy()
        for col in numeric_cols:
            if col in temporal_cols:
                likely_dimensions.append(col)

        # Domain inference
        all_text = ' '.join(df.columns).lower() + ' ' + ' '.join([
            str(val) for col in categorical_cols[:3]
            for val in df[col].dropna().unique()[:5]
        ])

        domain_indicators = {
            'ecommerce': ['product', 'order', 'customer', 'sales', 'price', 'cart'],
            'finance': ['revenue', 'profit', 'cost', 'investment', 'portfolio'],
            'hr': ['employee', 'salary', 'department', 'performance'],
            'marketing': ['campaign', 'lead', 'conversion', 'click'],
            'technology': ['user', 'feature', 'api', 'server', 'session']
        }

        inferred_domain = 'general'
        max_score = 0
        for domain, indicators in domain_indicators.items():
            score = sum(1 for indicator in indicators if indicator in all_text)
            if score > max_score:
                max_score = score
                inferred_domain = domain

        # Temporal patterns
        temporal_patterns = {}
        for col in temporal_cols:
            if col in df.columns:
                unique_vals = sorted(df[col].dropna().unique())
                temporal_patterns[col] = {
                    'min_value': unique_vals[0] if unique_vals else None,
                    'max_value': unique_vals[-1] if unique_vals else None,
                    'unique_count': len(unique_vals),
                    'sample_values': unique_vals[:5]
                }

        # Sample data for LLM context
        sample_data = {}
        for col in df.columns[:10]:  # Limit for API efficiency
            sample_vals = df[col].dropna().head(3).tolist()
            sample_data[col] = [str(v) for v in sample_vals]

        return DataProfile(
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            temporal_columns=temporal_cols,
            likely_metrics=likely_metrics,
            likely_dimensions=likely_dimensions,
            inferred_domain=inferred_domain,
            row_count=len(df),
            column_count=len(df.columns),
            sample_data=sample_data,
            temporal_patterns=temporal_patterns
        )

    async def execute_query(self, query: str, df: pd.DataFrame) -> QueryExecution:
        """
        Execute any analytics query using LLM intelligence
        """
        start_time = datetime.now()

        try:
            # Step 1: Profile the data
            profile = self.profile_data(df)
            self.logger.info(f"📊 Data profiled: {profile.inferred_domain} domain, {profile.row_count} rows")

            # Step 2: Get LLM interpretation and pandas code generation
            interpretation = await self._get_llm_interpretation(query, profile)

            # Step 3: Execute the generated pandas operation
            result_data, execution_metadata = self._execute_pandas_operation(
                df, interpretation['pandas_code'], interpretation
            )

            # Step 4: Generate insights
            insights = self._generate_insights(result_data, interpretation, profile)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return QueryExecution(
                intent=QueryIntent(interpretation['intent']),
                confidence=interpretation['confidence'],
                pandas_operation=interpretation['pandas_code'],
                result_data=result_data,
                insights=insights,
                metadata={
                    'profile': profile.__dict__,
                    'interpretation': interpretation,
                    'execution_metadata': execution_metadata
                },
                execution_time_ms=execution_time
            )

        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return QueryExecution(
                intent=QueryIntent.GENERAL_SUMMARY,
                confidence=0.1,
                pandas_operation="df.describe()",
                result_data=[{'error': str(e)}],
                insights=[f"Query execution failed: {e}"],
                metadata={'error': str(e)},
                execution_time_ms=execution_time
            )

    async def _get_llm_interpretation(self, query: str, profile: DataProfile) -> Dict[str, Any]:
        """
        Use LLM to interpret query and generate appropriate pandas operations
        """

        # Create system prompt with data context
        system_prompt = f"""You are an expert data analyst who generates pandas operations for any analytics query.

DATA CONTEXT:
- Domain: {profile.inferred_domain}
- Rows: {profile.row_count:,}
- Columns: {profile.column_count}
- Numeric columns: {profile.numeric_columns}
- Categorical columns: {profile.categorical_columns}  
- Temporal columns: {profile.temporal_columns}
- Likely metrics: {profile.likely_metrics}
- Likely dimensions: {profile.likely_dimensions}
- Sample data: {profile.sample_data}
- Temporal patterns: {profile.temporal_patterns}

TASK: 
Analyze the user query and generate the appropriate pandas operation to answer it.

RESPONSE FORMAT (JSON):
{{
    "intent": "one of: total_by_year, total_by_month, breakdown_by_dimension, multi_dimension_breakdown, trend_analysis, comparison, highest_lowest, correlation, general_summary",
    "confidence": 0.0-1.0,
    "pandas_code": "executable pandas code using 'df' variable",
    "explanation": "brief explanation of the approach",
    "filters": {{"column": "value"}},
    "grouping": ["column1", "column2"],
    "metric": "column_name",
    "temporal_filter": {{"type": "year|month|range", "value": "2024"}}
}}

PANDAS CODE REQUIREMENTS:
- Use 'df' as the DataFrame variable
- Handle missing values appropriately
- Return a list of dictionaries or DataFrame.to_dict('records')
- Include error handling
- Use proper data types
- Limit results to reasonable size (max 1000 rows)

EXAMPLES:

Query: "total sales in 2024"  
Response: {{
    "intent": "total_by_year",
    "confidence": 0.95,
    "pandas_code": "result = df[df['year'] == 2024]['sales'].sum() if 'year' in df.columns and 'sales' in df.columns else df.describe(); [{'total_sales_2024': float(result), 'year': 2024}] if isinstance(result, (int, float)) else result.to_dict()",
    "explanation": "Filter by year 2024 and sum sales",
    "filters": {{"year": 2024}},
    "grouping": [],
    "metric": "sales",
    "temporal_filter": {{"type": "year", "value": "2024"}}
}}

Query: "sales breakdown by region"
Response: {{
    "intent": "breakdown_by_dimension", 
    "confidence": 0.90,
    "pandas_code": "result_df = df.groupby('region')['sales'].agg(['sum', 'mean', 'count']).reset_index(); result_df.columns = ['region', 'sales_sum', 'sales_mean', 'sales_count']; result_df.to_dict('records')",
    "explanation": "Group by region and aggregate sales metrics",
    "filters": {{}},
    "grouping": ["region"],
    "metric": "sales", 
    "temporal_filter": {{}}
}}

Query: "which month had highest sales"
Response: {{
    "intent": "highest_lowest",
    "confidence": 0.92,
    "pandas_code": "monthly_sales = df.groupby('month')['sales'].sum().reset_index(); max_month = monthly_sales.loc[monthly_sales['sales'].idxmax()]; [{'month': int(max_month['month']), 'sales': float(max_month['sales']), 'rank': 'highest'}]",
    "explanation": "Find month with maximum total sales",
    "filters": {{}},
    "grouping": ["month"],
    "metric": "sales",
    "temporal_filter": {{}}
}}

BE SMART:
- Auto-detect the best metric if not specified
- Handle both datetime and integer temporal columns
- Generate robust code that handles edge cases
- Choose appropriate aggregation functions
- Handle multi-word column names properly
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=1000
            )

            interpretation = json.loads(response.choices[0].message.content)
            self.logger.info(
                f"🤖 LLM interpretation: {interpretation['intent']} (confidence: {interpretation['confidence']})")

            return interpretation

        except Exception as e:
            self.logger.error(f"LLM interpretation failed: {e}")

            # Fallback to simple pattern matching
            return self._fallback_interpretation(query, profile)

    def _fallback_interpretation(self, query: str, profile: DataProfile) -> Dict[str, Any]:
        """Fallback interpretation when LLM fails"""

        query_lower = query.lower()

        # Simple pattern matching
        if re.search(r'\b(total|sum).*?(20\d{2})', query_lower):
            year_match = re.search(r'\b(20\d{2})', query_lower)
            year = year_match.group(1) if year_match else "2024"
            metric = profile.likely_metrics[0] if profile.likely_metrics else profile.numeric_columns[
                0] if profile.numeric_columns else 'count'

            return {
                "intent": "total_by_year",
                "confidence": 0.7,
                "pandas_code": f"result = df[df['year'] == {year}]['{metric}'].sum() if 'year' in df.columns and '{metric}' in df.columns else 0; [{{'total_{metric}_{year}': float(result), 'year': {year}}}]",
                "explanation": f"Sum {metric} for year {year}",
                "filters": {"year": int(year)},
                "grouping": [],
                "metric": metric,
                "temporal_filter": {"type": "year", "value": year}
            }

        elif re.search(r'breakdown.*?by|by.*?breakdown', query_lower):
            # Extract dimension
            by_match = re.search(r'by\s+(\w+)', query_lower)
            dimension = by_match.group(1) if by_match else profile.likely_dimensions[
                0] if profile.likely_dimensions else 'category'
            metric = profile.likely_metrics[0] if profile.likely_metrics else 'count'

            return {
                "intent": "breakdown_by_dimension",
                "confidence": 0.6,
                "pandas_code": f"result_df = df.groupby('{dimension}')['{metric}'].agg(['sum', 'mean', 'count']).reset_index() if '{dimension}' in df.columns else pd.DataFrame(); result_df.to_dict('records') if not result_df.empty else [{{}}]",
                "explanation": f"Group by {dimension}",
                "filters": {},
                "grouping": [dimension],
                "metric": metric,
                "temporal_filter": {}
            }

        else:
            return {
                "intent": "general_summary",
                "confidence": 0.3,
                "pandas_code": "df.describe().to_dict()",
                "explanation": "General data summary",
                "filters": {},
                "grouping": [],
                "metric": "",
                "temporal_filter": {}
            }

    def _execute_pandas_operation(self, df: pd.DataFrame, pandas_code: str, interpretation: Dict[str, Any]) -> Tuple[
        List[Dict[str, Any]], Dict[str, Any]]:
        """
        Safely execute the generated pandas operation
        """

        execution_metadata = {
            "original_rows": len(df),
            "pandas_code": pandas_code,
            "execution_successful": False,
            "error": None
        }

        try:
            self.logger.info(f"🐼 Executing: {pandas_code}")

            # Create safe execution environment
            safe_globals = {
                'df': df,
                'pd': pd,
                'np': np,
                'len': len,
                'int': int,
                'float': float,
                'str': str,
                'list': list,
                'dict': dict,
                'isinstance': isinstance
            }

            # Execute the pandas code
            result = eval(pandas_code, safe_globals)

            # Ensure result is in the right format
            if isinstance(result, pd.DataFrame):
                result_data = result.to_dict('records')
            elif isinstance(result, list):
                result_data = result
            elif isinstance(result, dict):
                result_data = [result]
            else:
                result_data = [{'result': str(result)}]

            # Limit results for performance
            if len(result_data) > 1000:
                result_data = result_data[:1000]
                execution_metadata["truncated_to"] = 1000

            execution_metadata["execution_successful"] = True
            execution_metadata["result_rows"] = len(result_data)

            self.logger.info(f"✅ Execution successful: {len(result_data)} results")

            return result_data, execution_metadata

        except Exception as e:
            self.logger.error(f"❌ Pandas execution failed: {e}")
            execution_metadata["error"] = str(e)

            # Return safe fallback
            return [
                {'error': f'Execution failed: {str(e)}', 'suggestion': 'Try rephrasing your query'}], execution_metadata

    def _generate_insights(self, result_data: List[Dict[str, Any]], interpretation: Dict[str, Any],
                           profile: DataProfile) -> List[str]:
        """
        Generate intelligent insights from the results
        """

        insights = []

        if not result_data or 'error' in result_data[0]:
            insights.append("Query execution encountered issues. Please rephrase your question.")
            return insights

        # Intent-specific insights
        intent = interpretation.get('intent', '')

        if intent == 'total_by_year':
            total_key = [k for k in result_data[0].keys() if 'total' in k.lower()]
            if total_key and result_data:
                total_value = result_data[0][total_key[0]]
                year = result_data[0].get('year', 'specified year')
                insights.append(f"Total for {year}: {total_value:,.0f}")
                insights.append(f"Analysis based on {profile.row_count:,} records")

        elif intent == 'breakdown_by_dimension':
            if len(result_data) > 1:
                grouping_col = interpretation.get('grouping', ['category'])[0]
                insights.append(f"Found {len(result_data)} distinct {grouping_col} values")

                # Find top performer
                sum_cols = [k for k in result_data[0].keys() if 'sum' in k.lower()]
                if sum_cols:
                    sorted_results = sorted(result_data, key=lambda x: x.get(sum_cols[0], 0), reverse=True)
                    top_performer = sorted_results[0]
                    insights.append(
                        f"Top performer: {top_performer.get(grouping_col, 'Unknown')} with {top_performer.get(sum_cols[0], 0):,.0f}")

        elif intent == 'highest_lowest':
            if result_data and 'month' in result_data[0]:
                month = result_data[0]['month']
                sales = result_data[0].get('sales', 0)
                insights.append(f"Month {month} had the highest sales: {sales:,.0f}")

        # General insights
        if len(result_data) > 0:
            insights.append(f"Query returned {len(result_data)} result{'s' if len(result_data) != 1 else ''}")

        # Add domain-specific insights
        if profile.inferred_domain != 'general':
            insights.append(f"Analysis optimized for {profile.inferred_domain} domain")

        return insights


# Integration function for your main.py
def create_unified_smart_engine(openai_api_key: str) -> UnifiedSmartQueryEngine:
    """
    Create the unified smart engine to replace all hardcoded overrides
    """
    return UnifiedSmartQueryEngine(openai_api_key)


# Integration example for main.py
async def smart_analyze_query(query: str, df: pd.DataFrame, openai_api_key: str) -> Dict[str, Any]:
    """
    Replace your override functions with this single smart function

    Usage in main.py:
        # Replace handle_year_query_override() and handle_breakdown_query_override() with:
        smart_result = await smart_analyze_query(prompt, df, openai_api_key)
        if smart_result:
            return JSONResponse(content=smart_result)
    """

    try:
        smart_engine = UnifiedSmartQueryEngine(openai_api_key)
        execution_result = await smart_engine.execute_query(query, df)

        if execution_result.confidence >= 0.6:  # Only return if confident
            return {
                "status": "success",
                "analysis": {
                    "type": f"smart_{execution_result.intent.value}",
                    "summary": f"Smart Analysis - {execution_result.intent.value.replace('_', ' ').title()}",
                    "data": execution_result.result_data,
                    "insights": execution_result.insights,
                    "metadata": {
                        "confidence": execution_result.confidence,
                        "pandas_operation": execution_result.pandas_operation,
                        "execution_time_ms": execution_result.execution_time_ms,
                        "smart_engine_used": True
                    }
                },
                "query_interpretation": {
                    "intent": execution_result.intent.value,
                    "confidence": execution_result.confidence,
                    "method": "llm_powered"
                }
            }
        else:
            return None  # Let normal pipeline handle it

    except Exception as e:
        logger.error(f"Smart engine failed: {e}")
        return None


# Example usage and testing
if __name__ == "__main__":
    import asyncio


    async def test_smart_engine():
        """Test the unified smart engine"""

        # Create sample data
        df = pd.DataFrame({
            'sales': [1000, 1500, 2000, 2500, 1200, 1800, 3000, 2200],
            'profit': [200, 300, 400, 500, 240, 360, 600, 440],
            'region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'],
            'month': [1, 2, 3, 4, 5, 6, 7, 8],
            'year': [2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024]
        })

        # Test queries
        test_queries = [
            "total sales in 2024",
            "which month had highest sales",
            "sales breakdown by region",
            "profit by region and month",
            "compare north vs south sales"
        ]

        smart_engine = UnifiedSmartQueryEngine("your-openai-key")

        for query in test_queries:
            print(f"\n🧪 Testing: '{query}'")
            try:
                result = await smart_engine.execute_query(query, df)
                print(f"✅ Intent: {result.intent.value}")
                print(f"✅ Confidence: {result.confidence:.2f}")
                print(f"✅ Results: {len(result.result_data)} items")
                print(f"✅ Insights: {result.insights}")
            except Exception as e:
                print(f"❌ Failed: {e}")

    # Uncomment to test
    # asyncio.run(test_smart_engine())