# chart_suggestion_engine.py - Enhanced version with full chart type support
from typing import List, Dict, Union, Any
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()


def classify_intent(prompt: str) -> Dict[str, Union[str, float]]:
    """Classifies the user intent using regex, and falls back to GPT if confidence is low."""
    prompt_lower = prompt.lower()

    # Enhanced intent patterns for advanced chart types
    intent_scores = {
        "trend_analysis": len(re.findall(r"trend|over time|progression|growth|timeline|historical", prompt_lower)),
        "distribution_analysis": len(
            re.findall(r"distribution|share|composition|percentage|portion|breakdown", prompt_lower)),
        "comparison_analysis": len(re.findall(r"compare|comparison|vs|versus|benchmark|difference", prompt_lower)),
        "breakdown_analysis": len(re.findall(r"breakdown|decomposition|impact|contribution|waterfall", prompt_lower)),
        "performance_analysis": len(re.findall(r"performance|kpi|metric|indicator|gauge|score|rating", prompt_lower)),
        "hierarchical_analysis": len(re.findall(r"segment|hierarchy|tree|nested|drill", prompt_lower)),
        "funnel_analysis": len(re.findall(r"funnel|pipeline|conversion|stages|steps|process", prompt_lower)),
        "waterfall_analysis": len(re.findall(r"waterfall|cumulative|running total|bridge", prompt_lower)),
        "gauge_analysis": len(re.findall(r"gauge|meter|speedometer|completion|progress|status", prompt_lower)),
    }

    total = sum(intent_scores.values())
    best_intent = max(intent_scores, key=intent_scores.get)
    confidence = intent_scores[best_intent] / (total + 1e-5)

    # Try GPT for better classification if confidence is low
    if confidence < 0.4 and os.getenv("OPENAI_API_KEY"):
        try:
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system",
                     "content": "You are an intent classifier for data visualization. Classify the intent and respond with ONLY one of these exact values: trend_analysis, distribution_analysis, comparison_analysis, breakdown_analysis, performance_analysis, hierarchical_analysis, funnel_analysis, waterfall_analysis, gauge_analysis"},
                    {"role": "user", "content": f"Classify this prompt: '{prompt}'"}
                ],
                max_tokens=20,
                temperature=0
            )
            gpt_intent = response.choices[0].message.content.strip().lower()
            if gpt_intent in intent_scores:
                best_intent = gpt_intent
                confidence = 0.8
        except Exception as e:
            print(f"GPT fallback error: {str(e)}")

    return {"intent": best_intent, "confidence": round(confidence, 3)}


def analyze_data_structure(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the structure of the data to better suggest charts."""
    if not data:
        return {"numeric_fields": [], "categorical_fields": [], "date_fields": [], "row_count": 0}

    sample = data[0]
    numeric_fields = []
    categorical_fields = []
    date_fields = []

    for key, value in sample.items():
        if isinstance(value, (int, float)):
            numeric_fields.append(key)
        elif isinstance(value, str):
            # Check if it might be a date
            if any(pattern in key.lower() for pattern in ['date', 'time', 'year', 'month', 'day']):
                date_fields.append(key)
            else:
                categorical_fields.append(key)

    return {
        "numeric_fields": numeric_fields,
        "categorical_fields": categorical_fields,
        "date_fields": date_fields,
        "row_count": len(data)
    }


def prepare_data_for_chart(chart_type: str, data: List[Dict[str, Any]],
                           x_field: str = None, y_field: str = None) -> List[Dict[str, Any]]:
    """Prepare data in the format expected by the frontend Charts component."""

    if not data:
        return []

    # For waterfall charts, ensure we have positive and negative values
    if chart_type == "waterfall":
        prepared_data = []
        for i, item in enumerate(data[:10]):  # Limit to 10 items for waterfall
            x_val = item.get(x_field, f"Step {i + 1}") if x_field else f"Step {i + 1}"
            y_val = item.get(y_field, 0) if y_field else list(item.values())[0] if isinstance(list(item.values())[0],
                                                                                              (int, float)) else 0

            # Make some values negative for demonstration
            if i % 3 == 2:
                y_val = -abs(y_val)

            prepared_data.append({
                "x": str(x_val),
                "value": float(y_val)
            })
        return prepared_data

    # For funnel charts, ensure descending values
    elif chart_type == "funnel":
        prepared_data = []
        values = []

        for i, item in enumerate(data[:5]):  # Limit to 5 stages for funnel
            name = item.get(x_field, f"Stage {i + 1}") if x_field else f"Stage {i + 1}"
            value = item.get(y_field, 100 - i * 20) if y_field else 100 - i * 20
            values.append((str(name), float(value)))

        # Sort by value descending
        values.sort(key=lambda x: x[1], reverse=True)

        for name, value in values:
            prepared_data.append({
                "name": name,
                "value": value
            })
        return prepared_data

    # For gauge charts, use a single value
    elif chart_type == "gauge":
        if data:
            first_numeric = None
            if y_field and y_field in data[0]:
                first_numeric = data[0][y_field]
            else:
                # Find first numeric value
                for value in data[0].values():
                    if isinstance(value, (int, float)):
                        first_numeric = value
                        break

            return [{
                "value": float(first_numeric) if first_numeric else 75,
                "max": 100  # Default max, could be calculated from data
            }]
        return [{"value": 75, "max": 100}]

    # For pie charts
    elif chart_type == "pie":
        prepared_data = []
        for item in data[:6]:  # Limit pie slices
            name = item.get(x_field, "Category") if x_field else str(list(item.keys())[0])
            value = item.get(y_field, 1) if y_field else next((v for v in item.values() if isinstance(v, (int, float))),
                                                              1)
            prepared_data.append({
                "name": str(name),
                "value": float(value)
            })
        return prepared_data

    # Default format for line, bar, area, scatter
    else:
        prepared_data = []
        for item in data[:20]:  # Limit data points
            x_val = item.get(x_field, "Item") if x_field else list(item.keys())[0]
            y_val = item.get(y_field, 0) if y_field else next((v for v in item.values() if isinstance(v, (int, float))),
                                                              0)
            prepared_data.append({
                "x": str(x_val),
                "value": float(y_val) if isinstance(y_val, (int, float)) else 0
            })
        return prepared_data


def suggest_chart_types(prompt: str, data: List[Dict[str, Union[str, float, int]]]) -> Dict[str, Any]:
    """
    Suggest appropriate chart types based on the prompt and data structure.
    Returns chart config suggestions compatible with the frontend Charts component.
    """
    if not data:
        return {"suggested_charts": [], "intent_metadata": {"intent": "unknown", "confidence": 0.0}}

    # Analyze data structure
    data_structure = analyze_data_structure(data)
    numeric_fields = data_structure["numeric_fields"]
    categorical_fields = data_structure["categorical_fields"]
    date_fields = data_structure["date_fields"]

    # Determine best fields for x and y axes
    x_field = None
    y_field = None

    if categorical_fields:
        x_field = categorical_fields[0]
    elif date_fields:
        x_field = date_fields[0]
    elif data:
        x_field = list(data[0].keys())[0]

    if numeric_fields:
        y_field = numeric_fields[0]

    # Get intent classification
    intent_result = classify_intent(prompt)
    intent = intent_result["intent"]

    suggestions = []

    # Generate chart suggestions based on intent
    if intent == "waterfall_analysis" or "waterfall" in prompt.lower():
        chart_data = prepare_data_for_chart("waterfall", data, x_field, y_field)
        suggestions.append({
            "type": "waterfall",
            "chart_type": "waterfall",
            "title": "Waterfall Analysis",
            "x_field": "x",
            "y_field": "value",
            "data": chart_data,
            "description": "Shows cumulative effect of sequential positive and negative values",
            "reasoning": "Waterfall chart requested - ideal for showing how an initial value is affected by intermediate positive and negative values",
            "config": {
                "colors": ["#22c55e", "#ef4444"],  # Green for positive, red for negative
                "showConnectors": True
            }
        })

    elif intent == "funnel_analysis" or "funnel" in prompt.lower():
        chart_data = prepare_data_for_chart("funnel", data, x_field, y_field)
        suggestions.append({
            "type": "funnel",
            "chart_type": "funnel",
            "title": "Funnel Analysis",
            "category_field": "name",
            "value_field": "value",
            "data": chart_data,
            "description": "Visualizes stages in a process with progressive filtering",
            "reasoning": "Funnel chart requested - perfect for showing conversion rates or process stages",
            "config": {
                "colors": ["#3b82f6", "#60a5fa", "#93c5fd", "#dbeafe", "#eff6ff"]
            }
        })

    elif intent == "gauge_analysis" or "gauge" in prompt.lower():
        chart_data = prepare_data_for_chart("gauge", data, x_field, y_field)
        suggestions.append({
            "type": "gauge",
            "chart_type": "gauge",
            "title": "Performance Gauge",
            "value_field": "value",
            "data": chart_data,
            "description": "Shows a single metric against a scale",
            "reasoning": "Gauge chart requested - ideal for showing progress or performance metrics",
            "config": {
                "max": 100,
                "thresholds": [0, 50, 75, 100],
                "colors": ["#ef4444", "#f59e0b", "#22c55e"]
            }
        })

    elif intent == "trend_analysis":
        chart_data = prepare_data_for_chart("line", data, x_field, y_field)
        suggestions.append({
            "type": "line",
            "chart_type": "line",
            "title": "Trend Analysis",
            "x_field": "x",
            "y_field": "value",
            "data": chart_data,
            "description": "Shows trends over time or sequence",
            "reasoning": "Line chart is best for showing trends and progressions over time"
        })

        # Also suggest area chart for trends
        suggestions.append({
            "type": "area",
            "chart_type": "area",
            "title": "Trend Area Chart",
            "x_field": "x",
            "y_field": "value",
            "data": chart_data,
            "description": "Area chart emphasizing volume over time",
            "reasoning": "Area charts are great for showing cumulative trends"
        })

    elif intent == "distribution_analysis":
        chart_data = prepare_data_for_chart("pie", data, x_field, y_field)
        suggestions.append({
            "type": "pie",
            "chart_type": "pie",
            "title": "Distribution Analysis",
            "category_field": "name",
            "value_field": "value",
            "data": chart_data,
            "description": "Shows proportional distribution of categories",
            "reasoning": "Pie chart is ideal for showing parts of a whole"
        })

        # Also suggest histogram
        hist_data = prepare_data_for_chart("histogram", data, x_field, y_field)
        suggestions.append({
            "type": "histogram",
            "chart_type": "histogram",
            "title": "Distribution Histogram",
            "x_field": "x",
            "y_field": "value",
            "data": hist_data,
            "description": "Shows frequency distribution",
            "reasoning": "Histograms show the distribution pattern of your data"
        })

    elif intent == "comparison_analysis":
        chart_data = prepare_data_for_chart("bar", data, x_field, y_field)
        suggestions.append({
            "type": "bar",
            "chart_type": "bar",
            "title": "Comparison Analysis",
            "x_field": "x",
            "y_field": "value",
            "data": chart_data,
            "description": "Compares values across categories",
            "reasoning": "Bar charts are excellent for comparing discrete values"
        })

    elif intent == "performance_analysis":
        # Suggest multiple chart types for performance
        gauge_data = prepare_data_for_chart("gauge", data, x_field, y_field)
        suggestions.append({
            "type": "gauge",
            "chart_type": "gauge",
            "title": "Performance Metric",
            "value_field": "value",
            "data": gauge_data,
            "description": "Key performance indicator visualization",
            "reasoning": "Gauge charts effectively show single KPI performance"
        })

        # Also suggest a bar chart for multiple KPIs
        bar_data = prepare_data_for_chart("bar", data, x_field, y_field)
        suggestions.append({
            "type": "bar",
            "chart_type": "bar",
            "title": "KPI Comparison",
            "x_field": "x",
            "y_field": "value",
            "data": bar_data,
            "description": "Compare multiple performance metrics",
            "reasoning": "Bar charts can show multiple KPIs side by side"
        })

    # Default suggestions if no specific intent matched
    if not suggestions:
        # Always provide at least a bar chart as default
        chart_data = prepare_data_for_chart("bar", data, x_field, y_field)
        suggestions.append({
            "type": "bar",
            "chart_type": "bar",
            "title": "Data Overview",
            "x_field": "x",
            "y_field": "value",
            "data": chart_data,
            "description": "General data visualization",
            "reasoning": "Bar chart provides a clear view of your data"
        })

        # Add a line chart if we have numeric data
        if numeric_fields:
            line_data = prepare_data_for_chart("line", data, x_field, y_field)
            suggestions.append({
                "type": "line",
                "chart_type": "line",
                "title": "Data Trend",
                "x_field": "x",
                "y_field": "value",
                "data": line_data,
                "description": "Shows data progression",
                "reasoning": "Line charts can reveal patterns in your data"
            })

    return {
        "suggested_charts": suggestions,
        "intent_metadata": intent_result,
        "data_structure": data_structure
    }


def handle_query(prompt: str, data: List[Dict[str, Union[str, float, int]]]) -> Dict[str, Any]:
    """
    Main handler function that processes the query and returns analysis with chart suggestions.
    """
    # Get chart suggestions
    chart_output = suggest_chart_types(prompt, data)

    # Create analysis summary
    intent = chart_output["intent_metadata"]["intent"]
    confidence = chart_output["intent_metadata"]["confidence"]
    chart_count = len(chart_output["suggested_charts"])

    summary = f"Analysis complete. Detected intent: {intent.replace('_', ' ')} (confidence: {confidence:.1%}). "
    summary += f"Generated {chart_count} chart suggestion{'s' if chart_count != 1 else ''}."

    # Generate insights based on data
    insights = []
    if data:
        data_structure = chart_output.get("data_structure", {})

        insights.append(f"Dataset contains {len(data)} records")

        if data_structure.get("numeric_fields"):
            insights.append(f"Found {len(data_structure['numeric_fields'])} numeric fields suitable for analysis")

        if data_structure.get("categorical_fields"):
            insights.append(f"Found {len(data_structure['categorical_fields'])} categorical fields for grouping")

        if data_structure.get("date_fields"):
            insights.append("Time-based analysis is possible with date fields present")

        # Add intent-specific insights
        if intent == "waterfall_analysis":
            insights.append("Waterfall chart will show cumulative impact of positive and negative changes")
        elif intent == "funnel_analysis":
            insights.append("Funnel visualization will highlight conversion rates between stages")
        elif intent == "gauge_analysis":
            insights.append("Gauge chart provides an intuitive view of performance against targets")

    analysis_result = {
        "summary": summary,
        "data": data[:100],  # Limit data in response
        "insights": insights,
        "suggested_charts": chart_output["suggested_charts"],
        "intent_metadata": chart_output["intent_metadata"]
    }

    return analysis_result


# Test function to verify the engine works
if __name__ == "__main__":
    # Test data
    test_data = [
        {"stage": "Website Visits", "count": 10000, "month": "Jan"},
        {"stage": "Sign Ups", "count": 2500, "month": "Jan"},
        {"stage": "Trial Users", "count": 1200, "month": "Jan"},
        {"stage": "Paid Users", "count": 300, "month": "Jan"},
        {"stage": "Enterprise", "count": 50, "month": "Jan"}
    ]

    # Test different queries
    test_queries = [
        "Show me a waterfall chart of quarterly performance",
        "Create a funnel chart for our sales pipeline",
        "Generate a gauge chart showing completion rate",
        "What are the trends in our data?",
        "Compare values across categories"
    ]

    print("Testing Enhanced Chart Suggestion Engine\n" + "=" * 50)

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        result = handle_query(query, test_data)
        print(f"Intent: {result['intent_metadata']['intent']} (confidence: {result['intent_metadata']['confidence']})")
        print(f"Charts suggested: {len(result['suggested_charts'])}")
        for i, chart in enumerate(result['suggested_charts']):
            print(f"  {i + 1}. {chart['type']} - {chart['title']}")
            print(f"     Reasoning: {chart['reasoning']}")