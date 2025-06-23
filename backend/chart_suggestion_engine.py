from typing import List, Dict, Union
import re
import openai
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def classify_intent(prompt: str) -> Dict[str, Union[str, float]]:
    """Classifies the user intent using regex, and falls back to GPT if confidence is low."""
    prompt = prompt.lower()
    intent_scores = {
        "trend_analysis": len(re.findall(r"trend|over time|progression|growth", prompt)),
        "distribution_analysis": len(re.findall(r"distribution|share|composition|percentage", prompt)),
        "comparison_analysis": len(re.findall(r"compare|comparison|vs|versus|benchmark", prompt)),
        "breakdown_analysis": len(re.findall(r"breakdown|decomposition|impact|contribution", prompt)),
        "performance_analysis": len(re.findall(r"performance|kpi|metric|indicator", prompt)),
        "hierarchical_analysis": len(re.findall(r"segment|hierarchy|tree|nested", prompt)),
    }
    total = sum(intent_scores.values())
    best_intent = max(intent_scores, key=intent_scores.get)
    confidence = intent_scores[best_intent] / (total + 1e-5)

    if confidence < 0.4:
        try:
            gpt_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an intent classifier for data analysis."},
                    {"role": "user", "content": f"Classify this prompt into one of these intents: trend_analysis, distribution_analysis, comparison_analysis, breakdown_analysis, performance_analysis, hierarchical_analysis. Prompt: '{prompt}'"}
                ]
            )
            gpt_intent = gpt_response.choices[0].message.content.strip().lower()
            if gpt_intent in intent_scores:
                best_intent = gpt_intent
                confidence = 0.8
        except Exception as e:
            print("GPT fallback error:", str(e))

    return {"intent": best_intent, "confidence": round(confidence, 3)}


def suggest_chart_types(prompt: str, data: List[Dict[str, Union[str, float, int]]]) -> Dict[str, Union[List[Dict[str, str]], Dict[str, Union[str, float]]]]:
    """
    Suggest appropriate chart types based on the prompt and data structure.
    Returns a dict with chart config suggestions and detected intent metadata.
    """
    prompt_lower = prompt.lower()
    suggestions = []

    if not data:
        return {"suggested_charts": [], "intent_metadata": {"intent": "unknown", "confidence": 0.0}}

    sample = data[0]
    numeric_fields = [k for k, v in sample.items() if isinstance(v, (int, float))]
    categorical_fields = [k for k, v in sample.items() if isinstance(v, str)]

    x_field = categorical_fields[0] if categorical_fields else None

    intent_result = classify_intent(prompt)
    intent = intent_result["intent"]

    if intent == "trend_analysis":
        for y in numeric_fields:
            suggestions.append({"type": "line", "x": x_field, "y": y})

    elif intent == "distribution_analysis":
        for y in numeric_fields:
            suggestions.append({"type": "pie", "x": x_field, "y": y})

    elif intent == "comparison_analysis":
        for y in numeric_fields:
            suggestions.append({"type": "bar", "x": x_field, "y": y})

    elif intent == "breakdown_analysis":
        for y in numeric_fields:
            suggestions.append({"type": "waterfall", "x": x_field, "y": y})

    elif intent == "performance_analysis":
        suggestions.append({"type": "radar"})

    elif intent == "hierarchical_analysis":
        suggestions.append({"type": "treemap"})

    if not suggestions:
        for y in numeric_fields:
            suggestions.append({"type": "bar", "x": x_field, "y": y})

    return {"suggested_charts": suggestions, "intent_metadata": intent_result}


def handle_query(prompt: str, data: List[Dict[str, Union[str, float, int]]]) -> Dict:
    analysis_result = {
        "summary": "Here is your analysis...",
        "data": data,
        "insights": ["Insight A", "Insight B"]
    }

    chart_output = suggest_chart_types(prompt, data)
    analysis_result.update(chart_output)  # includes suggested_charts and intent_metadata

    return analysis_result
