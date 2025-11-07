import pandas as pd
import google.generativeai as genai
import json
import os
from flask import jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from bson import ObjectId

# ✅ Configure Gemini same as chatbot
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# -------------------------------
# 🔹 Build Transaction Features
# -------------------------------
def build_transaction_features(txs: pd.DataFrame):
    """Generate features from transactions."""
    if txs.empty:
        return {}

    txs = txs.copy()
    txs["date"] = pd.to_datetime(txs["date"], errors="coerce")
    txs = txs.dropna(subset=["date"])
    if txs.empty:
        return {}

    txs["month"] = txs["date"].dt.to_period("M")
    monthly = txs.groupby("month")["amount"].sum().rename("monthly_spend").reset_index()
    avg_monthly = monthly["monthly_spend"].mean()
    last_month = monthly.iloc[-1]["monthly_spend"] if len(monthly) else 0.0

    recency_days = (pd.Timestamp.now() - txs["date"].max()).days
    frequency = txs.shape[0] / max(1, (txs["date"].max() - txs["date"].min()).days / 30)
    monetary = txs["amount"].abs().mean()

    cat_counts = txs["category"].value_counts(normalize=True).to_dict()
    top_categories = txs["category"].value_counts().head(5).to_dict()

    return {
        "avg_monthly_spend": float(avg_monthly),
        "last_month_spend": float(last_month),
        "recency_days": int(recency_days),
        "tx_frequency_per_month": float(frequency),
        "avg_tx_amount": float(monetary),
        "category_distribution": cat_counts,
        "top_categories": top_categories
    }


# -------------------------------
# 🔹 Rule-based Recommendations
# -------------------------------
def recommend_from_features(features, debts_df, assets_df, goals_df):
    recs = []
    monthly = features.get("avg_monthly_spend", 0)
    liquid = assets_df["value"].sum() if not assets_df.empty else 0

    # Emergency fund
    if liquid < 3 * monthly:
        amount_needed = max(0, 3 * monthly - liquid)
        recs.append({
            "type": "emergency_fund",
            "priority": "High",
            "note": f"Build an emergency fund of ₹{amount_needed:.2f}. Save ₹{amount_needed/6:.2f}/month for 6 months."
        })

    # Spending trend
    if monthly > 5000:
        recs.append({
            "type": "budget_control",
            "priority": "Medium",
            "note": "Your spending seems high — try tracking daily expenses for better control."
        })

    # Goals advice
    for _, g in goals_df.iterrows():
        try:
            target = g.get("target_amount", 0)
            current = g.get("current_amount", 0)
            goal_name = g.get("goal_name", "Goal")
            td = pd.to_datetime(g.get("target_date"))
            months_left = max(1, (td.to_period("M") - pd.Timestamp.now().to_period("M")).n)
            recs.append({
                "type": "goal_funding",
                "priority": "Medium",
                "note": f"Save ₹{(target - current)/months_left:.2f}/month to achieve '{goal_name}'."
            })
        except Exception:
            continue

    return recs


# -------------------------------
# 🔹 AI Recommendations (Chatbot-style Gemini)
# -------------------------------
def generate_recommendations(user_data):
    """Generate personalized AI tips using the same method as chatbot."""
    try:
        # ✅ Use the same model as your working chatbot
        model = genai.GenerativeModel("gemini-2.5-flash")

        features = user_data.get("features", {})
        rules = user_data.get("rules", [])
        debts = user_data.get("debts", [])
        assets = user_data.get("assets", [])
        goals = user_data.get("goals", [])

        conversation = f"""
        User: Please act as my personal financial advisor. Here’s my data:
        - Features: {json.dumps(features, indent=2)}
        - Debts: {debts}
        - Assets: {assets}
        - Goals: {goals}
        - Rule-based notes: {rules}

        Assistant:
        1️⃣ Summarize my current financial health.
        2️⃣ Give 3 personalized actionable tips.
        3️⃣ Suggest 3 short-term and 3 long-term financial goals.
        Keep it concise, clear, and structured in markdown format.
        """

        response = model.generate_content(conversation)

        if response and hasattr(response, "text"):
            return response.text
        else:
            return "AI couldn’t generate personalized insights at this time."

    except Exception as e:
        print("💥 Gemini recommendation error:", e)
        return (
            "### AI-Powered Personalized Tips Unavailable 😔\n"
            "- Track your top 3 spending categories weekly.\n"
            "- Increase SIP or savings by 5% this month.\n"
            "- Build a 3-month emergency fund."
        )
