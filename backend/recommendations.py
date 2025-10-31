import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import ollama

# --- Feature Engineering ---
def build_transaction_features(txs: pd.DataFrame):
    """Generate features from transactions DataFrame."""
    if txs.empty:
        return {}

    txs = txs.copy()
    txs["date"] = pd.to_datetime(txs["date"], errors="coerce")  # ensure datetime
    txs = txs.dropna(subset=["date"])
    if txs.empty:
        return {}

    # Monthly spend
    txs["month"] = txs["date"].dt.to_period("M")
    monthly = txs.groupby("month")["amount"].sum().rename("monthly_spend").reset_index()
    avg_monthly = monthly["monthly_spend"].mean()
    last_month = monthly.iloc[-1]["monthly_spend"] if len(monthly) else 0.0

    # RFM-like: Recency, Frequency, Monetary
    recency_days = (pd.Timestamp.now() - txs["date"].max()).days
    frequency = txs.shape[0] / max(1, (txs["date"].max() - txs["date"].min()).days/30)
    monetary = txs["amount"].abs().mean()

    # Category distributions
    cat_counts = txs["category"].value_counts(normalize=True).to_dict()
    top_categories = txs["category"].value_counts().head(5).to_dict()

    # Trend of monthly spend
    monthly_float = monthly.copy()
    monthly_float["x"] = range(len(monthly_float))
    trend = 0.0
    if len(monthly_float) > 1:
        lr = LinearRegression()
        lr.fit(monthly_float[["x"]], monthly_float["monthly_spend"])
        trend = float(lr.coef_[0])

    # Recurring patterns
    recurring = txs.groupby(["category", "note"]).filter(lambda g: len(g) >= 3)
    recurring_summary = recurring.groupby(["category", "note"])["amount"].agg(["count", "mean"]).reset_index().to_dict("records")

    return {
        "avg_monthly_spend": float(avg_monthly),
        "last_month_spend": float(last_month),
        "recency_days": int(recency_days),
        "tx_frequency_per_month": float(frequency),
        "avg_tx_amount": float(monetary),
        "category_distribution": cat_counts,
        "top_categories": top_categories,
        "spend_trend_per_month": float(trend),
        "recurring_patterns": recurring_summary
    }

# --- Segmentation (Optional) ---
def segment_user_features(feature_dicts):
    df = pd.DataFrame(feature_dicts).fillna(0)
    X = df[["avg_monthly_spend","tx_frequency_per_month","spend_trend_per_month","recency_days"]]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42).fit(Xs)
    df["segment"] = km.labels_
    return df, km, scaler

# --- Rule-based recommendations ---
def recommend_from_features(features, debts_df, assets_df, goals_df):
    recs = []
    monthly = features.get("avg_monthly_spend", 0)
    liquid = assets_df["value"].sum() if not assets_df.empty else 0

    # Emergency fund
    if liquid < 3 * monthly:
        amount_needed = max(0, 3 * monthly - liquid)
        recs.append({
            "type":"emergency_fund",
            "priority":"High",
            "note": f"Build emergency fund of {amount_needed:.2f}. Save ~{amount_needed/6:.2f}/month to reach in 6 months."
        })

    # High-interest debt
    if not debts_df.empty:
        max_ir = debts_df["interest_rate"].max()
        high_debt = debts_df[debts_df["interest_rate"] == max_ir].iloc[0].to_dict()
        if max_ir > 0.12:
            recs.append({
                "type":"debt_payoff",
                "priority":"High",
                "note": f"High-interest debt '{high_debt['name']}' at {max_ir*100:.1f}% — consider paying faster."
            })

    # Rising spend trend
    if features.get("spend_trend_per_month", 0) > 50:
        recs.append({
            "type":"budget",
            "priority":"Medium",
            "note": "Spending is trending up. Try capping discretionary spend by 10% this month."
        })

    # Goal-based suggestions
    for _, g in goals_df.iterrows():
        target = g.get("target_amount", 0)
        current = g.get("current_amount", 0)
        try:
            td = pd.to_datetime(g.get("target_date"))
            months_left = max(1, (td.to_period("M") - pd.Timestamp.now().to_period("M")).n)
            needed = max(0, target - current)
            recs.append({
                "type":"goal_funding",
                "priority":"Medium",
                "note": f"To reach '{g.get('goal_name')}' in {months_left} months, save {needed/months_left:.2f}/month."
            })
        except Exception:
            pass
    return recs

def generate_recommendations(user_data):
    """
    Use Ollama (local LLM) to generate personalized recommendations
    based on user financial profile.
    """
    try:
        features = user_data.get("features", {})
        rules = user_data.get("rules", [])
        debts = user_data.get("debts", [])
        assets = user_data.get("assets", [])
        goals = user_data.get("goals", [])

        prompt = f"""
        You are an AI Financial Advisor.
        Analyze the following user data and provide personalized recommendations.

        --- USER FINANCIAL OVERVIEW ---
        📊 Features:
        Average Monthly Spend: {features.get("avg_monthly_spend", 0):.2f}
        Last Month Spend: {features.get("last_month_spend", 0):.2f}
        Spend Trend: {features.get("spend_trend_per_month", 0):.2f}
        Recency of Last Transaction (days): {features.get("recency_days", 0)}
        Transaction Frequency (per month): {features.get("tx_frequency_per_month", 0):.2f}

        --- CATEGORY DISTRIBUTION ---
        {features.get("category_distribution", {})}

        --- DEBTS ---
        {debts if debts else "No debts recorded."}

        --- ASSETS ---
        {assets if assets else "No assets recorded."}

        --- GOALS ---
        {goals if goals else "No goals recorded."}

        --- RULE-BASED RECOMMENDATIONS ---
        {rules}

        🧩 TASK:
        1. Summarize this user’s current financial health in plain English.
        2. Give 3 highly personalized actionable recommendations.
        3. Suggest 3 short-term and 3 long-term financial goals.
        4. Keep the tone friendly but professional.

        Provide the response in **clear markdown** with sections like:
        ## Summary
        ## Personalized Recommendations
        ## Short-term Goals
        ## Long-term Goals
        """

        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are an expert financial advisor helping users plan smarter."},
                {"role": "user", "content": prompt}
            ]
        )

        # Safely extract LLM output
        if "message" in response and "content" in response["message"]:
            return response["message"]["content"]
        elif "messages" in response and response["messages"]:
            return response["messages"][-1].get("content", "")
        else:
            return "AI could not generate recommendations."

    except Exception as e:
        print("Ollama generation error:", e)
        return "AI tips unavailable."
