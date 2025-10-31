def calculate_investment_recommendations(current_amount, goal_amount, time_horizon, risk_tolerance):
    # Convert risk tolerance to a category
    if risk_tolerance <= 3:
        risk_category = "conservative"
    elif risk_tolerance <= 7:
        risk_category = "moderate"
    else:
        risk_category = "aggressive"

    # Base allocations based on risk profile and time horizon
    allocations = {
        "conservative": {
            "short": {"Cash/Fixed Deposits": 0.40, "Gold": 0.15, "Blue-chip Stocks": 0.10},
            "long": {"Cash/Fixed Deposits": 0.20, "Government Bonds": 0.30, "Gold": 0.20, "Index Funds": 0.20,
                     "Blue-chip Stocks": 0.10}
        },
        "moderate": {
            "short": {"Cash/Fixed Deposits": 0.30, "Government Bonds": 0.25, "Gold": 0.20, "Index Funds": 0.15,
                      "Blue-chip Stocks": 0.10},
            "long": {"Cash/Fixed Deposits": 0.15, "Government Bonds": 0.20, "Gold": 0.15, "Index Funds": 0.30,
                     "Diversified Mutual Funds": 0.20}
        },
        "aggressive": {
            "short": {"Cash/Fixed Deposits": 0.20, "Government Bonds": 0.15, "Gold": 0.15, "Index Funds": 0.30,
                      "Sector Funds": 0.20},
            "long": {"Cash/Fixed Deposits": 0.05, "Government Bonds": 0.10, "Gold": 0.10, "Index Funds": 0.25,
                     "Diversified Mutual Funds": 0.25, "Growth Stocks": 0.25}
        }
    }

    # Get recommendation based on risk category and time horizon
    recommended_allocation = allocations[risk_category][time_horizon]

    # Calculate actual amount for each asset class
    investment_recommendation = {}
    for asset, percentage in recommended_allocation.items():
        investment_recommendation[asset] = current_amount * percentage

    # Calculate monthly savings needed to reach the goal
    years = 2 if time_horizon == "short" else 10  # Assumption: short = 2 years, long = 10 years
    months = years * 12

    # Simple calculation for required monthly savings
    # This is a simplified formula that doesn't account for investment growth
    gap = goal_amount - current_amount
    monthly_savings = gap / months if months > 0 else 0

    return {
        "allocation": investment_recommendation,
        "monthly_savings": monthly_savings,
        "risk_category": risk_category,
        "time_horizon_years": years
    }


def generate_description(recommendation, risk_category, time_horizon):
    """
    Generate a description of the investment recommendation
    """
    risk_descriptions = {
        "conservative": "Your conservative risk profile suggests you prefer stability over high returns. This portfolio focuses on preserving capital with modest growth potential.",
        "moderate": "Your moderate risk approach balances growth with stability. This portfolio aims for steady growth while managing volatility.",
        "aggressive": "Your high risk tolerance allows for a growth-oriented portfolio. This allocation seeks higher returns while accepting greater short-term volatility."
    }

    time_descriptions = {
        "short": "For your short-term goal (under 5 years), we've weighted your portfolio toward more stable assets.",
        "long": "Your long-term horizon (5+ years) allows for a greater allocation to growth-oriented investments that can weather market fluctuations."
    }

    return f"{risk_descriptions[risk_category]} {time_descriptions[time_horizon]}"

