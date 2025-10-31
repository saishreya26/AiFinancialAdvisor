def auto_category(description):
    """
    Simple keyword-based auto-categorization.
    """
    desc = description.lower()
    if "grocery" in desc or "supermarket" in desc:
        return "Groceries"
    elif "uber" in desc or "ola" in desc:
        return "Transport"
    elif "netflix" in desc or "prime" in desc:
        return "Entertainment"
    elif "restaurant" in desc or "food" in desc:
        return "Dining"
    return "Other"


def categorize_transaction(transaction):
    """
    Assign category to a transaction.
    """
    transaction["category"] = auto_category(transaction["description"])
    return transaction
