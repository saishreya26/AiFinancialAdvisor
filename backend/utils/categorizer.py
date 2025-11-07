# ------------------------------
# 🔹 Auto Category Detection
# ------------------------------
def auto_category(note: str) -> str:
    """Automatically categorize transactions based on description text."""
    if not note:
        return "Other"
    text = note.lower()

    # Transport
    if any(x in text for x in ["bmtc", "metro", "bus", "ola", "uber", "nuego", "rapido", "ka57", "ka01"]):
        return "Transport"

    # Food
    if any(x in text for x in ["bakery", "restaurant", "hotel", "food", "meals", "canteen", "hungerbox", "swiggy", "zomato"]):
        return "Food"

    # Shopping
    if any(x in text for x in ["amazon", "flipkart", "myntra", "ajio", "dmart", "bazaar", "store", "shop", "mart"]):
        return "Shopping"

    # Bills & Utilities
    if any(x in text for x in ["electricity", "gas", "water", "broadband", "wifi", "recharge", "mobile", "airtel", "jio"]):
        return "Bills & Utilities"

    # Entertainment
    if any(x in text for x in ["movie", "bookmyshow", "netflix", "prime", "hotstar", "spotify"]):
        return "Entertainment"

    # Healthcare
    if any(x in text for x in ["medical", "pharmacy", "hospital", "clinic"]):
        return "Healthcare"

    # Income / Received
    if "received" in text or "credited" in text:
        return "Income"

    # Transfers
    if "sent" in text or "transfer" in text:
        return "Transfer"

    return "Other"


# ------------------------------
# 🔹 Categorize Transaction Wrapper
# ------------------------------
def categorize_transaction(transaction):
    """Attach category and type fields to each transaction."""
    desc = transaction.get("description", "") or ""
    category = auto_category(desc)
    txn_type = "income" if category.lower() == "income" else "expense"

    transaction["category"] = category
    transaction["type"] = txn_type
    return transaction
