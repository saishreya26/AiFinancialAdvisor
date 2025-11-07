import re
import pdfplumber
from datetime import datetime
from dateutil import parser as date_parser

# ------------------------------
# 🔹 Safe Date Formatter
# ------------------------------
def format_date_safe(date_str):
    """Safely convert date strings into YYYY-MM-DD format."""
    if not date_str:
        return None
    try:
        dt = date_parser.parse(date_str, fuzzy=True, dayfirst=False)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


# ------------------------------
# 🔹 Google Pay Text Parser
# ------------------------------
def parse_google_pay_text(text):
    """Parse raw Google Pay statement text into structured data."""
    transactions = []
    pattern = re.compile(
        r"(?P<type>Paid|Sent|Received)\s*₹(?P<amount>[\d,]+\.?\d*)"
        r"(?:\s*to\s*(?P<desc>[A-Za-z0-9 &._-]+))?",
        re.IGNORECASE,
    )

    for match in pattern.finditer(text):
        txn_type = match.group("type").strip().capitalize()
        desc = (match.group("desc") or "").strip() or "N/A"
        try:
            amount = float(match.group("amount").replace(",", ""))
        except ValueError:
            amount = 0.0

        # Try to capture nearby date context
        date_match = re.search(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}", text
        )
        date_fmt = format_date_safe(date_match.group(0)) if date_match else datetime.now().strftime("%Y-%m-%d")

        transactions.append(
            {
                "date": date_fmt,
                "description": f"{txn_type} ₹{amount} {desc}",
                "amount": amount,
                "type": txn_type.upper(),
            }
        )

    return transactions


# ------------------------------
# 🔹 Generic Uploaded File Parser (CSV / TXT)
# ------------------------------
import re
from datetime import datetime

def parse_uploaded_transactions(file_content: bytes):
    """Parse text transactions from Google Pay or text exports."""
    text = file_content.decode("utf-8", errors="ignore")

    # Split by "Google Pay" since each block starts with it
    blocks = text.split("Google Pay")
    transactions = []

    for block in blocks:
        if not block.strip():
            continue

        # --- Extract amount ---
        amount_match = re.search(r"[₹â‚¹]\s*([\d,]+\.?\d*)", block)
        amount = float(amount_match.group(1).replace(",", "")) if amount_match else 0.0

        # --- Extract date ---
        date_match = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s*\d{4}", block)
        date = date_match.group(0) if date_match else datetime.now().strftime("%b %d, %Y")

        # --- Extract transaction direction ---
        if "received" in block.lower():
            note = "Received money"
        elif "paid" in block.lower():
            # Try to get recipient
            to_match = re.search(r"to\s+([A-Za-z0-9\s&.]+)", block)
            note = f"Paid to {to_match.group(1).strip()}" if to_match else "Paid"
        elif "sent" in block.lower():
            note = "Sent money"
        else:
            note = "Transaction"

        transactions.append({
            "amount": amount,
            "date": date,
            "description": note
        })

    return transactions



# ------------------------------
# 🔹 PhonePe PDF Parser
# ------------------------------
def parse_phonepe_pdf(file_content):
    """Parse PhonePe PDF statements into structured transactions."""
    transactions = []
    with pdfplumber.open(file_content) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue

            pattern = re.compile(
                r"(?P<date>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4})"
                r".*?(?P<type>DEBIT|CREDIT)\s*₹(?P<amount>[\d,]+(?:\.\d+)?)"
                r"(?:\s*(?:Paid to|Received from)?\s*(?P<desc>.*?))?(?=\n|$)",
                re.DOTALL,
            )

            for match in pattern.finditer(text):
                date_raw = match.group("date").strip()
                txn_type = match.group("type").strip()
                amount_str = match.group("amount").replace(",", "")
                desc = (match.group("desc") or "").strip() or "N/A"

                try:
                    amount = float(amount_str)
                except ValueError:
                    amount = 0.0

                date_fmt = format_date_safe(date_raw)
                transactions.append(
                    {
                        "date": date_fmt or date_raw,
                        "description": desc,
                        "amount": amount,
                        "type": txn_type,
                    }
                )

    # Filter invalid entries
    return [
        t
        for t in transactions
        if t["date"] and isinstance(t["amount"], (int, float)) and 0 < t["amount"] < 1e7
    ]


