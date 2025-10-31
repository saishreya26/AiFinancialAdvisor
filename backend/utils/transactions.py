import re

def parse_google_pay_text(text):
    """
    Parse raw Google Pay transaction text into structured data.
    """
    transactions = []
    for line in text.split("\n"):
        match = re.match(r"(.+) - (.+) - ₹(\d+)", line)
        if match:
            transactions.append({
                "date": match.group(1).strip(),
                "description": match.group(2).strip(),
                "amount": float(match.group(3))
            })
    return transactions


import re

def parse_uploaded_transactions(file_content):
    """
    Parse uploaded transaction file (CSV, TXT, etc.).
    """
    transactions = []
    for line in file_content.decode("utf-8", errors="ignore").split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]

        # Find the first numeric/₹ field for amount
        amount = None
        for p in parts:
            # remove currency symbols
            num = re.sub(r"[^\d.]", "", p)
            if num:
                try:
                    # look for something that can be float and is probably amount >0
                    f = float(num)
                    if f > 0:
                        amount = f
                        break
                except ValueError:
                    continue

        if amount is None:
            continue  # skip line if no amount found

        # just take first 2 as date/desc
        date = parts[0] if len(parts) > 0 else ""
        desc = parts[1] if len(parts) > 1 else ""

        transactions.append({
            "date": date,
            "description": desc,
            "amount": amount
        })

    return transactions
