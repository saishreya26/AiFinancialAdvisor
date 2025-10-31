import os
import re
import requests
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------- DATA-ASG FETCHERS -------------------
def fetch_gold_price_in_inr():
    """Fetches raw gold price in INR per ounce"""
    url = "https://data-asg.goldprice.org/dbXRates/INR"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["items"][0]["xauPrice"])
    except Exception as e:
        print("⚠️ Error fetching gold price:", e)
        return None


def fetch_silver_price_in_inr():
    """Fetches raw silver price in INR per ounce"""
    url = "https://data-asg.goldprice.org/dbXRates/INR"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["items"][0]["xagPrice"])
    except Exception as e:
        print("⚠️ Error fetching silver price:", e)
        return None


# ------------------- CONVERSION HELPERS -------------------
def convert_to_per_gram(price_per_ounce):
    """Convert ounce to gram"""
    return price_per_ounce / 31.1035


def adjust_gold_to_market_rate(price_per_gram):
    """
    Adjust gold spot bullion price to approximate Indian retail.
    Usually 8% higher to match GoodReturns / local prices.
    """
    return price_per_gram * 1.08


def adjust_silver_to_market_rate(price_per_gram):
    """
    Adjust silver spot bullion price to approximate Indian retail.
    Silver usually trades around 10–15% above international spot.
    """
    return price_per_gram * 1.15


def get_price_for_carat(price_per_gram_24k, carat):
    carat_factor = {
        "24 Carat": 1.00,
        "22 Carat": 22 / 24,
        "18 Carat": 18 / 24
    }.get(carat, 1.00)
    return price_per_gram_24k * carat_factor


# ------------------- MAIN CONTROLLER -------------------
def get_live_metal_prices(city='Bangalore', carat='24 Carat'):
    """
    Fetch live gold and silver prices (with adjustments)
    """
    print(f"\n{'=' * 60}\nFetching live gold & silver prices for {city} ({carat})\n{'=' * 60}")

    gold_price_oz = fetch_gold_price_in_inr()
    silver_price_oz = fetch_silver_price_in_inr()

    gold_data, silver_data = {}, {}

    # GOLD SECTION
    if gold_price_oz:
        price_per_gram = convert_to_per_gram(gold_price_oz)
        adjusted_price = adjust_gold_to_market_rate(price_per_gram)
        gold_data = {
            "24 Carat": {"price": round(adjusted_price, 2), "change": "+₹0"},
            "22 Carat": {"price": round(get_price_for_carat(adjusted_price, "22 Carat"), 2), "change": "+₹0"},
            "18 Carat": {"price": round(get_price_for_carat(adjusted_price, "18 Carat"), 2), "change": "+₹0"}
        }

    # SILVER SECTION
    if silver_price_oz:
        price_per_gram = convert_to_per_gram(silver_price_oz)
        adjusted_price = adjust_silver_to_market_rate(price_per_gram)
        silver_data = {"price": round(adjusted_price, 2), "change": "+₹0"}

    # ✅ Fallbacks if API fails
    if not gold_data:
        print("❌ Could not fetch gold data — using fallback.")
        gold_data = {
            "24 Carat": {"price": 7500, "change": "+₹0"},
            "22 Carat": {"price": 6875, "change": "+₹0"},
            "18 Carat": {"price": 5625, "change": "+₹0"}
        }

    if not silver_data:
        print("❌ Could not fetch silver data — using fallback.")
        silver_data = {"price": 90, "change": "+₹0"}

    # Format final output
    selected_carat = carat if carat in gold_data else "24 Carat"
    gold_price = gold_data[selected_carat]["price"]

    response_data = {
        "gold": {
            "carat": selected_carat,
            "price_per_gram": gold_price,
            "price_per_10g": round(gold_price * 10, 2),
            "change": gold_data[selected_carat]["change"],
            "city": city,
            "updated": datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        },
        "silver": {
            "price_per_gram": silver_data["price"],
            "price_per_kg": round(silver_data["price"] * 1000, 2),
            "change": silver_data["change"],
            "updated": datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
        }
    }

    print(f"\n✅ Final Live Response:\n{response_data}")
    print(f"{'=' * 60}\n")
    return response_data


