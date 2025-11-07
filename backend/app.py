from email.message import EmailMessage
from bson import ObjectId
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import ssl
from flask import Blueprint, Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import random
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from news import fetch_gold_price_in_inr, fetch_silver_price_in_inr, convert_to_per_gram, get_live_metal_prices ,adjust_gold_to_market_rate, adjust_silver_to_market_rate,get_price_for_carat, get_live_metal_prices
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
import  pandas as pd
from stock_utils import (
    fetch_stooq_data, search_mutual_fund,
    fetch_mutual_fund_data, create_features,
    prepare_data, train_models, calculate_performance_metrics,predict_future
)
from video_data import video_sections
from financial_path import calculate_investment_recommendations, generate_description
from utils.transactions import parse_google_pay_text, parse_phonepe_pdf, format_date_safe,parse_uploaded_transactions
from utils.categorizer import categorize_transaction, auto_category
import smtplib, ssl
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from functools import wraps
from cryptography.fernet import Fernet
import os
from bson.objectid import ObjectId
from cryptography.fernet import Fernet
import google.generativeai as genai
from pathlib import Path
from bson.binary import Binary

# --- Load Environment Variables ---

# 1. Define the base directory (where app.py lives: /backend)
BASE_DIR = Path(__file__).resolve().parent

# 2. Assume .env is in the project root (one level up: /AiFinn)
dotenv_path = BASE_DIR.parent / '.env'

# Use this specific path to load the environment variables
if dotenv_path.exists():
    load_dotenv(dotenv_path)
    print(f"SUCCESS: Loaded .env from: {dotenv_path}")
else:
    # If this prints, the file isn't in the root either!
    raise FileNotFoundError(f"CRITICAL ERROR: .env file not found at expected root location: {dotenv_path}")

# Diagnostic check
if not os.getenv("GEMINI_API_KEY"):
    raise EnvironmentError("FATAL: GEMINI_API_KEY is still not set. Check content and variable name in .env.")
else:
    print("API Key check passed successfully. Importing modules...")
# -----------------------------------------------------------------

from recommendations import generate_recommendations, build_transaction_features, recommend_from_features

def safe_decrypt(value):
    """Decrypt encrypted Mongo fields safely (supports Binary, bytes, and str)."""
    if value is None:
        return ""
    try:
        # If value is Binary or bytes (MongoDB)
        if isinstance(value, (Binary, bytes)):
            return cipher_suite.decrypt(bytes(value)).decode()
        # If value is a string, try decrypting, else return as is
        elif isinstance(value, str):
            try:
                return cipher_suite.decrypt(value.encode()).decode()
            except Exception:
                return value
        return str(value)
    except Exception:
        return str(value)


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]  # "Bearer <token>"
        if not token:
            return jsonify({"message": "Token is missing"}), 403
        try:
            data = jwt.decode(token, os.environ.get(SECRET_KEY), algorithms=["HS256"])
            current_user = data["email"]
        except:
            return jsonify({"message": "Token is invalid"}), 403
        return f(current_user, *args, **kwargs)
    return decorated

smtp_server = "smtp.gmail.com"
port = 465  # for SSL
context = ssl.create_default_context()

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Allow cross-origin requests from frontend

load_dotenv()
bcrypt = Bcrypt(app)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["fin_advisor"]
users_collection = db["users"]
contents_collection = db["contents"]   # for literacy learning content
progress_collection = db["progress"]   # for tracking user progress
quizzes_collection = db["quizzes"]

otp_store = {}

# Email config
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

app.config["JWT_SECRET_KEY"] = "sdre56uiyhv456ygvc"
jwt = JWTManager(app)

# Get the key from the environment variable
fernet_key = os.environ.get("FERNET_KEY")

if not fernet_key:
    raise ValueError("FERNET_KEY environment variable not set")

# Initialize the Fernet cipher suite
cipher_suite = Fernet(fernet_key)

def encrypt_field(value):
    """
    Encrypt a string value using Fernet.
    """
    if value is None:
        return None
    return cipher_suite.encrypt(str(value).encode())

def decrypt_field(value):
    """
    Decrypt a Fernet-encrypted string value.
    """
    if value is None:
        return None
    try:
        return cipher_suite.decrypt(value.encode()).decode()
    except Exception:
        # In case it's not encrypted or corrupted
        return str(value)


def send_email(receiver_email, otp):
    try:
        msg = EmailMessage()
        msg.set_content(f"Your OTP is {otp}")
        msg["Subject"] = "OTP Verification"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = receiver_email

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        return True
    except Exception as e:
        print(f"Email error: {e}")
        return False

otp_store = {}

@app.route("/register", methods=["POST"])
def register():
    data = request.json
    name = data.get("name")
    phone = data.get("phone")
    email = data.get("email")
    password = data.get("password")

    if users_collection.find_one({"email": email}):
        return jsonify({"error": "Email already registered"}), 400

    otp = str(random.randint(100000, 999999))
    otp_store[email] = {
        "otp": otp,
        "expires_at": datetime.now() + timedelta(minutes=5),
        "name": name,
        "phone": phone,
        # ✅ Securely hash password with bcrypt
        "password": bcrypt.generate_password_hash(password).decode("utf-8")
    }

    if send_email(email, otp):
        return jsonify({"message": "OTP sent to email"}), 200
    else:
        return jsonify({"error": "Error sending OTP"}), 500


@app.route("/send_otp", methods=["POST"])
def send_otp():
    try:
        data = request.get_json()
        email = data.get("email")
        otp = str(random.randint(100000, 999999))
        otp_store[email] = otp

        msg = EmailMessage()
        msg.set_content(f"Your OTP is {otp}")
        msg["Subject"] = "OTP Verification"
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = email

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)

        return jsonify({"message": "OTP sent successfully"})
    except Exception as e:
        print(f"Error sending OTP: {e}")  # <— see real reason in console
        return jsonify({"message": "Failed to send OTP"}), 500

@app.route("/verify-otp", methods=["POST"])
def verify_otp():
    data = request.json
    email = data.get("email")
    otp = data.get("otp")

    if email not in otp_store:
        return jsonify({"error": "No OTP found for this email"}), 400

    stored_data = otp_store[email]

    if datetime.now() > stored_data["expires_at"]:
        del otp_store[email]
        return jsonify({"error": "OTP expired"}), 400

    if stored_data["otp"] != otp:
        return jsonify({"error": "Invalid OTP"}), 400

    # Save user to DB
    users_collection.insert_one({
        "name": stored_data["name"],
        "phone": stored_data["phone"],
        "email": email,
        "password": stored_data["password"],
        "goals": [],
        "debts": [],
        "assets": [],
        "transactions": []
    })

    del otp_store[email]
    return jsonify({"message": "User registered successfully"}), 200

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    user = users_collection.find_one({"email": data["email"]})

    if user and bcrypt.check_password_hash(user["password"], data["password"]):
        access_token = create_access_token(identity=str(user["_id"]), expires_delta=timedelta(hours=1))
        return jsonify({"message": "Login successful", "token": access_token, "email": user["email"]}), 200

    return jsonify({"message": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    # In a real system, you'd blacklist the token. For now, just return a success message.
    return jsonify({"message": "Logout successful"}), 200

@app.route("/forgot-password", methods=["POST"])
def forgot_password():
    data = request.get_json()
    email = data.get("email")

    user = users_collection.find_one({"email": email})
    if not user:
        return jsonify({"message": "User not found"}), 404

    new_password = data.get("new_password")
    hashed_pw = bcrypt.generate_password_hash(new_password).decode("utf-8")

    users_collection.update_one({"email": email}, {"$set": {"password": hashed_pw}})
    return jsonify({"message": "Password updated successfully"}), 200

@app.route("/mydata", methods=["GET"])
@jwt_required()
def get_mydata():
    user_id = get_jwt_identity()
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"password": 0})
    if user:
        user["_id"] = str(user["_id"])
        return jsonify(user), 200
    return jsonify({"message": "User not found"}), 404

@app.route("/api/recommend", methods=["POST"])
def recommend():
    data = request.json
    current_amount = data["current_amount"]
    goal_amount = data["goal_amount"]
    time_horizon = data["time_horizon"]
    risk_tolerance = data["risk_tolerance"]

    result = calculate_investment_recommendations(current_amount, goal_amount, time_horizon, risk_tolerance)
    result["description"] = generate_description(result, result["risk_category"], time_horizon)
    return jsonify(result)

@app.route("/api/financial-literacy-videos", methods=["GET"])
def get_video_sections():
    return jsonify(video_sections)


@app.route('/api/metal-prices', methods=['GET'])
def get_metal_prices():
    try:
        city = request.args.get('city', 'Bangalore')
        carat = request.args.get('carat', '24 Carat')

        print(f"\n{'=' * 60}")
        print(f"Fetching live metal prices for {city} ({carat})")
        print(f"{'=' * 60}")

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

        # ✅ Fallback if fetch fails
        if not gold_data:
            print("❌ Live gold fetch failed — using fallback.")
            gold_data = {
                "24 Carat": {"price": 7500, "change": "+₹0"},
                "22 Carat": {"price": 6875, "change": "+₹0"},
                "18 Carat": {"price": 5625, "change": "+₹0"}
            }

        if not silver_data:
            print("❌ Live silver fetch failed — using fallback.")
            silver_data = {"price": 90, "change": "+₹0"}

        # Ensure valid carat
        if carat not in gold_data:
            print(f"⚠️ Invalid carat '{carat}', defaulting to 24 Carat")
            carat = "24 Carat"

        selected_price = gold_data[carat]["price"]
        selected_change = gold_data[carat]["change"]

        # ✅ Response data format
        response_data = {
            "gold": {
                "carat": carat,
                "price_per_gram": selected_price,
                "price_per_10g": round(selected_price * 10, 2),
                "change": selected_change,
                "city": city,
                "updated": datetime.now().isoformat()
            },
            "silver": {
                "price_per_gram": silver_data["price"],
                "price_per_kg": round(silver_data["price"] * 1000, 2),
                "change": silver_data["change"],
                "updated": datetime.now().isoformat()
            }
        }

        print(f"\n✅ Returning Live Prices:")
        print(f"  Gold ({carat}): ₹{selected_price}/gram")
        print(f"  Silver: ₹{silver_data['price']}/gram")
        print(f"{'=' * 60}\n")

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Error in get_metal_prices: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500



news_api = os.environ.get("NEWS_API_KEY")

@app.route('/api/news', methods=['GET'])
def get_news():
    try:
        region = request.args.get('region', 'worldwide')

        print(f"Fetching news for region: {region}")

        if region == "india":
            url = f"https://newsapi.org/v2/top-headlines?country=in&category=business&apiKey={news_api}"
        else:
            url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={news_api}"

        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            data = response.json()
            articles = data.get('articles', [])[:10]
            print(f"Fetched {len(articles)} articles from NewsAPI")
            return jsonify({"articles": articles})

        # Fallback mock data
        print(f"NewsAPI failed with status {response.status_code}, using fallback data")
        fallback_articles = [
            {
                "title": "Global Markets Rally on Strong Economic Data",
                "description": "Stock markets worldwide surge as economic indicators exceed expectations.",
                "url": "#",
                "publishedAt": "2025-10-09T10:30:00Z",
                "source": {"name": "Financial Times"}
            },
            {
                "title": "Tech Stocks Lead Market Gains",
                "description": "Technology sector shows strong performance amid AI boom.",
                "url": "#",
                "publishedAt": "2025-10-09T09:15:00Z",
                "source": {"name": "Bloomberg"}
            },
            {
                "title": "RBI Maintains Policy Stance",
                "description": "Reserve Bank of India holds interest rates steady.",
                "url": "#",
                "publishedAt": "2025-10-09T08:00:00Z",
                "source": {"name": "Economic Times"}
            },
            {
                "title": "Gold Prices Surge Amid Market Uncertainty",
                "description": "Precious metals see increased demand as investors seek safe havens.",
                "url": "#",
                "publishedAt": "2025-10-09T07:45:00Z",
                "source": {"name": "Reuters"}
            },
            {
                "title": "Indian Rupee Strengthens Against Dollar",
                "description": "Currency markets show positive trends for emerging economies.",
                "url": "#",
                "publishedAt": "2025-10-09T07:00:00Z",
                "source": {"name": "Business Standard"}
            }
        ]

        return jsonify({"articles": fallback_articles})
    except Exception as e:
        print(f"Error in get_news: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stocks', methods=['GET'])
def get_stocks():
    try:
        print("Fetching stock data")
        # Mock data - in production, integrate with yfinance or NSE/BSE API
        stocks_data = [
            {
                "symbol": "RELIANCE.NS",
                "company": "Reliance Industries",
                "price": "2,456.30",
                "change": "+2.45",
                "volume": "12.5M"
            },
            {
                "symbol": "TCS.NS",
                "company": "Tata Consultancy Services",
                "price": "3,890.50",
                "change": "+1.89",
                "volume": "8.3M"
            },
            {
                "symbol": "INFY.NS",
                "company": "Infosys",
                "price": "1,678.20",
                "change": "+1.67",
                "volume": "15.2M"
            },
            {
                "symbol": "HDFCBANK.NS",
                "company": "HDFC Bank",
                "price": "1,543.80",
                "change": "+1.34",
                "volume": "10.1M"
            },
            {
                "symbol": "ICICIBANK.NS",
                "company": "ICICI Bank",
                "price": "1,234.50",
                "change": "+1.12",
                "volume": "9.8M"
            }
        ]

        return jsonify({"stocks": stocks_data})
    except Exception as e:
        print(f"Error in get_stocks: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/api/metal-prices?city=Bangalore&carat=24 Carat",
            "/api/news?region=worldwide",
            "/api/stocks",
            "/api/health"
        ]
    })
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message")
        # Extract the messages array for conversational context
        previous_messages = request.json.get("messages", [])

        # The Gemini API often prefers a simplified history structure or role-based Parts
        # For simplicity in the Flask wrapper, we'll continue using the combined text approach.
        conversation = ""
        for msg in previous_messages:
            # We skip the initial "Hi! Ask me anything." assistant message to keep context clean
            if msg.get("role") == "assistant" and msg.get("content").startswith("Hi!"):
                continue

            role = "User" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            conversation += f"{role}: {content}\n"

        # Add the current user message and prompt for the assistant's reply
        conversation += f"User: {user_message}\nAssistant:"

        # Use the correct, widely available model name
        # FIX: Changed "gemini-1.5-flash" to "gemini-2.5-flash"
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Use generate_content
        response = model.generate_content(conversation)

        reply = response.text if response.text else "I seem to be having trouble processing that request."

        return jsonify({
            "message": reply
        })

    except Exception as e:
        # Log the specific error for better debugging
        print(f"💥 Flask API Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/fire', methods=['POST'])
def fire_calculator():
    try:
        data = request.json
        print("Received data:", data)  # DEBUG
        # extract fields safely
        current_age = data.get('current_age')
        retirement_age = data.get('retirement_age')
        life_expectancy = data.get('life_expectancy')
        current_annual_expenses = data.get('current_annual_expenses')
        current_savings = data.get('current_savings')
        expected_inflation = data.get('expected_inflation', 0) / 100
        expected_return_pre = data.get('expected_return_pre', 0) / 100
        expected_return_post = data.get('expected_return_post', 0) / 100
        withdrawal_rate = data.get('withdrawal_rate', 0) / 100

        # if any required field is None -> return error
        required_fields = [current_age, retirement_age, life_expectancy, current_annual_expenses, current_savings]
        if any(field is None for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        years_to_retirement = retirement_age - current_age
        years_in_retirement = life_expectancy - retirement_age
        expenses_at_retirement = current_annual_expenses * ((1 + expected_inflation) ** years_to_retirement)
        fire_number_withdrawal = expenses_at_retirement * (1 / withdrawal_rate)
        inflation_adjusted_return = ((1 + expected_return_post) / (1 + expected_inflation)) - 1
        fire_number_corpus = 0
        if inflation_adjusted_return > 0:
            fire_number_corpus = expenses_at_retirement * (
                    1 - (1 / ((1 + inflation_adjusted_return) ** years_in_retirement))
            ) / inflation_adjusted_return
        fire_number = max(fire_number_withdrawal, fire_number_corpus)

        future_value_current_savings = current_savings * ((1 + expected_return_pre) ** years_to_retirement)
        additional_savings_needed = fire_number - future_value_current_savings
        monthly_savings = 0
        if additional_savings_needed > 0:
            r = expected_return_pre / 12
            n = years_to_retirement * 12
            monthly_savings = additional_savings_needed * r / ((1 + r) ** n - 1)

        return jsonify({
            "fire_number": round(fire_number),
            "monthly_savings": round(monthly_savings),
            "expenses_at_retirement": round(expenses_at_retirement)
        })

    except Exception as e:
        print("🔥 ERROR in /api/fire:", str(e))  # DEBUG
        return jsonify({"error": str(e)}), 500


@app.route('/api/emi', methods=['POST'])
def emi_calculator():
    data = request.json
    loan_amount = data['loan_amount']
    loan_term_years = data['loan_term_years']
    annual_interest_rate = data['interest_rate']
    payment_frequency = data['payment_frequency']

    freq_multiplier = {
        "Monthly": 1,
        "Quarterly": 3,
        "Half-Yearly": 6,
        "Yearly": 12
    }

    payments_per_year = 12 / freq_multiplier[payment_frequency]
    num_payments = loan_term_years * payments_per_year
    periodic_interest_rate = (annual_interest_rate / 100) / payments_per_year

    emi = loan_amount * periodic_interest_rate * (1 + periodic_interest_rate) ** num_payments / ((1 + periodic_interest_rate) ** num_payments - 1)
    total_payment = emi * num_payments
    total_interest = total_payment - loan_amount

    return jsonify({
        "emi": round(emi),
        "total_payment": round(total_payment),
        "total_interest": round(total_interest)
    })

@app.route('/api/sip', methods=['POST'])
def sip_calculator():
    data = request.json
    sip_amount = data['sip_amount']
    investment_period = data['investment_period']
    expected_return_rate = data['expected_return_rate'] / 100
    step_up_percentage = data['step_up_percentage'] / 100

    monthly_rate = expected_return_rate / 12
    months = investment_period * 12
    investment_value = 0
    total_invested = 0
    current_sip = sip_amount

    for month in range(1, months + 1):
        if month > 12 and month % 12 == 1 and step_up_percentage > 0:
            current_sip *= (1 + step_up_percentage)
        total_invested += current_sip
        investment_value = (investment_value + current_sip) * (1 + monthly_rate)

    wealth_gained = investment_value - total_invested

    return jsonify({
        "total_invested": round(total_invested),
        "total_value": round(investment_value),
        "wealth_gained": round(wealth_gained)
    })

@app.route('/api/swp', methods=['POST'])
def swp_calculator():
    data = request.json
    initial_investment = data['initial_investment']
    monthly_withdrawal = data['monthly_withdrawal']
    expected_return_rate_swp = data['expected_return_rate_swp'] / 100
    withdrawal_period = data['withdrawal_period']
    withdrawal_step_up = data['withdrawal_step_up'] / 100

    monthly_rate = expected_return_rate_swp / 12
    months = withdrawal_period * 12
    balance = initial_investment
    total_withdrawn = 0
    current_withdrawal = monthly_withdrawal
    exhausted_month = 0

    for month in range(1, months + 1):
        if month > 12 and month % 12 == 1 and withdrawal_step_up > 0:
            current_withdrawal *= (1 + withdrawal_step_up)
        interest = balance * monthly_rate
        balance += interest - current_withdrawal
        total_withdrawn += current_withdrawal
        if balance <= 0:
            exhausted_month = month
            balance = 0
            break

    total_interest = total_withdrawn + balance - initial_investment
    years_lasted = exhausted_month / 12 if exhausted_month else withdrawal_period

    return jsonify({
        "total_withdrawn": round(total_withdrawn),
        "final_balance": round(balance),
        "total_interest": round(total_interest),
        "corpus_exhausted": exhausted_month > 0,
        "years_lasted": round(years_lasted, 1)
    })


@app.route('/api/search-mutual-funds', methods=['POST'])
def search_mf():
    """Search for mutual funds"""
    data = request.json
    fund_name = data.get('fund_name', '')

    if len(fund_name) < 3:
        return jsonify({'error': 'Fund name must be at least 3 characters'}), 400

    schemes = search_mutual_fund(fund_name)

    if schemes:
        return jsonify({'success': True, 'schemes': schemes})
    else:
        return jsonify({'success': False, 'message': 'No matching funds found'}), 404


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint"""
    data = request.json
    data_type = data.get('type', 'stock')  # 'stock' or 'mutual_fund'

    try:
        if data_type == 'stock':
            ticker = data.get('ticker')
            start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
            end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')
            prediction_days = int(data.get('prediction_days', 30))

            df = fetch_stooq_data(ticker, start_date, end_date)
            currency_symbol = '$'

        else:  # mutual_fund
            scheme_code = data.get('scheme_code')
            scheme_name = data.get('scheme_name')
            start_date = datetime.strptime(data.get('start_date'), '%Y-%m-%d')
            end_date = datetime.strptime(data.get('end_date'), '%Y-%m-%d')
            prediction_days = int(data.get('prediction_days', 30))

            df = fetch_mutual_fund_data(scheme_code, start_date, end_date)
            currency_symbol = '₹'

        if df is None or len(df) < 50:
            return jsonify({'error': 'Insufficient data. Please try different dates or ticker.'}), 400

        # Performance metrics
        perf = calculate_performance_metrics(df, df.index[0], df.index[-1])

        # Create features and train models
        df_features = create_features(df)
        X_train, X_test, y_train, y_test, scaler_X, scaler_y, train_idx, test_idx = prepare_data(df_features)
        results, predictions, best_model_name = train_models(X_train, y_train, X_test, y_test)

        # Prepare model results for JSON
        model_results = {}
        for name, metrics in results.items():
            model_results[name] = {k: v for k, v in metrics.items() if k != 'model'}

        # Future predictions
        last_row_features = df_features.iloc[-1][[col for col in df_features.columns if col != 'Close']].values
        ensemble_weights = results['Ensemble (Top 3)'].get('weights') if best_model_name == 'Ensemble (Top 3)' else None

        future_preds = predict_future(
            results, scaler_X, scaler_y, last_row_features,
            prediction_days, best_model_name, ensemble_weights
        )

        # Prepare historical data
        historical_data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'close': df['Close'].tolist(),
        }

        # Prepare test predictions
        y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_original = scaler_y.inverse_transform(predictions[best_model_name].reshape(-1, 1)).ravel()

        test_predictions = {
            'dates': test_idx.strftime('%Y-%m-%d').tolist(),
            'actual': y_test_original.tolist(),
            'predicted': y_pred_original.tolist()
        }

        # Future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days, freq='D')

        # Technical indicators
        df_tech = create_features(df)
        technical_data = {
            'dates': df_tech.index[-100:].strftime('%Y-%m-%d').tolist(),
            'close': df_tech['Close'].iloc[-100:].tolist(),
            'ma_20': df_tech['MA_20'].iloc[-100:].tolist(),
            'bb_upper': df_tech['BB_upper'].iloc[-100:].tolist(),
            'bb_lower': df_tech['BB_lower'].iloc[-100:].tolist(),
            'rsi': df_tech['RSI'].iloc[-100:].tolist(),
            'macd': df_tech['MACD'].iloc[-100:].tolist(),
            'current_rsi': float(df_tech['RSI'].iloc[-1]),
            'current_macd': float(df_tech['MACD'].iloc[-1]),
            'current_ma_20': float(df_tech['MA_20'].iloc[-1]),
            'current_volatility': float(df_tech['Volatility'].iloc[-1])
        }

        response = {
            'success': True,
            'performance': perf,
            'model_results': model_results,
            'best_model': best_model_name,
            'historical_data': historical_data,
            'test_predictions': test_predictions,
            'future_predictions': {
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'values': future_preds
            },
            'technical_data': technical_data,
            'currency_symbol': currency_symbol,
            'data_points': len(df)
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

# GOALS
@app.route("/users/goals", methods=["POST"])
@jwt_required()
def add_goal():
    user_id = get_jwt_identity()
    data = request.json or {}

    try:
        # Convert to float, then to string for encryption
        target_amount_str = str(float(data.get("target_amount") or 0))
        current_amount_str = str(float(data.get("current_amount") or 0))
    except ValueError:
        return jsonify({"error": "Invalid amount format"}), 400

    goal_name = data.get("goal_name")
    if not goal_name or float(target_amount_str) <= 0:
        return jsonify({"error": "Invalid input"}), 400

    # Encrypt the sensitive data
    encrypted_target_amount = cipher_suite.encrypt(target_amount_str.encode())
    encrypted_current_amount = cipher_suite.encrypt(current_amount_str.encode())

    goal = {
        "goal_name": goal_name,
        "target_amount": encrypted_target_amount,
        "current_amount": encrypted_current_amount,
        "start_date": data.get("start_date") or "",
        "target_date": data.get("target_date") or "",
        "priority": data.get("priority") or "Medium",
        "status": "Not Started",
        "created_at": datetime.now().isoformat()
    }

    result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"goals": goal}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found or goal not added"}), 404

    return jsonify({"message": f"Goal '{goal_name}' added successfully"}), 201

@app.route("/users/goals/<goal_name>", methods=["PUT"])
@jwt_required()
def update_goal(goal_name):
    user_id = get_jwt_identity()
    data = request.json or {}

    # Fetch user
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"goals": 1})
    if not user:
        return jsonify({"error": "User not found"}), 404

    # Find the goal in user's goals array
    goals = user.get("goals", [])
    goal_to_update = next((g for g in goals if g["goal_name"] == goal_name), None)

    if not goal_to_update:
        return jsonify({"error": "Goal not found"}), 404

    # Prepare updates
    update_fields = {}

    # Encrypt current_amount if provided
    if "current_amount" in data:
        try:
            current_amount_str = str(float(data["current_amount"]))
            encrypted_current_amount = cipher_suite.encrypt(current_amount_str.encode())
            update_fields["goals.$.current_amount"] = encrypted_current_amount
        except ValueError:
            return jsonify({"error": "Invalid current_amount format"}), 400

    # Update status if provided
    if "status" in data:
        update_fields["goals.$.status"] = data["status"]

    if not update_fields:
        return jsonify({"error": "No valid fields to update"}), 400

    # Update the specific goal using positional operator
    result = users_collection.update_one(
        {"_id": ObjectId(user_id), "goals.goal_name": goal_name},
        {"$set": update_fields}
    )

    if result.modified_count == 0:
        return jsonify({"error": "Goal update failed"}), 500

    return jsonify({"message": f"Goal '{goal_name}' updated successfully"}), 200


# ---------------- Get Goals ----------------
@app.route("/users/goals", methods=["GET"])
@jwt_required()
def get_goals():
    user_id = get_jwt_identity()
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "goals": 1})

    goals = user.get("goals", [])

    # Decrypt each goal's sensitive data
    for goal in goals:
        try:
            goal["target_amount"] = float(cipher_suite.decrypt(goal["target_amount"]).decode())
            goal["current_amount"] = float(cipher_suite.decrypt(goal["current_amount"]).decode())
        except Exception as e:
            # Handle potential decryption errors (e.g., corrupted data)
            print(f"Decryption error: {e}")
            goal["target_amount"] = None
            goal["current_amount"] = None

    return jsonify(goals)

# ---------------- Add Debt ----------------
@app.route("/users/debts", methods=["POST"])
@jwt_required()
def add_debt():
    user_id = get_jwt_identity()
    data = request.json or {}

    debt_name = data.get("name")
    if not debt_name:
        return jsonify({"error": "Debt name is required"}), 400

    try:
        encrypted_amount = encrypt_field(float(data.get("amount", 0)))
        encrypted_interest = encrypt_field(float(data.get("interest", 0)))
    except ValueError:
        return jsonify({"error": "Invalid number format"}), 400

    debt = {
        "name": debt_name,                    # optional: can encrypt too
        "type": data.get("type", "Other"),
        "amount": encrypted_amount,
        "interest_rate": encrypted_interest,
        "created_at": encrypt_field(datetime.now().isoformat())
    }

    result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"debts": debt}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found or debt not added"}), 404

    return jsonify({"message": f"Debt '{debt_name}' added successfully"}), 201

# ---------------- Get Debts ----------------
@app.route("/users/debts", methods=["GET"])
@jwt_required()
def get_debts():
    user_id = get_jwt_identity()
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "debts": 1})

    debts = user.get("debts", [])
    # Optionally, you can decrypt here if you want to return readable values
    for d in debts:
        try:
            d["amount"] = float(cipher_suite.decrypt(d["amount"]).decode())
            d["interest_rate"] = float(cipher_suite.decrypt(d["interest_rate"]).decode())
            d["created_at"] = cipher_suite.decrypt(d["created_at"]).decode()
        except Exception:
            pass  # skip if field not encrypted

    return jsonify(debts)

# ---------------- Add Asset ----------------
@app.route("/users/assets", methods=["POST"])
@jwt_required()
def add_asset():
    user_id = get_jwt_identity()
    data = request.json or {}

    asset_name = data.get("name")
    if not asset_name:
        return jsonify({"error": "Asset name is required"}), 400

    try:
        encrypted_value = encrypt_field(float(data.get("value", 0)))
    except ValueError:
        return jsonify({"error": "Invalid value format"}), 400

    asset = {
        "name": asset_name,                   # optional: encrypt if sensitive
        "value": encrypted_value,
        "created_at": encrypt_field(datetime.now().isoformat())
    }

    result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"assets": asset}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found or asset not added"}), 404

    return jsonify({"message": f"Asset '{asset_name}' added successfully"}), 201

# ---------------- Get Assets ----------------
@app.route("/users/assets", methods=["GET"])
@jwt_required()
def get_assets():
    user_id = get_jwt_identity()
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "assets": 1})

    assets = user.get("assets", [])
    # Decrypt the encrypted fields
    for a in assets:
        try:
            a["value"] = float(cipher_suite.decrypt(a["value"]).decode())
            a["created_at"] = cipher_suite.decrypt(a["created_at"]).decode()
        except Exception:
            pass  # skip if field not encrypted

    return jsonify(assets)

# ---------------- Add Transaction ----------------
@app.route("/users/transactions", methods=["POST"])
@jwt_required()
def add_transaction():
    user_id = get_jwt_identity()
    data = request.json or {}

    try:
        amount = float(data.get("amount", 0))
        if amount <= 0:
            return jsonify({"error": "Invalid amount"}), 400
    except ValueError:
        return jsonify({"error": "Invalid amount format"}), 400

    # Default values
    note = data.get("note", "")
    category = data.get("category") or auto_category(note)
    tx_type = data.get("type", "expense").lower()

    try:
        encrypted_amount = encrypt_field(amount)
        encrypted_date = encrypt_field(data.get("date", datetime.now().isoformat()))
        encrypted_note = encrypt_field(note)
        encrypted_category = encrypt_field(category)
        encrypted_type = encrypt_field(tx_type)
    except Exception as e:
        print("⚠️ Encryption error:", e)
        return jsonify({"error": "Encryption failed"}), 500

    transaction = {
        "amount": encrypted_amount,
        "category": encrypted_category,
        "type": encrypted_type,
        "date": encrypted_date,
        "note": encrypted_note
    }

    result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"transactions": transaction}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"message": "Transaction added successfully"}), 201


# ---------------- Add Investment ----------------
@app.route("/users/investments", methods=["POST"])
@jwt_required()
def add_investment():
    user_id = get_jwt_identity()
    data = request.json or {}

    # Validate required fields
    if not data.get("category") or not data.get("name") or float(data.get("amount", 0)) <= 0:
        return jsonify({"error": "Invalid or missing fields"}), 400

    try:
        encrypted_amount = encrypt_field(float(data.get("amount", 0)))
        encrypted_returns = encrypt_field(float(data.get("returns", 0)))
        encrypted_date = encrypt_field(data.get("date", datetime.now().isoformat()))
        encrypted_name = encrypt_field(data.get("name"))
    except Exception as e:
        return jsonify({"error": "Invalid data format"}), 400

    investment = {
        "category": data.get("category"),  # SIP or Stock (optional to encrypt)
        "name": encrypted_name,
        "amount": encrypted_amount,
        "returns": encrypted_returns,
        "date": encrypted_date,
        "created_at": encrypt_field(datetime.now().isoformat())
    }

    result = users_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$push": {"investments": investment}}
    )

    if result.modified_count == 0:
        return jsonify({"error": "User not found or investment not added"}), 404

    return jsonify({"message": f"Investment '{data.get('name')}' added successfully"}), 201


# ---------------- Get Investments ----------------
@app.route("/users/investments", methods=["GET"])
@jwt_required()
def get_investments():
    user_id = get_jwt_identity()
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "investments": 1})

    investments = user.get("investments", [])
    for inv in investments:
        try:
            inv["name"] = cipher_suite.decrypt(inv["name"]).decode()
            inv["amount"] = float(cipher_suite.decrypt(inv["amount"]).decode())
            inv["returns"] = float(cipher_suite.decrypt(inv["returns"]).decode())
            inv["date"] = cipher_suite.decrypt(inv["date"]).decode()
            inv["created_at"] = cipher_suite.decrypt(inv["created_at"]).decode()
        except Exception:
            pass  # skip any decryption errors silently

    return jsonify(investments)


# ---------------- Get Transactions ----------------

@app.route("/users/transactions", methods=["GET"])
@jwt_required()
def get_transactions():
    user_id = get_jwt_identity()
    transactions = fetch_user_transactions(user_id)

    # 🧹 Filter out invalid or missing fields before sending to frontend
    cleaned = []
    for t in transactions:
        try:
            amount = float(t.get("amount", 0))
            if not (0 < amount < 1e7):
                continue  # skip unrealistic values

            date_str = t.get("date", "").strip()
            if not date_str or "Invalid" in date_str or len(date_str) < 4:
                continue  # skip invalid dates

            # Ensure consistent output
            cleaned.append({
                "date": date_str,
                "amount": amount,
                "category": t.get("category", "Other"),
                "note": t.get("note", t.get("description", "")),
                "type": t.get("type", "N/A")
            })
        except Exception as e:
            print("⚠️ Transaction filter error:", e)
            continue

    return jsonify(cleaned), 200


# ---------------- Upload File ----------------
@app.route("/users/upload", methods=["POST"])
@jwt_required()
def upload_file():
    user_id = get_jwt_identity()

    if "file" not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message": "No selected file"}), 400

    filename = file.filename.lower()

    # 🧠 Parse the file
    if filename.endswith(".pdf"):
        transactions = parse_phonepe_pdf(file)
    else:
        file_content = file.read()
        transactions = parse_uploaded_transactions(file_content)

    stored_transactions = []
    income_total = 0
    expense_total = 0
    category_summary = {}

    for t in transactions:
        try:
            desc = t.get("description", "") or t.get("note", "")
            amount_val = float(t.get("amount", 0))

            if amount_val <= 0:
                continue

            # --- 🧠 Auto Categorize ---
            category = auto_category(desc)

            # --- 💰 Type Detection ---
            if category.lower() == "income":
                tx_type = "income"
                income_total += amount_val
            else:
                tx_type = "expense"
                expense_total += amount_val

            print("🔍 NOTE TEXT:", desc)
            print("➡ CATEGORY:", auto_category(desc))

            # --- 📊 Category Summary (for chart) ---
            category_summary[category] = category_summary.get(category, 0) + amount_val

            # --- 🔐 Encrypt before storing ---
            encrypted_amount = encrypt_field(amount_val)
            encrypted_date = encrypt_field(t.get("date", datetime.now().isoformat()))
            encrypted_note = encrypt_field(desc)
            encrypted_category = encrypt_field(category)
            encrypted_type = encrypt_field(tx_type)

            transaction = {
                "amount": encrypted_amount,
                "category": encrypted_category,
                "type": encrypted_type,
                "date": encrypted_date,
                "note": encrypted_note
            }

            users_collection.update_one(
                {"_id": ObjectId(str(user_id))},
                {"$push": {"transactions": transaction}}
            )

            stored_transactions.append({
                "amount": amount_val,
                "date": t.get("date", datetime.now().isoformat()),
                "category": category,
                "type": tx_type,
                "note": desc
            })

        except Exception as e:
            print("⚠️ Error saving transaction:", e)
            continue

    # --- 📊 Summary Data for Charts ---
    summary = {
        "income": round(income_total, 2),
        "expense": round(expense_total, 2),
        "balance": round(income_total - expense_total, 2),
        "categories": category_summary
    }

    return jsonify({
        "message": f"{len(stored_transactions)} transactions uploaded successfully ✅",
        "transactions": stored_transactions,
        "summary": summary
    }), 200



#fetch user's details
def fetch_user_transactions(user_id):
    """Fetch and decrypt all user transactions safely."""
    user = users_collection.find_one({"_id": ObjectId(str(user_id))})
    if not user or "transactions" not in user:
        return []

    decrypted_transactions = []

    for txn in user["transactions"]:
        try:
            date = safe_decrypt(txn.get("date"))
            amount_raw = safe_decrypt(txn.get("amount"))
            note = safe_decrypt(txn.get("note"))
            category = safe_decrypt(txn.get("category", "Other"))
            tx_type = safe_decrypt(txn.get("type", "expense"))

            # ✅ Clean and validate amount
            amount_str = str(amount_raw).replace(",", "").strip()
            # skip invalid or crazy values
            if not amount_str.replace(".", "", 1).isdigit():
                print(f"⚠️ Skipping invalid amount: {amount_str}")
                continue

            amount = float(amount_str)
            if amount <= 0 or amount > 1e7:
                continue

            decrypted_transactions.append({
                "date": date,
                "amount": amount,
                "note": note,
                "category": category or "Other",
                "type": tx_type or "expense"
            })

        except Exception as e:
            print("⚠️ Decrypt error:", e)
            continue

    print("🔥 Decrypted Transactions:", decrypted_transactions)
    return decrypted_transactions



def fetch_assets(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "assets": 1})
    assets = user.get("assets", [])
    for a in assets:
        try:
            a["value"] = float(cipher_suite.decrypt(a["value"]).decode())
            a["created_at"] = cipher_suite.decrypt(a["created_at"]).decode()
        except Exception:
            pass
    return assets

def fetch_debts(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "debts": 1})
    debts = user.get("debts", [])
    for d in debts:
        try:
            d["amount"] = float(cipher_suite.decrypt(d["amount"]).decode())
            d["interest_rate"] = float(cipher_suite.decrypt(d["interest_rate"]).decode())
            d["created_at"] = cipher_suite.decrypt(d["created_at"]).decode()
        except Exception:
            pass
    return debts

def fetch_goals(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)}, {"_id": 0, "goals": 1})
    goals = user.get("goals", [])
    for g in goals:
        try:
            g["target_amount"] = float(cipher_suite.decrypt(g["target_amount"]).decode())
            g["current_amount"] = float(cipher_suite.decrypt(g["current_amount"]).decode())
        except Exception:
            g["target_amount"] = None
            g["current_amount"] = None
    return goals


@app.route("/users/recommendations", methods=["GET"])
@jwt_required()
def get_recommendations():
    user_id = get_jwt_identity()

    transactions = pd.DataFrame(fetch_user_transactions(user_id))
    assets = pd.DataFrame(fetch_assets(user_id))
    debts = pd.DataFrame(fetch_debts(user_id))
    goals = pd.DataFrame(fetch_goals(user_id))

    if not transactions.empty and "date" in transactions.columns:
        transactions["date"] = pd.to_datetime(transactions["date"], errors="coerce")
        transactions = transactions.dropna(subset=["date"])

    features = build_transaction_features(transactions)
    rules_recs = recommend_from_features(features, debts, assets, goals)

    ai_tips = generate_recommendations({
        "features": features,
        "rules": rules_recs,
        "debts": debts.to_dict(orient="records"),
        "assets": assets.to_dict(orient="records"),
        "goals": goals.to_dict(orient="records")
    })

    return jsonify({
        "rules_recommendations": rules_recs,
        "ai_tips": ai_tips,
        "analysis": features
    })



def get_user_data(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    return (
        user.get("transactions", []),
        user.get("assets", []),
        user.get("debts", []),
        user.get("goals", [])
    )

# ---------------- Dashboard Summary (Optional) ----------------
@app.route("/users/dashboard_summary", methods=["GET"])
@jwt_required()
def dashboard_summary():
    """Return user data, transactions, and spending summary."""
    user_id = get_jwt_identity()
    user = users_collection.find_one({"_id": ObjectId(str(user_id))}, {"name": 1, "_id": 0})

    # Fetch all related data
    transactions = fetch_user_transactions(user_id)
    assets = fetch_assets(user_id)
    debts = fetch_debts(user_id)
    goals = fetch_goals(user_id)

    # ✅ Compute category-wise spending (only expenses)
    category_summary = {}
    for txn in transactions:
        try:
            amt = txn.get("amount", 0)
            if not isinstance(amt, (int, float)) or amt <= 0 or amt > 1e7:
                continue  # skip invalids

            tx_type = str(txn.get("type", "expense")).lower()
            if tx_type in ["income", "credit"]:
                continue  # skip incomes

            category = str(txn.get("category", "Other")).title()
            category_summary[category] = category_summary.get(category, 0) + amt
        except Exception as e:
            print("⚠️ Summary aggregation error:", e)
            continue

    # Sorted list for frontend
    spending_summary = [
        {"category": k, "total_spent": v}
        for k, v in sorted(category_summary.items(), key=lambda x: x[1], reverse=True)
    ]

    print("✅ Spending Summary:", spending_summary)

    # ✅ Sanitizer for JSON serialization (avoids bytes issues)
    def sanitize(obj):
        if isinstance(obj, (bytes, Binary)):
            return obj.decode(errors="ignore")
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(i) for i in obj]
        return obj

    response = {
        "name": user.get("name", "User") if user else "User",
        "transactions": sanitize(transactions),
        "assets": sanitize(assets),
        "debts": sanitize(debts),
        "goals": sanitize(goals),
        "spending_summary": sanitize(spending_summary)
    }

    return jsonify(response), 200





#1234567890-=
@app.route("/content", methods=["GET"])
def get_content():
    difficulty = request.args.get("difficulty")
    query = {}
    if difficulty:
        query["difficulty"] = difficulty  # match Beginner/Intermediate

    contents = list(contents_collection.find(query))
    for c in contents:
        c["_id"] = str(c["_id"])
    return jsonify(contents), 200



@app.route("/content/<id>", methods=["GET"])
def get_content_by_id(id):
    """Fetch a single topic by its _id."""
    try:
        # Try fetching either by ObjectId or string ID (topic1, topic2, etc.)
        topic = contents_collection.find_one({"_id": id}) or contents_collection.find_one({"_id": ObjectId(id)})
        if not topic:
            return jsonify({"error": "Content not found"}), 404

        topic["_id"] = str(topic["_id"])
        return jsonify(topic), 200
    except Exception as e:
        print("Error:", e)
        return jsonify({"error": "Invalid ID format"}), 400


# ----------------------------
# Progress Routes
# ----------------------------
@app.route("/progress", methods=["GET"])
@jwt_required()
def get_progress():
    user_id = get_jwt_identity()
    progress = progress_collection.find_one({"user_id": user_id}, {"_id": 0, "completed": 1})
    if progress and "completed" in progress:
        return jsonify(progress["completed"]), 200
    return jsonify([]), 200


@app.route("/progress/update", methods=["POST"])
@jwt_required()
def update_progress():
    try:
        user_id = get_jwt_identity()  # Extract user_id from JWT token
        data = request.get_json()

        completed_item = data.get("completed")  # The content ID (string)

        if not completed_item:
            return jsonify({"error": "Missing 'completed' field"}), 400

        # Use $addToSet to prevent duplicate entries for the same topic
        progress_collection.update_one(
            {"user_id": user_id},
            {"$addToSet": {"completed": completed_item}},
            upsert=True
        )

        return jsonify({
            "message": f"Progress updated successfully for {completed_item}",
            "user_id": user_id
        }), 200

    except Exception as e:
        print("Error updating progress:", e)
        return jsonify({"error": "Internal server error"}), 500


@app.route("/quiz/<topic_id>", methods=["GET"])
@jwt_required()
def get_quiz(topic_id):
    quiz = quizzes_collection.find_one({"topic_id": topic_id}, {"_id": 0})
    if not quiz:
        return jsonify({"msg": "Quiz not found"}), 404

    # Find topic difficulty automatically
    content = contents_collection.find_one({"_id": topic_id}, {"difficulty": 1, "_id": 0})
    quiz["difficulty"] = content["difficulty"] if content else "Unknown"

    return jsonify(quiz), 200

@app.route("/quiz", methods=["GET"])
@jwt_required(optional=True)
def get_quizzes_by_difficulty():
    difficulty = request.args.get("difficulty")
    query = {}
    if difficulty:
        query["difficulty"] = difficulty
    quizzes = list(quizzes_collection.find(query, {"_id": 0}))
    return jsonify(quizzes), 200


@app.route("/quiz/<topic_id>/submit", methods=["POST"])
@jwt_required()
def submit_quiz(topic_id):
    answers = request.json.get("answers")  # list of integers
    quiz = quizzes_collection.find_one({"topic_id": topic_id})
    if not quiz:
        return jsonify({"error": "Quiz not found"}), 404

    score = sum([1 for i, q in enumerate(quiz["questions"]) if q["answer"] == answers[i]])
    user_id = get_jwt_identity()
    progress_collection.update_one(
        {"user_id": user_id},
        {"$push": {"completed": {"topic_id": topic_id, "quiz_score": score}}},
        upsert=True
    )
    return jsonify({"score": score}), 200


@app.route("/api/modules/<id>", methods=["GET"])
def get_module_by_id(id):
    # If your _id is a string like "topic3"
    module = contents_collection.find_one({"_id": id})
    if module:
        module["_id"] = str(module["_id"])  # ensure JSON serializable
        return jsonify(module)
    return jsonify({"error": "Module not found"}), 404

@app.route("/api/modules", methods=["GET"])
def get_modules():
    difficulty = request.args.get("difficulty")
    query = {"difficulty": difficulty} if difficulty else {}
    cursor = contents_collection.find(query)
    modules = []
    for m in cursor:
        m["_id"] = str(m["_id"])
        modules.append(m)
    return jsonify(modules)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
