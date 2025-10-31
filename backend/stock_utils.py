from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend


# Helper Functions
def fetch_stooq_data(ticker, start_date, end_date):
    """Fetch stock data from Stooq"""
    try:
        start = start_date.strftime('%Y%m%d')
        end = end_date.strftime('%Y%m%d')
        url = f"https://stooq.com/q/d/l/?s={ticker}&d1={start}&d2={end}&i=d"

        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                df.set_index('Date', inplace=True)
                return df
        return None
    except Exception as e:
        print(f"Error fetching stock data: {str(e)}")
        return None


def search_mutual_fund(fund_name):
    """Search for mutual fund scheme code"""
    try:
        url = "https://www.amfiindia.com/spages/NAVAll.txt"
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            lines = response.text.split('\n')
            schemes = []

            for line in lines:
                if ';' in line:
                    parts = line.split(';')
                    if len(parts) >= 5:
                        scheme_code = parts[0].strip()
                        scheme_name = parts[3].strip()
                        nav = parts[4].strip()

                        if fund_name.lower() in scheme_name.lower() and nav and nav != '':
                            schemes.append({
                                'code': scheme_code,
                                'name': scheme_name,
                                'nav': nav
                            })

            return schemes if schemes else None
        return None
    except Exception as e:
        print(f"Error searching mutual funds: {str(e)}")
        return None


def fetch_mutual_fund_data(scheme_code, start_date, end_date):
    """Fetch mutual fund NAV data"""
    try:
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        response = requests.get(url, timeout=15)

        if response.status_code == 200:
            data = response.json()

            if 'data' in data:
                nav_data = data['data']
                df = pd.DataFrame(nav_data)
                df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
                df['nav'] = pd.to_numeric(df['nav'], errors='coerce')

                df = df.rename(columns={'date': 'Date', 'nav': 'Close'})
                df = df[['Date', 'Close']].dropna()

                df = df[(df['Date'] >= pd.to_datetime(start_date)) &
                        (df['Date'] <= pd.to_datetime(end_date))]

                df = df.sort_values('Date')
                df.set_index('Date', inplace=True)

                df['Open'] = df['Close']
                df['High'] = df['Close']
                df['Low'] = df['Close']
                df['Volume'] = 0

                return df
        return None
    except Exception as e:
        print(f"Error fetching mutual fund data: {str(e)}")
        return None


def create_features(df):
    """Create technical indicators and features"""
    df = df.copy()

    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100

    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)

    if 'Volume' in df.columns:
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()

    df.dropna(inplace=True)
    return df


def prepare_data(df, target_col='Close', test_size=0.2):
    """Prepare data for training"""
    feature_cols = [col for col in df.columns if col not in [target_col]]

    X = df[feature_cols]
    y = df[target_col]

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y, X_train.index, X_test.index


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple optimized models"""
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=42, n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=7, min_samples_split=5,
            learning_rate=0.05, subsample=0.8, random_state=42
        ),
        'XGBoost (Enhanced)': GradientBoostingRegressor(
            n_estimators=300, max_depth=8, min_samples_split=4,
            learning_rate=0.03, subsample=0.85, random_state=42
        ),
        'Linear Regression': LinearRegression()
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)

        mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-10))) * 100
        accuracy_pct = max(0, 100 - mape)

        results[name] = {
            'model': model,
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mae),
            'r2': float(r2),
            'train_r2': float(train_r2),
            'mape': float(mape),
            'accuracy': float(accuracy_pct),
            'overfitting': float(train_r2 - r2)
        }
        predictions[name] = y_pred_test

    # Ensemble
    sorted_models = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    top_3_models = sorted_models[:3]
    weights = np.array([m[1]['r2'] for m in top_3_models])
    weights = weights / weights.sum()

    ensemble_pred = np.zeros_like(y_test)
    for (name, metrics), weight in zip(top_3_models, weights):
        ensemble_pred += predictions[name] * weight

    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_mape = np.mean(np.abs((y_test - ensemble_pred) / (y_test + 1e-10))) * 100

    results['Ensemble (Top 3)'] = {
        'model': 'ensemble',
        'mse': float(ensemble_mse),
        'rmse': float(np.sqrt(ensemble_mse)),
        'mae': float(ensemble_mae),
        'r2': float(ensemble_r2),
        'train_r2': None,
        'mape': float(ensemble_mape),
        'accuracy': float(max(0, 100 - ensemble_mape)),
        'overfitting': 0,
        'weights': {m[0]: float(w) for m, w in zip(top_3_models, weights)}
    }
    predictions['Ensemble (Top 3)'] = ensemble_pred

    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]

    return results, predictions, best_model_name


def calculate_performance_metrics(df, start_date, end_date):
    """Calculate performance metrics"""
    period_data = df.loc[start_date:end_date]

    if len(period_data) < 2:
        return None

    start_price = period_data['Close'].iloc[0]
    end_price = period_data['Close'].iloc[-1]
    returns = ((end_price - start_price) / start_price) * 100

    daily_returns = period_data['Close'].pct_change()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe = (daily_returns.mean() * 252 - 0.02) / (daily_returns.std() * np.sqrt(252))

    cumulative = (1 + daily_returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    return {
        'start_price': float(start_price),
        'end_price': float(end_price),
        'returns': float(returns),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'trading_days': int(len(period_data))
    }


def predict_future(models_dict, scaler_X, scaler_y, last_features, days, best_model_name, ensemble_weights=None):
    """Predict future prices"""
    predictions = []
    current_features = last_features.copy()

    for _ in range(days):
        scaled_features = scaler_X.transform([current_features])

        if best_model_name == 'Ensemble (Top 3)' and ensemble_weights:
            scaled_pred = 0
            for model_name, weight in ensemble_weights.items():
                model_obj = models_dict[model_name]['model']
                scaled_pred += model_obj.predict(scaled_features)[0] * weight
        else:
            model = models_dict[best_model_name]['model']
            scaled_pred = model.predict(scaled_features)[0]

        pred = scaler_y.inverse_transform([[scaled_pred]])[0][0]
        predictions.append(float(pred))

        for i in range(5, 1, -1):
            current_features[-(6 - i)] = current_features[-(7 - i)]
        current_features[-5] = pred

    return predictions