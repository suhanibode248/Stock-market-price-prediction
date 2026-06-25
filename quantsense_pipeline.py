import yfinance as yf
import pandas as pd
import numpy as np
import ta
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_data(symbol="AAPL", period="2y"):
    """Fetch OHLCV data from Yahoo Finance"""
    logging.info(f"Fetching data for {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period)
    df.reset_index(inplace=True)
    df.rename(columns={"Date": "time", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
    # Ensure timezone unaware for simplicity
    df['time'] = df['time'].dt.tz_localize(None)
    return df

def feature_engineering(df):
    """Add technical indicators and features"""
    logging.info("Engineering features...")
    # Add indicators
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    df['macd'] = ta.trend.MACD(close=df['close']).macd()
    df['bollinger_hband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['bollinger_lband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    
    # Target variable: next day's close price
    df['target'] = df['close'].shift(-1)
    
    # Lag features
    for i in range(1, 6):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        
    df.dropna(inplace=True)
    return df

def directional_accuracy(y_true, y_pred, y_prev):
    """Calculate Directional Accuracy"""
    actual_dir = np.sign(y_true - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    return np.mean(actual_dir == pred_dir) * 100

def train_models(df, symbol):
    """Train XGBoost and MLPRegressor models"""
    logging.info(f"Training models for {symbol}...")
    features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr'] + [f'close_lag_{i}' for i in range(1, 6)]
    
    X = df[features]
    y = df['target']
    
    # Chronological split
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # 1. Train XGBoost
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # 2. Train Neural Network (MLP)
    mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp_model.fit(X_train, y_train)
    
    # Evaluate XGBoost
    y_pred_xgb = xgb_model.predict(X_test)
    mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
    
    # Directional accuracy
    y_prev_test = X_test['close'].values
    da_xgb = directional_accuracy(y_test.values, y_pred_xgb, y_prev_test)
    
    logging.info(f"XGBoost - MAE: {mae_xgb:.2f}, RMSE: {rmse_xgb:.2f}, DA: {da_xgb:.2f}%")
    
    # Save models and data
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    with open(f'models/{symbol}_xgb.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    with open(f'models/{symbol}_mlp.pkl', 'wb') as f:
        pickle.dump(mlp_model, f)
        
    df.to_csv(f'data/{symbol}_processed.csv', index=False)
    
    # Save metrics
    metrics = {
        'mae': mae_xgb,
        'rmse': rmse_xgb,
        'da': da_xgb
    }
    with open(f'data/{symbol}_metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

def run_pipeline(symbols=["AAPL", "MSFT", "GOOGL"]):
    for symbol in symbols:
        df = fetch_data(symbol)
        df = feature_engineering(df)
        train_models(df, symbol)
    logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_pipeline()
