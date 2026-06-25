from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import pickle
import os

app = FastAPI()

# Mount the static directory to serve HTML, CSS, JS
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/data/{symbol}")
def get_data(symbol: str, model_type: str = "xgb"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'data', f'{symbol.upper()}_processed.csv')
    model_path = os.path.join(base_dir, 'models', f'{symbol.upper()}_{model_type}.pkl')
    metrics_path = os.path.join(base_dir, 'data', f'{symbol.upper()}_metrics.pkl')
    
    if not os.path.exists(data_path):
        return {"error": "Data not found"}
        
    df = pd.read_csv(data_path)
    
    # Generate predictions if model exists
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr'] + [f'close_lag_{i}' for i in range(1, 6)]
        X = df[features]
        df['prediction'] = model.predict(X)
    else:
        df['prediction'] = None
        
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)
            
    # Send only the last 100 days for chart rendering performance
    df_recent = df.iloc[-100:].copy()
    
    return {
        "dates": df_recent['time'].tolist(),
        "open": df_recent['open'].tolist(),
        "high": df_recent['high'].tolist(),
        "low": df_recent['low'].tolist(),
        "close": df_recent['close'].tolist(),
        "prediction": df_recent['prediction'].tolist(),
        "latest_rsi": float(round(df.iloc[-1]['rsi'], 2)),
        "latest_close": float(round(df.iloc[-1]['close'], 2)),
        "prev_close": float(round(df.iloc[-2]['close'], 2)),
        "next_pred": float(round(df.iloc[-1]['prediction'], 2)) if 'prediction' in df else None,
        "metrics": {k: float(v) for k, v in metrics.items()} if metrics else {}
    }
