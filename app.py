import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pickle
import shap
import numpy as np
import os

# --- Page Config & Aesthetics ---
st.set_page_config(page_title="QuantSense AI", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for Premium Glassmorphism & Animations
st.markdown("""
    <style>
    /* Global styles and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 24px;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Metric animations */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.9; }
        50% { opacity: 1; text-shadow: 0 0 20px rgba(56, 189, 248, 0.4); }
        100% { opacity: 0.9; }
    }
    
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Header ---
st.markdown("""
<div class='glass-card' style='text-align: center;'>
    <h1 style='font-size: 3rem; margin-bottom: 0;'>🚀 QuantSense AI</h1>
    <p style='color: #94a3b8; font-size: 1.2rem;'>Institutional-Grade Predictive Analytics Platform</p>
</div>
""", unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.title("Configuration")
symbol = st.sidebar.selectbox("Select Asset", ["AAPL", "MSFT", "GOOGL"])
model_type = st.sidebar.radio("Select Model", ["XGBoost", "Neural Network (MLP)"])

# --- Load Data & Models ---
@st.cache_data
def load_data(sym):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', f'{sym}_processed.csv')
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        return df
    return None

@st.cache_resource
def load_models(sym):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xgb_path = os.path.join(base_dir, 'models', f'{sym}_xgb.pkl')
    mlp_path = os.path.join(base_dir, 'models', f'{sym}_mlp.pkl')
    metric_path = os.path.join(base_dir, 'data', f'{sym}_metrics.pkl')
    
    models = {}
    if os.path.exists(xgb_path):
        with open(xgb_path, 'rb') as f:
            models['XGBoost'] = pickle.load(f)
    if os.path.exists(mlp_path):
        with open(mlp_path, 'rb') as f:
            models['Neural Network (MLP)'] = pickle.load(f)
    if os.path.exists(metric_path):
        with open(metric_path, 'rb') as f:
            models['metrics'] = pickle.load(f)
            
    return models

df = load_data(symbol)
assets = load_models(symbol)

if df is not None and assets:
    st.markdown(f"<h2>{symbol} Market Overview</h2>", unsafe_allow_html=True)
    
    # Extract recent data
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    price_change = latest['close'] - prev['close']
    pct_change = (price_change / prev['close']) * 100
    
    # Top Metrics Layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='glass-card'>
            <p style='color: #94a3b8; margin: 0;'>Current Price</p>
            <div class='metric-value'>${latest['close']:.2f}</div>
            <p style='color: {"#22c55e" if price_change >= 0 else "#ef4444"}; margin: 0;'>
                {"+" if price_change >=0 else ""}{price_change:.2f} ({pct_change:.2f}%)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='glass-card'>
            <p style='color: #94a3b8; margin: 0;'>RSI (14)</p>
            <div class='metric-value' style='background: linear-gradient(90deg, #c084fc, #f472b6); -webkit-background-clip: text;'>{latest['rsi']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class='glass-card'>
            <p style='color: #94a3b8; margin: 0;'>Directional Accuracy</p>
            <div class='metric-value' style='background: linear-gradient(90deg, #34d399, #10b981); -webkit-background-clip: text;'>{assets.get('metrics', {}).get('da', 0):.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class='glass-card'>
            <p style='color: #94a3b8; margin: 0;'>Model RMSE</p>
            <div class='metric-value' style='background: linear-gradient(90deg, #fbbf24, #f59e0b); -webkit-background-clip: text;'>${assets.get('metrics', {}).get('rmse', 0):.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # Predictions
    model = assets.get(model_type)
    if model:
        features = ['open', 'high', 'low', 'close', 'volume', 'rsi', 'macd', 'bollinger_hband', 'bollinger_lband', 'atr'] + [f'close_lag_{i}' for i in range(1, 6)]
        X = df[features]
        df['prediction'] = model.predict(X)
        next_day_pred = df['prediction'].iloc[-1]
        pred_change = next_day_pred - latest['close']
        
        st.markdown(f"""
        <div class='glass-card' style='text-align: center; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3);'>
            <h3 style='margin-bottom: 5px;'>Next Day Forecast ({model_type})</h3>
            <div class='metric-value' style='font-size: 3rem;'>${next_day_pred:.2f}</div>
            <p style='color: {"#22c55e" if pred_change >= 0 else "#ef4444"}; font-size: 1.2rem; margin: 0;'>
                Predicted Change: {"+" if pred_change >=0 else ""}{pred_change:.2f}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Main Chart
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Price Action & AI Forecast")
        
        # Plotly Candlestick
        fig = go.Figure()
        
        # Slicing for better visibility (last 100 days)
        df_plot = df.iloc[-100:]
        
        fig.add_trace(go.Candlestick(
            x=df_plot['time'],
            open=df_plot['open'],
            high=df_plot['high'],
            low=df_plot['low'],
            close=df_plot['close'],
            name="OHLC",
            increasing_line_color='#22c55e',
            decreasing_line_color='#ef4444'
        ))
        
        # Add Predictions
        fig.add_trace(go.Scatter(
            x=df_plot['time'], 
            y=df_plot['prediction'],
            mode='lines', 
            name=f'Predicted Close ({model_type})',
            line=dict(color='#38bdf8', width=2, dash='dot')
        ))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Explainable AI (SHAP) - Only for XGBoost as it's a tree model
        if model_type == "XGBoost":
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.subheader("Explainable AI (Feature Importance)")
            st.write("Understand what drives the AI's predictions using SHAP values.")
            
            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                # Sample last 50 for speed
                X_sample = X.iloc[-50:]
                shap_values = explainer.shap_values(X_sample)
                
                # Plotly Bar Chart for Feature Importance
                mean_shap = np.abs(shap_values).mean(axis=0)
                shap_df = pd.DataFrame({"Feature": features, "Importance": mean_shap}).sort_values(by="Importance", ascending=True)
                
                fig_shap = go.Figure(go.Bar(
                    x=shap_df["Importance"],
                    y=shap_df["Feature"],
                    orientation='h',
                    marker=dict(color='#818cf8')
                ))
                fig_shap.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=400
                )
                st.plotly_chart(fig_shap, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("No data or models found. Please run `python quantsense_pipeline.py` first to generate the models and data files.")
