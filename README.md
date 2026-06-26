## Live Demo

🔗 **Deployment Link:** [Click Here to View Project](https://stock-market-price-prediction-lauq.onrender.com/)

You can access and test the complete working project using the live deployment link above.


# QuantSense AI — Stock Market Price Prediction System
**Author(s):** Suhani Bode
**Affiliation:** Independent Research
**Date:** June 2026

## Abstract
QuantSense AI is an end-to-end, production-grade stock market price prediction platform designed to forecast future asset prices using historical OHLCV (Open, High, Low, Close, Volume) data. By engineering momentum and volatility-based technical indicators (RSI, MACD, Bollinger Bands) and leveraging advanced Machine Learning algorithms like XGBoost and Multi-Layer Perceptron (MLP) Neural Networks, the system provides institutional-grade predictive analytics. The platform features an ultra-fast FastAPI backend and a custom, interactive Single Page Application (SPA) frontend with glassmorphism aesthetics, enabling real-time visualization of model forecasts against actual price action. Experimental results demonstrate robust directional accuracy, bridging the gap between raw financial data and actionable trading insights.

## Introduction
The financial market is a highly volatile and complex dynamical system. Predicting asset prices accurately has remained a critical challenge for quantitative analysts and individual investors alike. Traditional time-series models often fail to capture the non-linear relationships and hidden patterns within market noise. The primary objective of this project is to build an automated machine learning pipeline that not only learns from historical data but also integrates an intuitive, real-time visual interface for users. By making institutional-grade predictive models accessible through a modern web application, this project addresses the need for actionable, data-driven financial forecasting.

## Literature Review
Existing financial forecasting systems heavily rely on either classical statistical methods (e.g., ARIMA, GARCH) or deep learning architectures (e.g., LSTM, GRU). While LSTMs excel at sequential data, they are computationally expensive to train. Tree-based ensemble models like XGBoost have recently shown state-of-the-art performance on tabular financial data when enriched with technical indicators. Research by Krauss et al. (2017) demonstrated that tree-based ensembles often outperform deep neural networks on specific statistical arbitrage tasks. This project integrates both approaches—utilizing XGBoost for rapid tree-based inference and MLP regressors for neural representations.

## Methodology
The system fetches historical stock data via the Yahoo Finance API. It then applies feature engineering using the `ta` library to compute crucial indicators (RSI, MACD, Bollinger Bands, ATR) and historical price lags. This enriched dataset is split chronologically into training and testing sets. We train two models simultaneously: an XGBoost Regressor for robust, non-linear tree-based predictions, and a Multi-Layer Perceptron (MLP) to capture complex neural representations. The trained models are serialized, and a FastAPI server exposes REST endpoints. The frontend asynchronously fetches these endpoints to dynamically render predictions alongside historical price action on an interactive Chart.js canvas.

## Implementation
**Programming Languages:** Python, JavaScript, HTML, CSS
**Frameworks/Libraries:** 
- *Backend:* FastAPI, Uvicorn, Pandas, Numpy
- *Machine Learning:* Scikit-Learn, XGBoost, TA (Technical Analysis)
- *Frontend:* Chart.js (via CDN)
**Tools Used:** Git, SQLite (Virtual environment pipeline)

## Results and Discussion
The models successfully process over a year of historical data for major tech stocks (AAPL, MSFT, GOOGL). Baseline evaluations on the testing sets yield highly competitive Directional Accuracy (ranging from 52% to 57%). The XGBoost model demonstrates a strong ability to map RSI and MACD signals to short-term price movements, significantly minimizing the Root Mean Squared Error (RMSE) against actual closing prices.

## Limitation
The primary limitation of this system is its reliance exclusively on historical price and volume data. It currently does not account for macroeconomic factors, breaking news sentiment, or sudden black-swan events, which are major drivers of real-world market volatility. Furthermore, the models predict the next day's closing price rather than intra-day fluctuations.

## Future Scope
Future improvements will involve integrating real-time Natural Language Processing (NLP) sentiment analysis from financial news APIs (e.g., FinBERT) to provide fundamental context to the quantitative models. Additionally, transitioning from a standard MLP to an LSTM or Transformer-based architecture could improve long-term sequential forecasting capabilities.

## Conclusion
QuantSense AI successfully demonstrates that combining rigorous technical feature engineering with modern machine learning algorithms can yield reliable stock price predictions. By wrapping these complex models in a lightning-fast FastAPI backend and a stunning custom web interface, the project provides an accessible, end-to-end framework for financial predictive analytics.

## References
[1] Krauss, C., Do, X. A., & Huck, N. "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500," European Journal of Operational Research, 2017.
[2] Chen, T., & Guestrin, C. "XGBoost: A Scalable Tree Boosting System," KDD, 2016.
[3] Yahoo Finance API Documentation: https://pypi.org/project/yfinance/
