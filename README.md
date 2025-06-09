# StockPredictionApp 

StockPredictionApp is a web-based application that predicts stock prices for BIST 100 companies using historical data, technical indicators (RSI, MACD), sentiment analysis from news, and LSTM-based machine learning models. Built with Flask, the app allows users to analyze and predict stock prices, manage portfolios, and view historical predictions.

## Features
- Predict stock prices for 1 day, 5 days, and 1 month.
- Sentiment analysis of news articles using a Turkish BERT model.
- Technical indicators (RSI, MACD) and BIST 100 market trends.
- User authentication with registration and login.
- Portfolio management for tracked stocks.
- Historical prediction tracking.
- Responsive web interface with chart visualizations.

## Technologies
- **Backend**: Flask, Gunicorn
- **Machine Learning**: TensorFlow, LSTM, Transformers (BERT)
- **Data**: yfinance, NewsAPI, RSS feeds
- **Database**: SQLite
- **Other**: pandas, numpy, flask-bcrypt, flask-mail

## Prerequisites
- Python 3.8+
- Git
- Render (for deployment)

## Installation

### Local Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/berke/StockPredictionApp.git
   cd StockPredictionApp
