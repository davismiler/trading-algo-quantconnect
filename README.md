# TradingAlgo

An advanced machine learning-powered trading algorithm system that combines deep learning price prediction with reinforcement learning optimization for automated forex and CFD trading.

## Overview

This project implements a sophisticated trading system that:

- Uses neural networks to predict price movements based on historical OHLC data
- Employs technical indicators (Bollinger Bands, RSI, ADX) for signal confirmation
- Optimizes trading parameters using reinforcement learning (Deep Q-Network)
- Integrates with QuantConnect platform for backtesting and live trading
- Supports multiple trading pairs (XAUUSD, GBPUSD, EURGBP, etc.)

## Project Structure

- **`research.ipynb`** - Jupyter notebook for model development and experimentation
- **`forex.py`** - Main trading algorithm for forex pairs (GBPUSD)
- **`cfd.py`** - CFD trading algorithm optimized for XAUUSD (Gold)
- **`optimiser.py`** - Reinforcement learning agent for parameter optimization
- **`access_api.py`** - QuantConnect API integration for backtesting and data access
- **`eval_score_calculator.py`** - Performance evaluation and scoring system
- **`algorithm_evaluation.csv`** - Historical performance results across different trading pairs

## Key Features

### 1. Machine Learning Price Prediction

- **Neural Network Architecture**: Sequential model with Dense layers (30 → 20 → 1)
- **Input Features**: 30-minute rolling window of OHLC percentage changes
- **Prediction Target**: Binary classification (price up/down)
- **Training Data**: 2020-2022 historical market data

### 2. Technical Analysis Integration

- **Bollinger Bands**: Identify overbought/oversold conditions
- **RSI (Relative Strength Index)**: Momentum oscillator for trend confirmation
- **ADX (Average Directional Index)**: Measure trend strength

### 3. Risk Management

- **Stop Loss**: 2% risk per trade
- **Position Sizing**: Full capital allocation per trade
- **Exit Strategy**: Middle Bollinger Band reversion or stop loss trigger

### 4. Reinforcement Learning Optimization

- **Algorithm**: Deep Q-Network (DQN)
- **Environment**: Custom trading environment with parameter adjustment actions
- **Reward Function**: Based on performance score improvement
- **Optimization Target**: Technical indicator parameters (BB length, RSI length, ADX length)

### 5. Performance Evaluation

Comprehensive scoring system based on:

- **Sharpe Ratio** (25% weight)
- **Maximum Drawdown** (25% weight)
- **Annualized Return** (20% weight)
- **Volatility** (15% weight)
- **Cumulative Return** (8% weight)
- **Sortino Ratio** (3% weight)
- **Win Rate** (2% weight)
- **Average Trade Return** (2% weight)

## Trading Strategy

### Entry Conditions

**Long Position**:

- Price below lower Bollinger Band
- RSI < 30 (oversold)
- ADX > 20 (strong trend)
- ML model predicts "Up"

**Short Position**:

- Price above upper Bollinger Band
- RSI > 70 (overbought)
- ADX > 20 (strong trend)
- ML model predicts "Down"

### Exit Conditions

- Price reaches middle Bollinger Band (profit taking)
- Stop loss triggered (2% loss)
- End of trading session

## Results

Based on backtesting results in `algorithm_evaluation.csv`, the algorithm shows varying performance across different pairs:

- **XAUUSD (Gold)**: Best performing pair with -0.39 overall score
- **EURGBP**: -0.77 overall score
- **GBPUSD**: -2.76 overall score (poorest performer)

## Requirements

- Python 3.11+
- TensorFlow/Keras
- NumPy, Pandas
- PyTorch (for RL optimization)
- QuantConnect Platform access
- Matplotlib (for visualization)
- Tabulate (for result formatting)

## Usage

1. **Model Training**: Run `research.ipynb` to train the price prediction model
2. **Algorithm Deployment**: Use `forex.py` or `cfd.py` depending on the trading pair
3. **Parameter Optimization**: Execute `optimiser.py` to find optimal indicator parameters
4. **Performance Evaluation**: Use `eval_score_calculator.py` to assess results

## QuantConnect Integration

The system integrates with QuantConnect's cloud platform for:

- Historical data access
- Backtesting execution
- Live trading deployment
- Performance monitoring

API credentials and project configuration are managed through `access_api.py`.

## Risk Disclaimer

This is an experimental trading system for educational and research purposes. Past performance does not guarantee future results. Always use proper risk management and never risk more than you can afford to lose.
