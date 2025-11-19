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
- **`base_trading_algorithm.py`** - Base class for trading algorithms (reduces code duplication)
- **`forex.py`** - Main trading algorithm for forex pairs (GBPUSD)
- **`cfd.py`** - CFD trading algorithm optimized for XAUUSD (Gold)
- **`optimiser.py`** - Reinforcement learning agent for parameter optimization
- **`access_api.py`** - QuantConnect API integration for backtesting and data access
- **`eval_score_calculator.py`** - Performance evaluation and scoring system
- **`algorithm_evaluation.csv`** - Historical performance results across different trading pairs
- **`.env.example`** - Template for environment variables (copy to `.env` and fill in credentials)
- **`requirements.txt`** - Python package dependencies

## Key Features

### 1. Machine Learning Price Prediction

- **Neural Network Architecture**: Improved Sequential model with Dense layers (64 → 32 → 1)
- **Regularization**: L2 regularization and Dropout (0.3) to prevent overfitting
- **Training Features**: Batch normalization, early stopping, and learning rate scheduling
- **Input Features**: 30-minute rolling window of OHLC percentage changes
- **Prediction Target**: Binary classification (price up/down)
- **Training Data**: 2020-2022 historical market data
- **Data Split**: 60% training, 20% validation, 20% test

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

### Python Version Compatibility

**Important:** TensorFlow does not yet support Python 3.14 (as of 2024). TensorFlow currently supports Python 3.9-3.12.

**Recommended Setup:**
- **For local development:** Use Python 3.11 or 3.12 for full package compatibility
- **For QuantConnect:** Python version is managed by the platform (TensorFlow/Keras are pre-installed)
- **If using Python 3.14:** Core packages work, but TensorFlow must be used on QuantConnect platform only

### Package Requirements

- **Core:** NumPy, Pandas, Matplotlib, Tabulate
- **Machine Learning:** TensorFlow/Keras (provided by QuantConnect platform)
- **Reinforcement Learning:** PyTorch, Gym (for local optimization)
- **API:** Requests, python-dotenv
- **Platform:** QuantConnect Platform access

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TradingAlgoQuantConnect
   ```

2. **Install dependencies**
   
   **If using Python 3.11 or 3.12:**
   ```bash
   pip install -r requirements-py311.txt
   ```
   
   **If using Python 3.14:**
   ```bash
   pip install -r requirements-minimal.txt
   # Note: TensorFlow will be provided by QuantConnect platform
   ```
   
   **Or install core packages manually:**
   ```bash
   pip install numpy pandas matplotlib tabulate requests python-dotenv
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your QuantConnect credentials:
   ```
   QC_USER_ID=your_user_id
   QC_API_TOKEN=your_api_token
   QC_PROJECT_ID=your_project_id
   QC_PAIR_NAME=XAUUSD
   ```

4. **Train the ML model** (on QuantConnect platform)
   - Open `research.ipynb` in QuantConnect Research environment
   - Run all cells to train and save the model

5. **Deploy trading algorithms**
   - Upload `forex.py` or `cfd.py` to QuantConnect
   - Ensure the trained model is saved in ObjectStore
   - Configure parameters (bb_length, rsi_length, adx_length) if using optimization

## Usage

1. **Model Training**: Run `research.ipynb` in QuantConnect Research to train the price prediction model
2. **Algorithm Deployment**: Use `forex.py` or `cfd.py` depending on the trading pair (both inherit from `base_trading_algorithm.py`)
3. **Parameter Optimization**: Execute `optimiser.py` locally (requires `.env` configuration) to find optimal indicator parameters
4. **Performance Evaluation**: Use `eval_score_calculator.py` to assess results

## Recent Improvements

### Code Quality
- ✅ Moved hardcoded credentials to `.env` file for security
- ✅ Fixed typos (`.Value` → `.value`) in trading algorithms
- ✅ Refactored code to reduce duplication using base class
- ✅ Added comprehensive error handling and validation
- ✅ Improved API error handling with proper exception management

### Model Architecture
- ✅ Added validation set (60/20/20 split)
- ✅ Implemented L2 regularization and dropout
- ✅ Added batch normalization for training stability
- ✅ Implemented early stopping and learning rate scheduling
- ✅ Enhanced evaluation metrics (precision, recall)
- ✅ Fixed train/test split bug in research notebook

### Code Structure
- ✅ Created `BaseTradingAlgorithm` class to eliminate code duplication
- ✅ Improved error handling throughout codebase
- ✅ Added type hints and documentation
- ✅ Created `requirements.txt` for dependency management

## QuantConnect Integration

The system integrates with QuantConnect's cloud platform for:

- Historical data access
- Backtesting execution
- Live trading deployment
- Performance monitoring

API credentials and project configuration are managed through `access_api.py`.

## Risk Disclaimer

This is an experimental trading system for educational and research purposes. Past performance does not guarantee future results. Always use proper risk management and never risk more than you can afford to lose.
