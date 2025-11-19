# region imports
from AlgorithmImports import *
from tensorflow.keras.models import Sequential
import json
import numpy as np
import pandas as pd
# endregion


class BaseTradingAlgorithm(QCAlgorithm):
    """
    Base class for trading algorithms with ML prediction and technical indicators.
    Provides common functionality for forex and CFD trading strategies.
    """
    
    def initialize(self):
        """Initialize the algorithm with common setup."""
        self.set_start_date(2022, 1, 1)
        self.set_cash(100000)

        # Load ML model if available
        self.model = self._load_model()
        if self.model is None:
            self.Debug("Warning: ML model not found. Trading will proceed without ML predictions.")
            self.use_ml = False
        else:
            self.use_ml = True

        self.rolling_window = RollingWindow[QuoteBar](40)

        # Get optimized parameters or use defaults
        self.bb_length = self.get_parameter("bb_length", 20)
        self.rsi_length = self.get_parameter("rsi_length", 20)
        self.adx_length = self.get_parameter("adx_length", 20)

        # Initialize symbol and indicators (to be set by subclasses)
        self.symbol = None
        self.bb = None
        self.rsi = None
        self.adx = None

        # Risk management
        self.exit_price = None
        self.percentage_risk = 0.02
        self.is_trading_enabled = False

        # Create chart for visualization
        self._setup_chart()
        
        self.set_benchmark("SPY")

    def _load_model(self):
        """Load the ML model from ObjectStore if available."""
        model_key = 'forex_price_predictor'
        try:
            if self.ObjectStore.ContainsKey(model_key):
                model_str = self.ObjectStore.Read(model_key)
                config = json.loads(model_str)['config']
                return Sequential.from_config(config)
        except Exception as e:
            self.Debug(f"Error loading model: {e}")
        return None

    def _setup_chart(self):
        """Setup the trading chart for visualization."""
        stock_plot = Chart("Trade Plot")
        stock_plot.add_series(Series("Buy", SeriesType.SCATTER, "$", 
            Color.Green, ScatterMarkerSymbol.TRIANGLE))
        stock_plot.add_series(Series("Sell", SeriesType.SCATTER, "$", 
            Color.Red, ScatterMarkerSymbol.TRIANGLE_DOWN))
        stock_plot.add_series(Series("Liquidate", SeriesType.SCATTER, "$", 
            Color.Blue, ScatterMarkerSymbol.DIAMOND))
        self.add_chart(stock_plot)

    def setup_symbol_and_indicators(self, symbol_name, resolution, market, symbol_type="forex"):
        """
        Setup the trading symbol and technical indicators.
        
        Args:
            symbol_name: Name of the symbol (e.g., "GBPUSD", "XAUUSD")
            resolution: Data resolution (e.g., Resolution.MINUTE)
            market: Market identifier (e.g., Market.OANDA)
            symbol_type: Type of symbol ("forex" or "cfd")
        """
        try:
            if symbol_type.lower() == "forex":
                self.symbol = self.add_forex(symbol_name, resolution, market).symbol
            elif symbol_type.lower() == "cfd":
                self.symbol = self.add_cfd(symbol_name, resolution, market).symbol
            else:
                raise ValueError(f"Unknown symbol type: {symbol_type}")

            # Initialize technical indicators
            self.bb = self.BB(self.symbol, self.bb_length, 2)
            self.rsi = self.RSI(self.symbol, self.rsi_length)
            self.adx = self.ADX(self.symbol, self.adx_length)

            # Setup data consolidation
            self.consolidate(symbol_name, timedelta(minutes=30), self.on_30_data)
            self.consolidate(symbol_name, Resolution.MINUTE, self.on_minute_data)

            # Setup trading schedule
            self._setup_trading_schedule()
        except Exception as e:
            self.Debug(f"Error setting up symbol and indicators: {e}")
            raise

    def _setup_trading_schedule(self):
        """Setup trading schedule (enable/disable trading at specific times)."""
        # Enable trading 40 mins after market open
        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.after_market_open(self.symbol, 40),
            self.enable_trading
        )

        # Disable trading 5 mins before market closes
        self.Schedule.On(
            self.DateRules.EveryDay(self.symbol),
            self.TimeRules.before_market_close(self.symbol, 5),
            self.disable_trading
        )

    def on_minute_data(self, bar):
        """Add minute data to the rolling window."""
        if bar is not None:
            self.rolling_window.add(bar)

    def on_30_data(self, bar):
        """Handle 30-minute consolidated data for trading decisions."""
        if not self._is_ready():
            return

        price = bar.close
        self._plot_indicators(price)

        if not self.portfolio.invested:
            self._check_entry_signals(price)
        else:
            self._check_exit_signals(price)

    def _is_ready(self):
        """Check if all indicators and conditions are ready for trading."""
        if not self.is_trading_enabled:
            return False
        if self.bb is None or not self.bb.is_ready:
            return False
        if self.rsi is None or not self.rsi.is_ready:
            return False
        if self.adx is None or not self.adx.is_ready:
            return False
        if self.symbol is None:
            return False
        return True

    def _plot_indicators(self, price):
        """Plot price and indicator values."""
        try:
            self.plot("Trade Plot", "Price", price)
            self.plot("Trade Plot", "MiddleBand", self.bb.middle_band.current.value)
            self.plot("Trade Plot", "UpperBand", self.bb.upper_band.current.value)
            self.plot("Trade Plot", "LowerBand", self.bb.lower_band.current.value)
        except Exception as e:
            self.Debug(f"Error plotting indicators: {e}")

    def _check_entry_signals(self, price):
        """Check for entry signals and execute trades."""
        try:
            # Long entry conditions
            long_conditions = (
                self.bb.lower_band.current.value > price and
                self.rsi.current.value < 30 and
                self.adx.current.value > 20
            )
            
            # Short entry conditions
            short_conditions = (
                self.bb.upper_band.current.value < price and
                self.rsi.current.value > 70 and
                self.adx.current.value > 20
            )

            # Add ML prediction if available
            if self.use_ml:
                prediction = self.get_prediction()
                if prediction == "Up" and long_conditions:
                    self._enter_long(price)
                elif prediction == "Down" and short_conditions:
                    self._enter_short(price)
            else:
                # Trade without ML confirmation
                if long_conditions:
                    self._enter_long(price)
                elif short_conditions:
                    self._enter_short(price)
        except Exception as e:
            self.Debug(f"Error checking entry signals: {e}")

    def _enter_long(self, price):
        """Enter a long position."""
        try:
            self.set_holdings(self.symbol, 1)
            self.plot("Trade Plot", "Buy", price)
            self.exit_price = (1 - self.percentage_risk) * price
        except Exception as e:
            self.Debug(f"Error entering long position: {e}")

    def _enter_short(self, price):
        """Enter a short position."""
        try:
            self.set_holdings(self.symbol, -1)
            self.plot("Trade Plot", "Sell", price)
            self.exit_price = (1 + self.percentage_risk) * price
        except Exception as e:
            self.Debug(f"Error entering short position: {e}")

    def _check_exit_signals(self, price):
        """Check for exit signals and close positions."""
        try:
            if self.portfolio[self.symbol].is_long:
                if self.bb.middle_band.current.value <= price:
                    self._exit_position(price)
            else:  # Short position
                if self.bb.middle_band.current.value >= price:
                    self._exit_position(price)
        except Exception as e:
            self.Debug(f"Error checking exit signals: {e}")

    def _exit_position(self, price):
        """Exit the current position."""
        try:
            self.liquidate()
            self.plot("Trade Plot", "Liquidate", price)
            self.exit_price = None
        except Exception as e:
            self.Debug(f"Error exiting position: {e}")

    def on_data(self, data: Slice):
        """Handle stop loss checks on every data update."""
        if self.exit_price is None or self.symbol is None:
            return

        try:
            if self.symbol not in data or data[self.symbol] is None:
                return

            price = data[self.symbol].price

            # Check stop loss for long positions
            if self.portfolio[self.symbol].is_long:
                if price < self.exit_price:
                    self._exit_position(price)
            # Check stop loss for short positions
            elif self.portfolio[self.symbol].is_short:
                if price > self.exit_price:
                    self._exit_position(price)
        except Exception as e:
            self.Debug(f"Error in on_data: {e}")

    def get_prediction(self):
        """
        Get ML model prediction for price direction.
        
        Returns:
            "Up" if model predicts price increase, "Down" otherwise
        """
        if not self.use_ml or self.model is None:
            return None

        try:
            # Validate rolling window has enough data
            if self.rolling_window.count < 30:
                self.Debug(f"Warning: Insufficient data in rolling window ({self.rolling_window.count} < 30)")
                return None

            # Format data from rolling window
            data = [{
                'open': bar.Open,
                'high': bar.High,
                'low': bar.Low,
                'close': bar.Close
            } for bar in self.rolling_window]

            # Generate input data for the model
            df = pd.DataFrame(data)
            df_change = df[["open", "high", "low", "close"]].pct_change().dropna()
            
            if len(df_change) < 30:
                self.Debug("Warning: Insufficient data after percentage change calculation")
                return None

            model_input = []
            for index, row in df_change.tail(30).iterrows():
                model_input.append(np.array(row))
            model_input = np.array([model_input])

            # Get model prediction
            prediction = self.model.predict(model_input, verbose=0)[0][0]
            
            if round(prediction) == 0:
                return "Down"
            else:
                return "Up"
        except Exception as e:
            self.Debug(f"Error getting prediction: {e}")
            return None

    def enable_trading(self):
        """Enable trading for the day."""
        self.is_trading_enabled = True

    def disable_trading(self):
        """Disable trading for the day."""
        self.is_trading_enabled = False

