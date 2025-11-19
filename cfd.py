# region imports
from AlgorithmImports import *
from base_trading_algorithm import BaseTradingAlgorithm
# endregion


class SharedProject(BaseTradingAlgorithm):
    """
    CFD trading algorithm optimized for XAUUSD (Gold).
    Inherits from BaseTradingAlgorithm to reduce code duplication.
    Uses optimized parameters from RL agent.
    """

    def initialize(self):
        """Initialize the CFD trading algorithm."""
        # Call parent initialization
        super().initialize()

        # Setup XAUUSD CFD pair with indicators
        # Parameters (bb_length, rsi_length, adx_length) are loaded from QuantConnect parameters
        # or use defaults set in base class
        self.setup_symbol_and_indicators(
            symbol_name="XAUUSD",
            resolution=Resolution.MINUTE,
            market=Market.OANDA,
            symbol_type="cfd"
        )
