# region imports
from AlgorithmImports import *
from base_trading_algorithm import BaseTradingAlgorithm
# endregion


class HyperActiveRedChinchilla(BaseTradingAlgorithm):
    """
    Forex trading algorithm for GBPUSD pair.
    Inherits from BaseTradingAlgorithm to reduce code duplication.
    """

    def initialize(self):
        """Initialize the forex trading algorithm."""
        # Call parent initialization
        super().initialize()

        # Setup GBPUSD forex pair with indicators
        self.setup_symbol_and_indicators(
            symbol_name="GBPUSD",
            resolution=Resolution.MINUTE,
            market=Market.OANDA,
            symbol_type="forex"
        )
