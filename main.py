# region imports
from AlgorithmImports import *
# endregion

class SharedProject(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_cash(100000)

        self.xauusd = self.add_cfd("XAUUSD", Resolution.Daily, Market.OANDA).symbol
        self.bb = self.BB(self.xauusd, 20, 2)
        self.rsi = self.RSI(self.xauusd, 20)

        stock_plot = Chart("Trade Plot")
        stock_plot.add_series(Series("Buy", SeriesType.SCATTER, "$", 
        Color.Green, ScatterMarkerSymbol.TRIANGLE))
        stock_plot.add_series(Series("Sell", SeriesType.SCATTER, "$", 
        Color.Green, ScatterMarkerSymbol.TRIANGLE_DOWN))
        stock_plot.add_series(Series("Liquidate", SeriesType.SCATTER, "$", 
        Color.Green, ScatterMarkerSymbol.DIAMOND))

        self.set_benchmark("SPY")

        self.add_chart(stock_plot)

    def on_data(self, data: Slice):
        if not self.bb.is_ready or not self.rsi.is_ready:
            return

        price = data[self.xauusd].price

<<<<<<< HEAD
        self.plot("Trade Plot", "Price", price)
        self.plot("Trade Plot", "MiddleBand", self.bb.middle_band.current.value)
        self.plot("Trade Plot", "UpperBand", self.bb.upper_band.current.value)
        self.plot("Trade Plot", "LowerBand", self.bb.lower_band.current.value)

        if not self.portfolio.invested:
            if self.bb.lower_band.current.value > price and self.rsi.current.Value < 30:
                self.set_holdings(self.xauusd, 1)
                self.plot("Trade Plot", "Buy", price)
            elif self.bb.upper_band.current.value < price and self.rsi.current.Value < 70:
                self.set_holdings(self.xauusd, -1)
                self.plot("Trade Plot", "Sell", price)
=======
        if not self.Portfolio.Invested:
            if self.bb.LowerBand.Current.Value > price and self.rsi.Current.Value < 25:
                self.SetHoldings(self.xauusd, 1)
                self.Plot("Trade Plot", "Buy", price)
            elif self.bb.UpperBand.Current.Value < price and self.rsi.Current.Value < 75:
                self.SetHoldings(self.xauusd, -1)
                self.Plot("Trade Plot", "Sell", price)
>>>>>>> baa480e245d37d0a2863eb8881fd2caee20f357e
        else:
            if self.portfolio[self.xauusd].is_long:
                if self.bb.middle_band.current.value < price:
                    self.liquidate()
                    self.plot("Trade Plot", "Liquidate", price)
            elif self.bb.middle_band.current.value > price:
                    self.liquidate()
                    self.plot("Trade Plot", "Liquidate", price)