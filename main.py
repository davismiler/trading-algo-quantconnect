# region imports
from AlgorithmImports import *
# endregion

class SharedProject(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2022, 1, 1)
        self.set_cash(100000)

        self.xauusd = self.add_cfd("XAUUSD", Resolution.MINUTE, Market.OANDA).symbol
        self.bb = self.BB(self.xauusd, 20, 2)
        self.rsi = self.RSI(self.xauusd, 20)

        # operates on a 30 min time frame
        self.consolidate("XAUUSD", timedelta(minutes=15), self.on_30_data)

        # creates a stop loss
        self.exit_price = None
        self.percentage_risk = 0.02

        # creates plot of portfolio activity 
        stock_plot = Chart("Trade Plot")
        stock_plot.add_series(Series("Buy", SeriesType.SCATTER, "$", 
        Color.Green, ScatterMarkerSymbol.TRIANGLE))
        stock_plot.add_series(Series("Sell", SeriesType.SCATTER, "$", 
        Color.Green, ScatterMarkerSymbol.TRIANGLE_DOWN))
        stock_plot.add_series(Series("Liquidate", SeriesType.SCATTER, "$", 
        Color.Green, ScatterMarkerSymbol.DIAMOND))

        self.set_benchmark("SPY")

        self.add_chart(stock_plot)

    def on_30_data(self, bar):
        if not self.bb.is_ready or not self.rsi.is_ready:
            return

        price = bar.close

        self.plot("Trade Plot", "Price", price)
        self.plot("Trade Plot", "MiddleBand", self.bb.middle_band.current.value)
        self.plot("Trade Plot", "UpperBand", self.bb.upper_band.current.value)
        self.plot("Trade Plot", "LowerBand", self.bb.lower_band.current.value)

        if not self.portfolio.invested:
            if self.bb.lower_band.current.value > price and self.rsi.current.Value < 30:
                self.set_holdings(self.xauusd, 1)
                self.plot("Trade Plot", "Buy", price)
                self.exit_price = (1 - self.percentage_risk) * price
            elif self.bb.upper_band.current.value < price and self.rsi.current.Value < 70:
                self.set_holdings(self.xauusd, -1)
                self.plot("Trade Plot", "Sell", price)
                self.exit_price = (1 + self.percentage_risk) * price
        else:
            if self.portfolio[self.xauusd].is_long:
                if self.bb.middle_band.current.value <= price:
                    self.liquidate()
                    self.plot("Trade Plot", "Liquidate", price)
                    self.exit_price = None
            elif self.bb.middle_band.current.value >= price:
                    self.liquidate()
                    self.plot("Trade Plot", "Liquidate", price)
                    self.exit_price = None
            

    def on_data(self, data: Slice):
        if not self.exit_price:
            return

        price = data[self.xauusd].price

        if self.portfolio[self.xauusd].is_long:
            if price < self.exit_price:
                self.liquidate()
                self.exit_price = None
                
        else:
            if price > self.exit_price:
                self.liquidate()
                self.exit_price = None