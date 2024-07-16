# region imports
from AlgorithmImports import *
# endregion

class SharedProject(QCAlgorithm):

    def initialize(self):
        self.SetStartDate(2018, 1, 12)
        self.SetEndDate(2024, 1, 12)
        self.SetCash(100000)
        self.xauusd = self.AddCfd("XAUUSD", Resolution.Daily, Market.Oanda).Symbol
        self.bb = self.BB(self.xauusd, 20, 2)

        stockPlot = Chart("Trade Plot")
        stockPlot.AddSeries(Series("Buy", SeriesType.Scatter, "$", Color.Green, ScatterMarkerSymbol.Triangle))
        stockPlot.AddSeries(Series("Sell", SeriesType.Scatter, "$", Color.Red, ScatterMarkerSymbol.TriangleDown))
        stockPlot.AddSeries(Series("Liquidate", SeriesType.Scatter, "$", Color.Blue, ScatterMarkerSymbol.Diamond))
        self.AddChart(stockPlot)

    def on_data(self, data: Slice):
        if not self.bb.IsReady:
            return
        
        price = data[self.xauusd].Price
        self.Plot("Trade Plot", "Price", price)
        self.Plot("Trade Plot", "MiddleBand", self.bb.MiddleBand.Current.Value)
        self.Plot("Trade Plot", "UpperBand", self.bb.UpperBand.Current.Value)
        self.Plot("Trade Plot", "LowerBand", self.bb.LowerBand.Current.Value)

        if not self.Portfolio.Invested:
            if self.bb.LowerBand.Current.Value > price:
                self.SetHoldings(self.xauusd, 1)
                self.Plot("Trade Plot", "Buy", price)
            elif self.bb.UpperBand.Current.Value < price:
                self.SetHoldings(self.xauusd, -1)
                self.Plot("Trade Plot", "Sell", price)
        else:
            if self.Portfolio[self.xauusd].IsLong:
                if self.bb.MiddleBand.Current.Value < price:
                    self.Liquidate()
                    self.Plot("Trade Plot", "Liquidate", price)
            elif self.bb.MiddleBand.Current.Value > price:
                self.Liquidate()
                self.Plot("Trade Plot", "Liquidate", price)