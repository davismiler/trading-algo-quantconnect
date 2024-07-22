#This program is designed to measure the overall performance of the algorthim using the results from back testing 
Sharpe_ratio = float(input("Input Sharpe Ratio:")) * 0.25
Maximum_Drawdown = float(input("Maximum Drawdown:")) * 0.25
Annualised_Return = float(input("Annualised Return:")) * 0.20
Volatility = float(input("Volatility:")) * 0.15
Cumulative_Return = float(input("Cumulative Return:")) * 0.05
Sortino_Ratio = float(input("Sortino Ratio:")) * 0.03
Winning_Rate = float(input("Winning Rate:")) * 0.02
Profit = float(input("Profit:"))* 0.02
Average_Trade_Return = float(input("Average Trade Return:"))* 0.02
Average_hold = float(input("Average Holding Period:")) * 0.01

overall = Sharpe_ratio + Maximum_Drawdown + Annualised_Return + Volatility + Cumulative_Return +Sortino_Ratio + Winning_Rate + Profit + Average_Trade_Return + Average_hold

print("Evaluation score for your algorithm: "+str(overall))
