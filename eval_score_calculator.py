import csv


def generate_row():
    # Enter percentages
    pair = input("Name of the pair: ")
    sharpe_ratio = float(input("Sharpe Ratio: ")) * 0.25
    maximum_drawdown = float(input("Maximum Drawdown: ")) * -0.25     # Drawdown
    annualised_return = float(input("Annualised Return: ")) * 0.20   # Compounding Annual Return
    volatility = float(input("volatility: ")) * -0.15                 # Annual Standard deviation
    cumulative_return = float(input("Cumulative Return: ")) * 0.08   # Return
    sortino_ratio = float(input("Sortino Ratio: ")) * 0.03
    winning_rate = float(input("Winning Rate: ")) * 0.02
    average_trade_return = float(input("Average Trade Return: ")) * 0.02     # (win rate * average win) + (loss rate * average loss)

    overall = sharpe_ratio + maximum_drawdown + annualised_return + volatility + cumulative_return + sortino_ratio + winning_rate + average_trade_return 

    # Creates a dictionary representing the row
    row = {
        "Pair": pair,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": maximum_drawdown,
        "Annualised Return": annualised_return,
        "Volatility": volatility,
        "Cumulative Return": cumulative_return,
        "Sortino Ratio": sortino_ratio,
        "Winning Rate": winning_rate,
        "Average Trade Return": average_trade_return,
        "Overall": overall
    }

    return row


def save_rows_to_csv(filename, rows):
    headers = ["Pair", "Sharpe Ratio", "Maximum Drawdown", "Annualised Return", "Volatility", "Cumulative Return", "Sortino Ratio", "Winning Rate", "Average Trade Return", "Overall"]
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def load_rows_from_csv(filename):
    rows = []
    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            for csv_row in reader:
                row = {}
                # Converts all values to appropriate types
                row["Pair"] = str(csv_row["Pair"])
                row["Sharpe Ratio"] = float(csv_row["Sharpe Ratio"])
                row["Maximum Drawdown"] = float(csv_row["Maximum Drawdown"])
                row["Annualised Return"] = float(csv_row["Annualised Return"])
                row["Volatility"] = float(csv_row["Volatility"])
                row["Cumulative Return"] = float(csv_row["Cumulative Return"])
                row["Sortino Ratio"] = float(csv_row["Sortino Ratio"])
                row["Winning Rate"] = float(csv_row["Winning Rate"])
                row["Average Trade Return"] = float(csv_row["Average Trade Return"])
                row["Overall"] = float(csv_row["Overall"])
                rows.append(row)
    except FileNotFoundError:
        pass  # If the file does not exist, return an empty list
    return rows

# File where the data will be saved
filename = "algorithm_evaluation.csv"


choice = input("Load or input values (L/I): ")

if choice.lower() == "l":

    # Loads existing rows from the CSV file
    rows = load_rows_from_csv(filename)

else:

    # Creates an indefinite number of rows 
    rows = []
    while True:
        rows.append(generate_row())
        more = input("Do you want to add another row? (Y/N): ")
        if more.lower() != 'y':
            break

    # Save the rows to the CSV file
    save_rows_to_csv(filename, rows)


# Creates and prints the table
from tabulate import tabulate
headers = ["Pair", "Sharpe Ratio", "Maximum Drawdown", "Annualised Return", "Volatility", "Cumulative Return", "Sortino Ratio", "Winning Rate", "Average Trade Return", "Overall"]
table = [list(row.values()) for row in rows]
print(tabulate(table, headers=headers, floatfmt=".4f", tablefmt="grid"))