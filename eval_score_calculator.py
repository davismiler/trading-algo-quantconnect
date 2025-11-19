import csv
from typing import List, Dict, Optional
from tabulate import tabulate


def validate_float_input(prompt: str, allow_negative: bool = True) -> float:
    """
    Safely get and validate float input from user.
    
    Args:
        prompt: Input prompt message
        allow_negative: Whether negative values are allowed
    
    Returns:
        Validated float value
    """
    while True:
        try:
            value = float(input(prompt))
            if not allow_negative and value < 0:
                print("Please enter a non-negative value.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a numeric value.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            raise


def generate_row() -> Dict[str, float]:
    """
    Generate a row of evaluation metrics from user input.
    
    Returns:
        Dictionary containing all metrics and overall score
    """
    try:
        pair = input("Name of the pair: ").strip()
        if not pair:
            raise ValueError("Pair name cannot be empty")
        
        sharpe_ratio = validate_float_input("Sharpe Ratio: ") * 0.25
        maximum_drawdown = validate_float_input("Maximum Drawdown: ") * -0.25
        annualised_return = validate_float_input("Annualised Return: ") * 0.20
        volatility = validate_float_input("Volatility: ", allow_negative=False) * -0.15
        cumulative_return = validate_float_input("Cumulative Return: ") * 0.08
        sortino_ratio = validate_float_input("Sortino Ratio: ") * 0.03
        winning_rate = validate_float_input("Winning Rate: ", allow_negative=False) * 0.02
        average_trade_return = validate_float_input("Average Trade Return: ") * 0.02

        overall = (
            sharpe_ratio + maximum_drawdown + annualised_return + 
            volatility + cumulative_return + sortino_ratio + 
            winning_rate + average_trade_return
        )

        # Create dictionary representing the row
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
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Error generating row: {e}")
        raise


def save_rows_to_csv(filename: str, rows: List[Dict]) -> bool:
    """
    Save rows to CSV file.
    
    Args:
        filename: Name of the CSV file
        rows: List of dictionaries containing row data
    
    Returns:
        True if successful, False otherwise
    """
    if not rows:
        print("No rows to save.")
        return False
    
    headers = [
        "Pair", "Sharpe Ratio", "Maximum Drawdown", "Annualised Return",
        "Volatility", "Cumulative Return", "Sortino Ratio", "Winning Rate",
        "Average Trade Return", "Overall"
    ]
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Successfully saved {len(rows)} rows to {filename}")
        return True
    except IOError as e:
        print(f"Error saving to CSV file: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error saving CSV: {e}")
        return False


def load_rows_from_csv(filename: str) -> List[Dict]:
    """
    Load rows from CSV file.
    
    Args:
        filename: Name of the CSV file to load
    
    Returns:
        List of dictionaries containing row data
    """
    rows = []
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Validate headers
            expected_headers = [
                "Pair", "Sharpe Ratio", "Maximum Drawdown", "Annualised Return",
                "Volatility", "Cumulative Return", "Sortino Ratio", "Winning Rate",
                "Average Trade Return", "Overall"
            ]
            
            if reader.fieldnames != expected_headers:
                print(f"Warning: CSV headers don't match expected format")
            
            for row_num, csv_row in enumerate(reader, start=2):  # Start at 2 (header is row 1)
                try:
                    row = {
                        "Pair": str(csv_row.get("Pair", "")).strip(),
                        "Sharpe Ratio": float(csv_row.get("Sharpe Ratio", 0)),
                        "Maximum Drawdown": float(csv_row.get("Maximum Drawdown", 0)),
                        "Annualised Return": float(csv_row.get("Annualised Return", 0)),
                        "Volatility": float(csv_row.get("Volatility", 0)),
                        "Cumulative Return": float(csv_row.get("Cumulative Return", 0)),
                        "Sortino Ratio": float(csv_row.get("Sortino Ratio", 0)),
                        "Winning Rate": float(csv_row.get("Winning Rate", 0)),
                        "Average Trade Return": float(csv_row.get("Average Trade Return", 0)),
                        "Overall": float(csv_row.get("Overall", 0))
                    }
                    rows.append(row)
                except (ValueError, KeyError) as e:
                    print(f"Warning: Skipping row {row_num} due to error: {e}")
                    continue
            
            if rows:
                print(f"Successfully loaded {len(rows)} rows from {filename}")
            else:
                print(f"No valid rows found in {filename}")
                
    except FileNotFoundError:
        print(f"File {filename} not found. Starting with empty list.")
    except IOError as e:
        print(f"Error reading CSV file: {e}")
    except Exception as e:
        print(f"Unexpected error loading CSV: {e}")
    
    return rows


def display_table(rows: List[Dict]) -> None:
    """
    Display rows in a formatted table.
    
    Args:
        rows: List of dictionaries containing row data
    """
    if not rows:
        print("No data to display.")
        return
    
    headers = [
        "Pair", "Sharpe Ratio", "Maximum Drawdown", "Annualised Return",
        "Volatility", "Cumulative Return", "Sortino Ratio", "Winning Rate",
        "Average Trade Return", "Overall"
    ]
    
    try:
        table = [list(row.values()) for row in rows]
        print(tabulate(table, headers=headers, floatfmt=".4f", tablefmt="grid"))
    except Exception as e:
        print(f"Error displaying table: {e}")


def main():
    """Main function to run the evaluation score calculator."""
    filename = "algorithm_evaluation.csv"
    
    try:
        choice = input("Load or input values (L/I): ").strip().lower()
        
        if choice == "l":
            # Load existing rows from CSV file
            rows = load_rows_from_csv(filename)
            if rows:
                display_table(rows)
        elif choice == "i":
            # Create new rows from user input
            rows = []
            while True:
                try:
                    rows.append(generate_row())
                    more = input("Do you want to add another row? (Y/N): ").strip().lower()
                    if more != 'y':
                        break
                except (ValueError, KeyboardInterrupt):
                    print("Cancelled row entry.")
                    if rows:
                        save_choice = input(f"Save {len(rows)} existing rows? (Y/N): ").strip().lower()
                        if save_choice == 'y':
                            save_rows_to_csv(filename, rows)
                    break
            
            if rows:
                # Save the rows to CSV file
                save_rows_to_csv(filename, rows)
                # Display the table
                display_table(rows)
        else:
            print("Invalid choice. Please enter 'L' to load or 'I' to input.")
            return
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
