import base64
import hashlib 
import time
import requests
from typing import Optional, List, Dict, Any


class APIAccess:
    """
    QuantConnect API v2 access class for managing projects, backtests, and parameters.
    Provides error handling and validation for all API operations.
    """
    
    def __init__(self, user_id: int, api_token: str, project_id: int, name_of_pair: str):
        """
        Initialize API access with credentials.
        
        Args:
            user_id: QuantConnect user ID
            api_token: QuantConnect API token
            project_id: QuantConnect project ID
            name_of_pair: Trading pair name (e.g., "XAUUSD")
        
        Raises:
            ValueError: If credentials are invalid
            RuntimeError: If authentication or compilation fails
        """
        # Validate inputs
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("user_id must be a positive integer")
        if not api_token or not isinstance(api_token, str):
            raise ValueError("api_token must be a non-empty string")
        if not isinstance(project_id, int) or project_id <= 0:
            raise ValueError("project_id must be a positive integer")
        if not name_of_pair or not isinstance(name_of_pair, str):
            raise ValueError("name_of_pair must be a non-empty string")

        self.USER_ID = user_id
        self.API_TOKEN = api_token
        self.PROJECT_ID = project_id
        self.PAIR_NAME = name_of_pair
        self.counter = 1

        # Authenticate and compile
        self.headers = self.authenticate()
        if self.headers is None:
            raise RuntimeError("Failed to authenticate with QuantConnect API")
        
        self.compile_id = self.compile_project()
        if self.compile_id is None:
            raise RuntimeError("Failed to compile project")

    def authenticate(self) -> Optional[Dict[str, str]]:
        """
        Establish connection to the API and return required headers for authentication.
        
        Returns:
            Headers dictionary if successful, None otherwise
        """
        try:
            # Get timestamp
            timestamp = str(int(time.time()))
            time_stamped_token = self.API_TOKEN + ':' + timestamp

            # Get hashed API token
            hashed_token = hashlib.sha256(time_stamped_token.encode('utf-8')).hexdigest()
            authentication = "{}:{}".format(self.USER_ID, hashed_token)
            encoded_token = base64.b64encode(authentication.encode('utf-8')).decode('ascii')

            # Create headers dictionary
            headers = {
                'Authorization': 'Basic %s' % encoded_token,
                'Timestamp': timestamp
            }

            # Create POST Request
            response = requests.post(
                "https://www.quantconnect.com/api/v2/authenticate",
                data={},
                json={},
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if response_json.get("success") != True:
                error_msg = response_json.get("errors", ["Unknown error"])
                print(f"Authentication failed: {error_msg}")
                return None
            else:
                print("Authentication was successful")
                return headers
                
        except requests.exceptions.RequestException as e:
            print(f"Network error during authentication: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during authentication: {e}")
            return None

    def get_parameters(self) -> Optional[List[float]]:
        """
        Read parameters from the project and return them as an array.
        
        Returns:
            List of parameter values [bb_length, rsi_length, adx_length] if successful, None otherwise
        """
        if self.headers is None:
            print("Cannot get parameters: Not authenticated")
            return None

        try:
            response = requests.get(
                "https://www.quantconnect.com/api/v2/projects/read",
                json={"projectId": self.PROJECT_ID},
                headers=self.headers,
                timeout=30
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if response_json.get("success") != True:
                error_msg = response_json.get("errors", ["Unknown error"])
                print(f"Read failed: {error_msg}")
                return None
            
            projects = response_json.get("projects", [])
            if not projects:
                print("No projects found")
                return None
            
            params = projects[0].get("parameters", [])
            if len(params) < 3:
                print(f"Warning: Expected 3 parameters, got {len(params)}")
            
            parameters = [0.0, 0.0, 0.0]
            for index, parameter in enumerate(params[:3]):  # Only take first 3 parameters
                try:
                    parameters[index] = float(parameter.get("value", 0))
                except (ValueError, TypeError) as e:
                    print(f"Error parsing parameter {index}: {e}")
                    parameters[index] = 0.0

            return parameters
            
        except requests.exceptions.RequestException as e:
            print(f"Network error getting parameters: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error parsing response: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error getting parameters: {e}")
            return None
        
    def update_parameters(self, bb_value: float, rsi_value: float, adx_value: float) -> bool:
        """
        Update the project's parameters using the given values.
        
        Args:
            bb_value: Bollinger Bands length
            rsi_value: RSI length
            adx_value: ADX length
        
        Returns:
            True if successful, False otherwise
        """
        if self.headers is None:
            print("Cannot update parameters: Not authenticated")
            return False

        # Validate parameter values
        try:
            bb_value = int(bb_value)
            rsi_value = int(rsi_value)
            adx_value = int(adx_value)
            
            if bb_value <= 0 or rsi_value <= 0 or adx_value <= 0:
                print("Parameter values must be positive integers")
                return False
        except (ValueError, TypeError):
            print("Parameter values must be numeric")
            return False

        try:
            response = requests.post(
                "https://www.quantconnect.com/api/v2/projects/update",
                json={
                    "projectId": self.PROJECT_ID,
                    "bb_length": bb_value,
                    "rsi_length": rsi_value,
                    "adx_value": adx_value
                },
                headers=self.headers,
                timeout=30
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if response_json.get("success") != True:
                errors = response_json.get("errors", ["Unknown error"])
                print(f"Update failed: {errors}")
                return False
            
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Network error updating parameters: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error updating parameters: {e}")
            return False

    def compile_project(self) -> Optional[str]:
        """
        Compile the project and return the compile ID.
        
        Returns:
            Compile ID if successful, None otherwise
        """
        if self.headers is None:
            print("Cannot compile project: Not authenticated")
            return None

        try:
            response = requests.post(
                "https://www.quantconnect.com/api/v2/compile/create",
                json={"projectId": self.PROJECT_ID},
                headers=self.headers,
                timeout=60  # Compilation may take longer
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if response_json.get("success") != True:
                errors = response_json.get("errors", ["Unknown error"])
                print(f"Compile failed: {errors}")
                return None
            
            compile_id = response_json.get("compileId")
            if compile_id is None:
                print("Compile ID not found in response")
                return None
            
            return compile_id
            
        except requests.exceptions.RequestException as e:
            print(f"Network error compiling project: {e}")
            return None
        except (KeyError, TypeError) as e:
            print(f"Error parsing compile response: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error compiling project: {e}")
            return None

    def backtest(self) -> Optional[str]:
        """
        Backtest the project and return the backtest ID.
        
        Returns:
            Backtest ID if successful, None otherwise
        """
        if self.headers is None:
            print("Cannot run backtest: Not authenticated")
            return None
        
        if self.compile_id is None:
            print("Cannot run backtest: Project not compiled")
            return None

        try:
            response = requests.post(
                "https://www.quantconnect.com/api/v2/projects/update",
                json={
                    "projectId": self.PROJECT_ID,
                    "compileId": self.compile_id,
                    "backtestName": f"APIbacktest{self.counter}"
                },
                headers=self.headers,
                timeout=120  # Backtests may take longer
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if response_json.get("success") != True:
                errors = response_json.get("errors", ["Unknown error"])
                print(f"Backtest failed: {errors}")
                return None
            
            backtests = response_json.get("backtest", [])
            if not backtests:
                print("No backtest ID in response")
                return None
            
            backtest_id = backtests[0].get("backtestId")
            if backtest_id is None:
                print("Backtest ID not found in response")
                return None
            
            self.counter += 1
            return backtest_id
            
        except requests.exceptions.RequestException as e:
            print(f"Network error running backtest: {e}")
            return None
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error parsing backtest response: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error running backtest: {e}")
            return None
        
    def compute_score_from_results(self, backtest_id: str) -> Optional[float]:
        """
        Read results from the backtest and compute a performance score.
        
        Args:
            backtest_id: The backtest ID to retrieve results for
        
        Returns:
            Overall performance score if successful, None otherwise
        """
        if self.headers is None:
            print("Cannot compute score: Not authenticated")
            return None
        
        if not backtest_id:
            print("Cannot compute score: Invalid backtest ID")
            return None

        try:
            response = requests.post(
                "https://www.quantconnect.com/api/v2/projects/update",
                json={
                    "projectId": self.PROJECT_ID,
                    "backtestId": backtest_id
                },
                headers=self.headers,
                timeout=30
            )
            
            response.raise_for_status()
            response_json = response.json()
            
            if response_json.get("success") != True:
                errors = response_json.get("errors", ["Unknown error"])
                print(f"Failed to retrieve backtest results: {errors}")
                return None
            
            backtests = response_json.get("backtest", [])
            if not backtests:
                print("No backtest data in response")
                return None
            
            rolling_window = backtests[0].get("rollingWindow", {})
            if not rolling_window:
                print("No rolling window data in backtest results")
                return None
            
            trade_statistics = rolling_window.get("tradeStatistics", {})
            portfolio_statistics = rolling_window.get("portfolioStatistics", {})
            runtime_statistics = rolling_window.get("runtimeStatistics", {})
            
            # Validate required statistics exist
            required_keys = {
                "tradeStatistics": ["sharpeRatio", "sortinoRatio", "averageWin", "averageLoss"],
                "portfolioStatistics": ["drawdown", "compoundingAnnualReturn", "annualStandardDeviation", "winRate", "lossRate"],
                "runtimeStatistics": ["Return"]
            }
            
            for stat_type, keys in required_keys.items():
                stats = locals()[stat_type]
                for key in keys:
                    if key not in stats:
                        print(f"Warning: Missing statistic {key} in {stat_type}")
                        return None
            
            # Calculate weighted score components
            sharpe_ratio = trade_statistics.get("sharpeRatio", 0) * 0.25
            sortino_ratio = trade_statistics.get("sortinoRatio", 0) * 0.03
            maximum_drawdown = portfolio_statistics.get("drawdown", 0) * -0.25
            annualised_return = portfolio_statistics.get("compoundingAnnualReturn", 0) * 0.20
            volatility = portfolio_statistics.get("annualStandardDeviation", 0) * -0.15
            cumulative_return = runtime_statistics.get("Return", 0) * 0.08
            winning_rate = portfolio_statistics.get("winRate", 0) * 0.02

            losing_rate = portfolio_statistics.get("lossRate", 0)
            avg_win = trade_statistics.get("averageWin", 0)
            avg_loss = trade_statistics.get("averageLoss", 0)
            average_trade_return = ((winning_rate * avg_win) + (losing_rate * avg_loss)) * 0.02

            overall = (
                sharpe_ratio + maximum_drawdown + annualised_return + 
                volatility + cumulative_return + sortino_ratio + 
                winning_rate + average_trade_return
            )

            return overall
            
        except requests.exceptions.RequestException as e:
            print(f"Network error computing score: {e}")
            return None
        except (KeyError, TypeError, ValueError) as e:
            print(f"Error parsing score data: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error computing score: {e}")
            return None
