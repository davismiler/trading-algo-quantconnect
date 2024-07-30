import base64
import hashlib 
import time
import requests
import time


class APIAccess():
    def __init__(self, user_id, api_token, project_id, name_of_pair):

        self.USER_ID = user_id
        self.API_TOKEN = api_token
        self.PROJECT_ID = project_id
        self.PAIR_NAME = name_of_pair

        self.counter = 1

        self.headers = self.authenticate()
        self.compile_id = self.compile_project()

    def authenticate(self):

        # Get timestamp
        timestamp = str(int(time.time()))
        time_stamped_token = self.API_TOKEN + ':' + timestamp

        # Get hased API token
        hashed_token = hashlib.sha256(time_stamped_token.encode('utf-8')).hexdigest()
        authentication = "{}:{}".format(self.USER_ID, hashed_token)
        self.API_TOKEN = base64.b64encode(authentication.encode('utf-8')).decode('ascii')

        # Create headers dictionary.
        headers = {
            'Authorization': 'Basic %s' % self.API_TOKEN,
            'Timestamp': timestamp
        }

        # Create POST Request with headers (optional: Json Content as data argument).
        response = requests.post("https://www.quantconnect.com/api/v2/authenticate", 
                                data = {}, 
                                json = {},    # Some request requires json param (must remove the data param in this case)
                                headers = headers)
        response_json = response.json()
        if response_json["success"] != True:
            print("Authentication failed")
        else:
            print("Authentiction was successful")
            return headers
        return None

    def get_parameters(self):
        parameters = [0,0,0]
        response = requests.get("https://www.quantconnect.com/api/v2/projects/read", 
                                json = {
                                    "projectId": self.PROJECT_ID
                                }, 
                                headers = self.headers)
        response_json = response.json()
        if response_json["success"] != True:
            print("Read failed")
        else:
            params = response_json["projects"][0]["parameters"]
            index = 0
            for parameter in params:
                # print("new parameter")
                # for parameter_key_values in parameter.keys():
                #     print(f"{parameter_key_values}: {parameter[parameter_key_values]}" )
                parameters[index] = parameter["value"]
                index += 1

            return parameters
        return None
        

    def update_parameters(self, bb_value, rsi_value, adx_value):
        response = requests.post("https://www.quantconnect.com/api/v2/projects/update", 
                                json = {
                                    "projectId": self.PROJECT_ID,
                                    "bb_length": bb_value,
                                    "rsi_length": rsi_value,
                                    "adx_value" : adx_value
                                }, 
                                headers = self.headers)
        response_json = response.json()
        if response_json["success"] != True:
            print("Update failed")
            for error in response_json["errors"]:
                print(error)
        return None

    def compile_project(self):

        response = requests.post("https://www.quantconnect.com/api/v2/compile/create", 
                                json = {
                                    "projectId": self.PROJECT_ID
                                }, 
                                headers = self.headers)
        response_json = response.json()
        if response_json["success"] != True:
            print("Compile failed")
            for error in response_json["errors"]:
                print(error)
        else:
            compile_id = response_json["compileId"]
            return compile_id
        return None

    def backtest(self):
        response = requests.post("https://www.quantconnect.com/api/v2/projects/update", 
                                json = {
                                    "projectId": self.PROJECT_ID,
                                    "compileId": self.compile_id,
                                    "backtestName":f"APIbacktest{self.counter}"
                                }, 
                                headers = self.headers)
        response_json = response.json()
        if response_json["success"] != True:
            print("Backtest failed")
            for error in response_json["errors"]:
                print(error)
        else:
            backtest_id = response_json["backtest"][0]["backtestId"]
            self.counter += 1
            return backtest_id
        
    def compute_score_from_results(self, backtest_id):
            response = requests.post("https://www.quantconnect.com/api/v2/projects/update", 
                                json = {
                                    "projectId": self.PROJECT_ID,
                                    "backtestId": backtest_id
                                }, 
                                headers = self.headers)
            
            response_json = response.json()
            trade_statistics = response_json["backtest"][0]["rollingWindow"]["tradeStatistics"]
            portfolio_statistics = response_json["backtest"][0]["rollingWindow"]["portfolioStatistics"]
            runtime_statistics = response_json["backtest"][0]["rollingWindow"]["runtimeStatistics"]

            sharpe_ratio = trade_statistics["sharpeRatio"] * 0.25
            sortino_ratio = trade_statistics["sortinoRatio"] * 0.03
            maximum_drawdown = portfolio_statistics["drawdown"] * -0.25
            annualised_return = portfolio_statistics["compoundingAnnualReturn"] * 0.20
            volatility = portfolio_statistics["annualStandardDeviation"] * -0.15
            cumulative_return = runtime_statistics["Return"] * 0.08
            winning_rate = portfolio_statistics["winRate"] * 0.02

            losing_rate = portfolio_statistics["lossRate"]
            avg_win = trade_statistics["averageWin"]
            avg_loss = trade_statistics["averageLoss"]
            average_trade_return = ((winning_rate * avg_win) + (losing_rate * avg_loss)) * 0.02

            overall = sharpe_ratio + maximum_drawdown + annualised_return + volatility + cumulative_return + sortino_ratio + winning_rate + average_trade_return 

            return overall