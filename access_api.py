
USER_ID = None
API_TOKEN = None
PROJECT_ID = 18808389
COMPILE_ID = "26e9b46eb873175f1b6ab5c2e6096cec-aef1f3ab94c19a75c2e8230a3bbd87ee"

import base64
import hashlib 
import time
import requests



def authenticate():

    # Get timestamp
    timestamp = str(int(time.time()))
    time_stamped_token = API_TOKEN + ':' + timestamp

    # Get hased API token
    hashed_token = hashlib.sha256(time_stamped_token.encode('utf-8')).hexdigest()
    authentication = "{}:{}".format(USER_ID, hashed_token)
    api_token = base64.b64encode(authentication.encode('utf-8')).decode('ascii')

    # Create headers dictionary.
    headers = {
        'Authorization': 'Basic %s' % api_token,
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

def read_project():
    response = requests.post("https://www.quantconnect.com/api/v2/projects/read", 
                            json = {
                                "projectId": PROJECT_ID
                            }, 
                            headers = headers)
    response_json = response.json()
    print(response_json["projects"][0]["projectId"])
    params = response_json["projects"][0]["parameters"]
    for parameter in params:
        for parameter_key_values in parameter.keys():
            print(f"{parameter_key_values}: {parameter[parameter_key_values]}" )

def update_and_compile():
    pass

headers = authenticate()
read_project()