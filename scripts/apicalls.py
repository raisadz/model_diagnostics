import requests
import os
import json

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = os.path.join(config['output_model_path']) 

#Call each API endpoint and store the responses
response1 = requests.get(f'{URL}/prediction?filename=testdata/testdata.csv').content
response2 = requests.get(f'{URL}/scoring').content
response3 = requests.get(f'{URL}/summarystats').content
response4 = requests.get(f'{URL}/diagnostics').content

#combine all API responses
responses = response1.decode() + '\n' + response2.decode() + '\n' + response3.decode() + '\n' + response4.decode()

#write the responses to your workspace

with open(output_model_path+'/apireturns.txt', 'w') as f:
    f.write(responses)
