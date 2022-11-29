import json
import os
import ast
import sys
import pandas as pd
from scripts.scoring import score_model

with open('config.json','r') as f:
    config = json.load(f) 

prod_deployment_path = os.path.join(config['prod_deployment_path']) 
input_folder_path = os.path.join(config['input_folder_path']) 
output_folder_path = config['output_folder_path']

##################Check and read new data
#first, read ingestedfiles.txt
with open(prod_deployment_path+'/ingestedfiles.txt', 'r') as f:
    ingestedfiles = ast.literal_eval(f.read())
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
current_files = os.listdir(input_folder_path)
new_files = [x for x in current_files if x not in ingestedfiles]

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if new_files == []:
    sys.exit()

os.system("python -m scripts.ingestion")
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(prod_deployment_path+'/latestscore.txt', 'r') as f:
    latestscore = ast.literal_eval(f.read())

most_recent_data = pd.read_csv(output_folder_path+'/finaldata.csv')
new_score = score_model(most_recent_data)
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if new_score >= latestscore:
    sys.exit()
else:
##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
    os.system("python -m scripts.training")
    os.system("python -m scripts.deployment")

##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system("python -m scripts.reporting")
os.system("python -m scripts.app")
os.system("python -m scripts.apicalls")






