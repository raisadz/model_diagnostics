import pandas as pd
import os
import json

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

os.makedirs(output_folder_path, exist_ok=True)

#############Function for data ingestion
def merge_multiple_dataframe():
    df_total = pd.DataFrame()
    filename_list = []
    #check for datasets, compile them together, and write to an output file
    for filename in os.listdir(input_folder_path):
        df = pd.read_csv(input_folder_path+'/'+filename)
        df_total = df_total.append(df.reset_index(drop=True), ignore_index = True)
        filename_list.append(filename)
    df_dedup = df_total.drop_duplicates()
    df_dedup.to_csv(output_folder_path+'/finaldata.csv', index=False)
    with open(output_folder_path+'/ingestedfiles.txt', "w") as f:
        f.write(str(filename_list))

if __name__ == '__main__':
    merge_multiple_dataframe()
