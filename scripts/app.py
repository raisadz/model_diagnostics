from flask import Flask, request
import pandas as pd
import json
import os
from scripts.diagnostics import model_predictions, dataframe_summary, dataframe_percent_na, execution_time, outdated_packages_list
from scripts.scoring import score_model



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 

# prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['GET','OPTIONS'])
def prediction():        
    #call the prediction function you created in Step 3
    filename = request.args.get('filename')
    thedata=pd.read_csv(filename)
    preds = model_predictions(thedata)
    return str(preds) #add return value for prediction outputs

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    filename = test_data_path+'/testdata.csv'
    thedata=pd.read_csv(filename)
    f1_test = score_model(thedata)
    return str(f1_test) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():        
    #check means, medians, and modes for each column
    filename = test_data_path+'/testdata.csv'
    thedata=pd.read_csv(filename)
    df_summary = dataframe_summary(thedata)
    return str(df_summary) #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():        
    filename = test_data_path+'/testdata.csv'
    thedata=pd.read_csv(filename)
    #check timing and percent NA values
    return_time = execution_time()
    na_percent = dataframe_percent_na(thedata)
    outdated_req = outdated_packages_list()
    return str(return_time) + '\n' + str(na_percent) +'\n' + str(outdated_req)

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
