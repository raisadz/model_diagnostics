import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import f1_score
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

output_model_path = os.path.join(config['output_model_path']) 
test_data_path = os.path.join(config['test_data_path']) 


#################Function for model scoring
def score_model():
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    with open(output_model_path+'/'+'trainedmodel.pkl', 'rb') as f:
        model = pickle.load(f)
    test_data = pd.DataFrame()
    #check for datasets, compile them together, and write to an output file
    for filename in os.listdir(test_data_path):
        df = pd.read_csv(test_data_path+'/'+filename)
        test_data = test_data.append(df.reset_index(drop=True), ignore_index = True)
        
    X_test = np.array(test_data[['lastmonth_activity','lastyear_activity','number_of_employees']])
    y_test = test_data['exited'].values.reshape(-1, 1).ravel()
    preds = model.predict(X_test)
    f1_test = f1_score(y_test, preds)
    with open(output_model_path+'/'+'latestscore.txt', 'w') as f:
        f.write(str(round(f1_test, 6)))

if __name__ == '__main__':
    score_model()