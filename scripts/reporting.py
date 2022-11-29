import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scripts.diagnostics import model_predictions
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path'])
output_model_path = os.path.join(config['output_model_path']) 



##############Function for reporting
def score_model(test_data):
    preds = model_predictions(test_data)
    y_test = test_data['exited'].tolist()
    #calculate a confusion matrix using the test data and the deployed model
    conf_mat = confusion_matrix(y_test, preds)
    #write the confusion matrix to the workspace
    fig, ax = plt.subplots()
    sns.heatmap(conf_mat, annot = True)
    ax.set_xlabel('predictions')
    ax.set_ylabel('target')
    plt.rc('font', size=50) 
    fig.show()
    fig.savefig(output_model_path+'/confusionmatrix.png')




if __name__ == '__main__':
    test_data = pd.read_csv(test_data_path+'/testdata.csv')
    score_model(test_data)
