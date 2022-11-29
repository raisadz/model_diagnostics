# A Dynamic Risk Assessment System

Project **A Dynamic Risk Assessment System** of Machine Learning DevOps Engineer Nanodegree Udacity.

## Project Description
The goal of the project is to set up processes and scripts to re-train, re-deploy, monitor, and report on the ML model that estimates the attrition risk of a company.

## Files and data description
Folder `scripts` contains the main implemented scripts for the project. The following steps were implemented:

* **Data ingestion**. Automatically check a database for new data that can be used for model training. Compile all training data to a training dataset and save it to persistent storage. Write metrics related to the completed data ingestion tasks to persistent storage.

* **Training, scoring, and deploying**. Write scripts that train an ML model that predicts attrition risk, and score the model. Write the model and the scoring metrics to persistent storage.

* **Diagnostics**. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. Check for dependency changes and package updates.

* **Reporting**. Automatically generate plots and documents that report on model metrics. Provide a Flask API endpoint that can return model predictions and metrics.

* **Process Automation**. Create a script and cron job that automatically run all previous steps at regular intervals.
 
## Installation
Clone the repo:

```bash
git clone git@https://github.com/raisadz/model_diagnostics.git
cd model_diagnostics
```

Install [mamba](https://pypi.org/project/mamba/).
Create a conda environment:

```bash
mamba create -n model_diagnostics python=3.8
```

Activate the environment:

```bash
mamba activate model_diagnostics
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements:
```bash
pip install -r requirements.txt
```

## Reproducing the project
To run the full process of re-training and re-deployment in case of detected model drift:
```bash
python -m scripts.fullprocess
```
To calculate model predictions on test data using Flask API first run:
```bash
python -m scripts.app
```
Then in a new terminal window run:
```bash
curl 127.0.0.1:8000/prediction?filename=testdata/testdata.csv
```