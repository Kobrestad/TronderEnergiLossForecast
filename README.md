# Grid Loss Time Series Forecasting: Three Machine Learning Approaches

## Project setup

This project manages virtual environments using pipenv.
This can be downloaded using `pip install pipenv`.

When the projects is cloned open the directory in a terminal and run `pipenv shell`, and then `pipenv install`, to install dependencies.
Then everything should be good to go.

## File structure overview

```
📦TronderEnergiLossForecast
 ┣ 📂Data
 ┃ ┗ 📂raw (Test Data Location)
 ┃ ┃ ┣ 📜test.csv
 ┃ ┃ ┣ 📜test_backfilled_missing_data.csv
 ┃ ┃ ┗ 📜train.csv
 ┣ 📂Notebooks
 ┃ ┗ 📜LSTMgridloss.ipynb (Notebook containing LSTM Model)
 ┣ 📂Source
 ┃ ┣ 📂RegressionModels
 ┃ ┃ ┗ 📜Linear.py (Wrapper class for Linear Regression)
 ┃ ┣ 📜Baseline.py (The baseline prediction function)
 ┃ ┣ 📜DataLoading.py (Used for loading data to models)
 ┃ ┣ 📜DataVisualization.py (Plotting utility)
 ┃ ┣ 📜Evaluation.py (Some Evaluation metrics)
 ┃ ┣ 📜ExponentialSmoothing.py (Run to see Holt-Winters results)
 ┃ ┣ 📜ExponentialSmoothingOptimization.py (Fitting method Holt-Winters)
 ┃ ┣ 📜ExponentialSmoothingPlotting.py (Plotting for Holt-Winters)
 ┃ ┣ 📜main.py ()
 ┃ ┣ 📜Metrics.py (More evaluation metrics)
 ┃ ┗ 📜Regression.py (Contains helper functions for running the Linear model)
 ┣ 📜Pipfile (packages in pyenv)
 ┣ 📜Pipfile.lock
 ┗ 📜README.md
```

## Guide to run the models

### Holt-Winters

### Linear Regression
To run evaluation on test set:
`python Source/Regression.py`

### LSTM

There are 2 options to run the LSTM model

1. The notebook is available in Kaggle: https://www.kaggle.com/potetsos/lstmgridloss

This notebook is already computed, and the output values and results are therefore available in the notebook.  
The notebook can also be run in Kaggle to reproduce the results. 

2. Download and run LSTMgridloss.ipynb  
In order to run the notebook locally, the input locations of the .csv files need to be changed to point at the local .csv files.


## Various resources

data source: https://www.kaggle.com/trnderenergikraft/grid-loss-time-series-dataset

reference paper: https://aaai.org/ojs/index.php/AAAI/article/view/7018

some quick experiments: https://www.kaggle.com/askberk/grid-loss-exploration

LSTM kaggle notebook with outputs: https://www.kaggle.com/potetsos/lstmgridloss
