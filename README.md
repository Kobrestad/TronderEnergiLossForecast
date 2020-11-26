# Grid Loss Time Series Forecasting: Three Machine Learning Approaches

## Description
This project is inspired by the kaggle competition and paper linked below.

data source: https://www.kaggle.com/trnderenergikraft/grid-loss-time-series-dataset

reference paper: https://aaai.org/ojs/index.php/AAAI/article/view/7018

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

The files regarding the Holt-Winters method are

- [`ExponentialSmoothing.py`](./Source/ExponentialSmoothing.py)
- [`ExponentialSmoothingOptimization.py`](./Source/ExponentialSmoothingOptimization.py)
- [`ExponentialSmoothingPlotting.py`](./Source/ExponentialSmoothingPlotting.py)

The implementations of the Holt-Winters functions are found in [`ExponentialSmoothing.py`](./Source/ExponentialSmoothing.py). For this file it is enough to run it. The main method will run through the dataset using, the offline method, evaluating it and plotting it. Then it will run the online method, evaluate it and plot it twice, firstly with the whole dataset, and the second time only with the test set.

The model parameters that are set as default in the holt-winters methods are found running the `optimalization` method in [`ExponentialSmoothingOptimization.py`](./Source/ExponentialSmoothingOptimization.py). This file can also be run as it is, however might take some time to complete. The output from running the file will be the found optimal values for α, β and γ.

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
