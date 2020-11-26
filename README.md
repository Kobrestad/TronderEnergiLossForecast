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
 ┃ ┃ ┗ 📜Linear.py
 ┃ ┣ 📜Baseline.py (The baseline prediction function)
 ┃ ┣ 📜DataLoading.py (Used for loading data to models)
 ┃ ┣ 📜DataVisualization.py (Plotting utility)
 ┃ ┣ 📜Evaluation.py (Some Evaluation metrics)
 ┃ ┣ 📜ExponentialSmoothing.py (Run to see Holt-Winters results)
 ┃ ┣ 📜ExponentialSmoothingOptimization.py (Fitting method Holt-Winters)
 ┃ ┣ 📜ExponentialSmoothingPlotting.py (Plotting for Holt-Winters)
 ┃ ┣ 📜main.py ()
 ┃ ┣ 📜Metrics.py (More evaluation metrics)
 ┃ ┗ 📜Regression.py (Contains the regression model)
 ┣ 📜Pipfile (packages in pyenv)
 ┣ 📜Pipfile.lock
 ┗ 📜README.md
```

## Guide to run the models

---

### Holt-Winters

The files regarding the Holt-Winters method are

- [`ExponentialSmoothing.py`](./Source/ExponentialSmoothing.py)
- [`ExponentialSmoothingOptimalization.py`](./Source/ExponentialSmoothingOptimalization.py)
- [`ExponentialSmoothingPlotting.py`](./Source/ExponentialSmoothingPlotting.py)

The implementations of the Holt-Winters functions are found in [`ExponentialSmoothing.py`](./Source/ExponentialSmoothing.py). For this file it is enough to run it. The main method will run through the dataset using, the offline method, evaluating it and plotting it. Then it will run the online method, evaluate it and plot it twice, firstly with the whole dataset, and the second time only with the test set.

The model parameters that are set as default in the holt-winters methods are found running the `optimalization` method in [`ExponentialSmoothingOptimalization.py`](./Source/ExponentialSmoothing.py). This file can also be run as it is, however might take some time to complete. The output from running the file will be the found optimal values for α, β and γ.

### Linerar Regression

### LSTM

## Various resources

data source: https://www.kaggle.com/trnderenergikraft/grid-loss-time-series-dataset

reference paper: https://aaai.org/ojs/index.php/AAAI/article/view/7018

some quick experiments: https://www.kaggle.com/askberk/grid-loss-exploration

LSTM kaggle notebook with outputs: https://www.kaggle.com/potetsos/lstmgridloss
