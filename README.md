# Grid Loss Time Series Forecasting: Three Machine Learning Approaches

## File structure overview

```
📦TronderEnergiLossForecast
 ┣ 📂Data
 ┃ ┗ 📂raw (Test Data Location)
 ┃ ┃ ┣ 📜test.csv
 ┃ ┃ ┣ 📜test_backfilled_missing_data.csv
 ┃ ┃ ┗ 📜train.csv
 ┣ 📂Notebooks (Notebooks contain LSTM Model)
 ┃ ┣ 📜lstm.ipynb
 ┃ ┗ 📜LSTMgridloss.ipynb
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

## Various resources

data source: https://www.kaggle.com/trnderenergikraft/grid-loss-time-series-dataset

reference paper: https://aaai.org/ojs/index.php/AAAI/article/view/7018

some quick experiments: https://www.kaggle.com/askberk/grid-loss-exploration

LSTM kaggle notebook with outputs: https://www.kaggle.com/potetsos/lstmgridloss
