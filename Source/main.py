from Regression import get_trained_linear_model, get_data
from RegressionModels.Linear import GridLossLinearModel
from DataVisualization import create_timeseries_figure


def main():
    (x_train, y_train), (X, y) = get_data()
    model = GridLossLinearModel.get_untrained_model(
        {
            "linear__fit_intercept": True,
            "linear__normalize": True,
            "polys__degree": 2,
        }
    ).fit(x_train, y_train)

    # predictions = model.online_prediction(X, y)
    predictions = model.predict(X)

    fig1 = create_timeseries_figure(y)
    fig1.savefig("actual.pdf")

    fig2 = create_timeseries_figure(
        predictions,
        title="Linear Regression grid loss prediction with nightly retraining",
        xlabel="Hour (h)",
        ylabel="Grid loss (MWh)",
    )
    fig2.savefig("predictions_online.pdf")


if __name__ == "__main__":
    main()