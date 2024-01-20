import pandas as pd

from prophet_predictions.prophet_model import make_predictions as prophet_predict


# from prophet_predictions.prophet_model import *
from rnn.rnn_model import make_predictions as rnn_predict


def main():
    dataset = pd.read_csv("dataset/SensorMLDataset.csv")
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    # heatmaps(dataset)
    # create_box_plots(dataset)
    # correlation_matrix(dataset)
    # prophet_predict(dataset)
    rnn_predict(dataset, 5)


if __name__ == '__main__':
    main()
