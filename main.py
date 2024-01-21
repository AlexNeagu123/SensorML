import pandas as pd

from data_analysis.analysis import correlation_matrix, create_box_plots
from prophet_predictions.prophet_model import ProphetModel
from rnn.rnn_model import RnnModel

from globals import *


def main():
    dataset = pd.read_csv("dataset/SensorMLDataset.csv")
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

    # correlation_matrix(dataset)
    # create_box_plots(dataset)
    # prophet_model = ProphetModel(TRAINING_HOURS, PREDICTION_HOURS)
    # prophet_model.make_predictions(dataset, 'temp1')

    rnn_model = RnnModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE, RNN_EPOCHS, 5)
    print(rnn_model.make_predictions(dataset, 'temp1'))


if __name__ == '__main__':
    main()
