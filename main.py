import pandas as pd

from data_analysis.analysis import correlation_matrix, create_box_plots
from prophet_predictions.prophet_model import ProphetModel
from rnn.rnn_model import RnnModel
from seq2seq.seq2seq_model import Seq2SeqModel
from globals import *


def main():
    dataset = pd.read_csv("dataset/SensorMLDataset.csv")
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

    # correlation_matrix(dataset)
    # create_box_plots(dataset)
    prophet_model = ProphetModel(TRAINING_HOURS, PREDICTION_HOURS)
    prophet_model.make_predictions(dataset, 'pres')

    # rnn_model = RnnModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
    #                      RNN_EPOCHS, 24, autoregressive=False)
    # print(rnn_model.make_predictions(dataset, 'pres'))

    # seq2seq_model = Seq2SeqModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
    #                              RNN_EPOCHS, 24, 12, autoregressive=False)
    # seq2seq_model.make_predictions(dataset, 'pres')



if __name__ == '__main__':
    main()
