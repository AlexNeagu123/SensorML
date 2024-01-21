import pandas as pd

from globals import *
from nn_predictions.rnn_model import RnnModel
from nn_predictions.seq2seq_model import Seq2SeqModel
from prophet_predictions.prophet_model import ProphetModel


def main():
    dataset = pd.read_csv("dataset/SensorMLDataset.csv")
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

    # correlation_matrix(dataset)
    # create_box_plots(dataset)
    # prophet_model = ProphetModel(TRAINING_HOURS, PREDICTION_HOURS)
    # print(prophet_model.make_predictions(dataset, 'temp1'))

    rnn_model = RnnModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
                         RNN_EPOCHS, 24, autoregressive=True)
    print(rnn_model.make_predictions(dataset, 'temp1')[1])

    # seq2seq_model = Seq2SeqModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
    #                              RNN_EPOCHS, 24, 12, autoregressive=False)
    # seq2seq_model.make_predictions(dataset, 'pres')


if __name__ == '__main__':
    main()
