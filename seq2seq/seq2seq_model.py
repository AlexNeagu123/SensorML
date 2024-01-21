import numpy as np
from tensorflow.keras.layers import Input, RNN, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class Seq2SeqModel:
    def __init__(self, training_hours, prediction_hours, hidden_units, layer_units, learning_rate, epochs, window_enc, window_dec):
        self.training_hours = training_hours
        self.prediction_hours = prediction_hours
        self.hidden_units = hidden_units
        self.layer_units = layer_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.window_enc = window_enc
        self.window_dec = window_dec

    def _data_loader(self, df):
        encoder_inputs, decoder_inputs, expected_outputs = [], [], []
        for i in range(len(df) - self.window_enc - self.window_dec):
            encoder_inputs.append(df[i:i+self.window_enc])
            decoder_inputs.append(df[i+self.window_enc-self.window_dec:i+self.window_enc])
            expected_outputs.append(df[i+self.window_enc])
        return np.array(encoder_inputs), np.array(decoder_inputs), np.array(expected_outputs)

    def build_model(self):
        encoder_inputs = Input(shape=(self.window_enc, 1), name='encoder_input')
        lstm_encoder = LSTM(self.hidden_units, return_sequences=True, return_state=True, name='encoder_lstm')
        encoder_outputs, state_h, state_c = lstm_encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(self.window_dec, 1), name='decoder_input')
        lstm_decoder = LSTM(self.hidden_units, return_sequences=False, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = lstm_decoder(decoder_inputs, initial_state=encoder_states)

        output_layer = Dense(1)
        model_outputs = output_layer(decoder_outputs)

        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=model_outputs)
        model.compile(loss=MeanSquaredError(), optimizer=Adam(self.learning_rate), metrics=[RootMeanSquaredError()])
        return model

    def make_predictions(self, dataset, variable):
        variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']
        if variable not in variables_to_plot:
            return

        original_variable_data = dataset[variable].copy()
        original_variable_data = original_variable_data.values.reshape(-1, 1)

        scaler = MinMaxScaler()
        variable_data = scaler.fit_transform(original_variable_data)
        encoder_inputs, decoder_inputs, expected_outputs = self._data_loader(variable_data)

        enc_train_inputs, dec_train_inputs = encoder_inputs[:self.training_hours], decoder_inputs[:self.training_hours]
        enc_test_inputs = encoder_inputs[self.training_hours:(self.training_hours+self.prediction_hours)]
        dec_test_inputs = decoder_inputs[self.training_hours:(self.training_hours+self.prediction_hours)]

        train_outputs = expected_outputs[:self.training_hours]
        test_outputs = expected_outputs[self.training_hours:(self.training_hours+self.prediction_hours)]

        seq2seq_model = self.build_model()

        seq2seq_model.fit([enc_train_inputs, dec_train_inputs], train_outputs,
                          epochs=self.epochs, callbacks=[ModelCheckpoint('models/', save_best_only=True)])

        test_predictions = seq2seq_model.predict([enc_test_inputs, dec_test_inputs])
        train_predictions = seq2seq_model.predict([enc_train_inputs, dec_train_inputs])

        test_predictions_2d = test_predictions.reshape(-1, 1)
        train_predictions_2d = train_predictions.reshape(-1, 1)

        plt.figure(figsize=(12, 6))

        plt.scatter(dataset['Timestamp'][:self.training_hours],
                    original_variable_data[:self.training_hours].flatten(),
                    color='black', label='Training Data', s=10)

        plt.plot(dataset['Timestamp'][:self.training_hours+self.prediction_hours],
                 scaler.inverse_transform(np.concatenate([train_predictions_2d, test_predictions_2d])),
                 color='blue', label='Predicted')

        plt.scatter(dataset['Timestamp'][self.training_hours:self.training_hours+self.prediction_hours],
                    scaler.inverse_transform(test_outputs).flatten(),
                    color='red', label='Actual', s=10)

        plt.xlabel('ds')
        plt.ylabel('y')
        plt.title(f"Forecast for {variable}")
        plt.legend()
        plt.xticks(rotation=-90)
        plt.savefig(f"plots/plot.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()
