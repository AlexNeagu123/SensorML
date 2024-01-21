import numpy as np
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from nn_predictions.nn_model import NnModel


class Seq2SeqModel(NnModel):
    def __init__(self, training_hours, prediction_hours, hidden_units, layer_units, learning_rate, epochs, window_enc,
                 window_dec, autoregressive=False):
        super().__init__(training_hours, prediction_hours, hidden_units, layer_units, learning_rate,
                         epochs, autoregressive)
        self.window_enc = window_enc
        self.window_dec = window_dec

    def _data_loader(self, df):
        encoder_inputs, decoder_inputs, expected_outputs = [], [], []
        for i in range(len(df) - self.window_enc - self.window_dec):
            encoder_inputs.append(df[i:i + self.window_enc])
            decoder_inputs.append(df[i + self.window_enc - self.window_dec:i + self.window_enc])
            expected_outputs.append(df[i + self.window_enc])
        return np.array(encoder_inputs), np.array(decoder_inputs), np.array(expected_outputs)

    def _build_model(self):
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

    def get_test_predictions(self, model, enc_train_inputs, dec_train_inputs,
                             enc_test_inputs, dec_test_inputs, original_variable_data, expected_outputs):
        test_predictions = []
        if self.autoregressive:
            first_enc_batch = enc_train_inputs[-1]
            first_dec_batch = dec_train_inputs[-1]

            current_enc_batch = first_enc_batch.reshape((1, self.window_enc, 1))
            current_dec_batch = first_dec_batch.reshape((1, self.window_dec, 1))

            for i in range(self.prediction_hours):
                current_prediction = model.predict([current_enc_batch, current_dec_batch])[0]
                test_predictions.append(current_prediction)
                current_enc_batch = np.append(current_enc_batch[:, 1:, :], [[current_prediction]], axis=1)
                current_dec_batch = np.append(current_dec_batch[:, 1:, :], [[current_prediction]], axis=1)
            actual_testing_values = original_variable_data[self.training_hours:self.training_hours+self.prediction_hours]
        else:
            test_predictions = model.predict([enc_test_inputs, dec_test_inputs])
            actual_testing_values = self.scaler.inverse_transform(
                expected_outputs[self.training_hours:self.training_hours+self.prediction_hours])

        testing_error = mean_squared_error(self.scaler.inverse_transform(test_predictions), actual_testing_values)
        return test_predictions, testing_error

    def make_predictions(self, dataset, variable):
        variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']
        if variable not in variables_to_plot:
            return

        original_variable_data = dataset[variable].copy()
        original_variable_data = original_variable_data.values.reshape(-1, 1)

        variable_data = self.scaler.fit_transform(original_variable_data)
        encoder_inputs, decoder_inputs, expected_outputs = self._data_loader(variable_data)

        enc_train_inputs, dec_train_inputs = encoder_inputs[:self.training_hours], decoder_inputs[:self.training_hours]
        enc_test_inputs = encoder_inputs[self.training_hours:(self.training_hours + self.prediction_hours)]
        dec_test_inputs = decoder_inputs[self.training_hours:(self.training_hours + self.prediction_hours)]

        train_outputs = expected_outputs[:self.training_hours]
        seq2seq_model = self._build_model()

        seq2seq_model.fit([enc_train_inputs, dec_train_inputs], train_outputs,
                          epochs=self.epochs, callbacks=[ModelCheckpoint('models/', save_best_only=True)])

        test_predictions, testing_error = self.get_test_predictions(seq2seq_model, enc_train_inputs, dec_train_inputs,
                                                                    enc_test_inputs, dec_test_inputs,
                                                                    original_variable_data, expected_outputs)

        train_predictions = seq2seq_model.predict([enc_train_inputs, dec_train_inputs])

        return self._make_plot(dataset, original_variable_data, test_predictions, train_predictions, variable), testing_error
