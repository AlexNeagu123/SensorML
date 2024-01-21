import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from nn_predictions.nn_model import NnModel
from sklearn.metrics import mean_squared_error


class RnnModel(NnModel):
    def __init__(self, training_hours, prediction_hours, hidden_units, layer_units,
                 learning_rate, epochs, window, autoregressive=False):
        super().__init__(training_hours, prediction_hours, hidden_units, layer_units, learning_rate,
                         epochs, autoregressive)
        self.window = window

    def _data_loader(self, df):
        inputs = list()
        expected_outputs = list()

        for i in range(len(df) - self.window):
            inputs.append(df[i:i+self.window])
            expected_outputs.append(df[i+self.window])

        return np.array(inputs), np.array(expected_outputs)

    def _build_model(self):
        model = Sequential()
        model.add(InputLayer((self.window, 1)))
        model.add(LSTM(self.hidden_units))
        model.add(Dense(self.layer_units, 'relu'))
        model.add(Dense(1, 'linear'))

        model.compile(loss=MeanSquaredError(), optimizer=Adam(self.learning_rate), metrics=[RootMeanSquaredError()])
        return model

    def get_test_predictions(self, model, train_inputs, test_inputs, original_variable_data, expected_outputs):
        test_predictions = list()

        if self.autoregressive:
            first_batch = train_inputs[-1]
            current_batch = first_batch.reshape((1, self.window, 1))

            for i in range(self.prediction_hours):
                current_prediction = model.predict(current_batch)[0]
                test_predictions.append(current_prediction)
                current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis=1)
            actual_testing_values = original_variable_data[self.training_hours:self.training_hours+self.prediction_hours]
        else:
            test_predictions = model.predict(test_inputs)
            actual_testing_values = self.scaler.inverse_transform(expected_outputs[self.training_hours:self.training_hours+self.prediction_hours])

        testing_error = mean_squared_error(self.scaler.inverse_transform(test_predictions), actual_testing_values)
        return test_predictions, testing_error

    def make_predictions(self, dataset, variable):
        variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']

        if variable not in variables_to_plot:
            return

        original_variable_data = dataset[variable].copy()
        original_variable_data = original_variable_data.values.reshape(-1, 1)

        variable_data = self.scaler.fit_transform(original_variable_data)

        inputs, expected_outputs = self._data_loader(variable_data)

        train_inputs, train_outputs = inputs[:self.training_hours], expected_outputs[:self.training_hours]
        test_inputs = inputs[self.training_hours:self.training_hours+self.prediction_hours]

        rnn_model = self._build_model()
        rnn_model.fit(train_inputs, train_outputs, epochs=self.epochs, callbacks=[ModelCheckpoint('models/',
                                                                                                  save_best_only=True)])

        test_predictions, training_error = self.get_test_predictions(rnn_model, train_inputs, test_inputs,
                                                                     original_variable_data, expected_outputs)
        train_predictions = rnn_model.predict(train_inputs).flatten()

        return self._make_plot(dataset, original_variable_data, test_predictions, train_predictions, variable), training_error
