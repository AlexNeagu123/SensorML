import numpy as np
from matplotlib import pyplot as plt, dates
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from utils import check_for_air_temperature_disease, check_for_air_humidity_disease


class RnnModel:
    def __init__(self, training_hours, prediction_hours, hidden_units, layer_units, learning_rate, epochs, window):
        self.training_hours = training_hours
        self.prediction_hours = prediction_hours
        self.hidden_units = hidden_units
        self.layer_units = layer_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.window = window

    def _data_loader(self, df):
        inputs = list()
        expected_outputs = list()

        for i in range(len(df) - self.window):
            inputs.append(df[i:i+self.window])
            expected_outputs.append(df[i + self.window])

        return np.array(inputs), np.array(expected_outputs)

    def build_model(self):
        model = Sequential()
        model.add(InputLayer((self.window, 1)))
        model.add(LSTM(self.hidden_units))
        model.add(Dense(self.layer_units, 'relu'))
        model.add(Dense(1, 'linear'))

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

        inputs, expected_outputs = self._data_loader(variable_data)

        train_inputs, train_outputs = inputs[:self.training_hours], expected_outputs[:self.training_hours]
        test_inputs = inputs[self.training_hours:(self.training_hours + self.prediction_hours)]
        test_outputs = expected_outputs[self.training_hours:(self.training_hours + self.prediction_hours)]

        rnn_model = self.build_model()
        rnn_model.fit(train_inputs, train_outputs, epochs=self.epochs, callbacks=[ModelCheckpoint('models/',
                                                                                                  save_best_only=True)])

        test_predictions = rnn_model.predict(test_inputs).flatten()
        train_predictions = rnn_model.predict(train_inputs).flatten()

        plt.scatter(dataset['Timestamp'][:self.training_hours],
                    original_variable_data[:self.training_hours].flatten(),
                    color='black', label='Training Data', s=10)

        test_predictions_2d = test_predictions.reshape(-1, 1)
        train_predictions_2d = train_predictions.reshape(-1, 1)

        plt.plot(dataset['Timestamp'][:self.training_hours + self.prediction_hours],
                 scaler.inverse_transform(np.concatenate([train_predictions_2d, test_predictions_2d])),
                 color='blue', label='Predicted')

        plt.scatter(dataset['Timestamp'][self.training_hours:self.training_hours + self.prediction_hours],
                    scaler.inverse_transform(test_outputs),
                    color='red', label='Actual', s=10)

        plt.xlabel('ds')
        plt.ylabel('y')
        plt.title(f"Forecast for {variable}")
        plt.legend()
        plt.savefig(f"plots/plot.png")
        plt.close()

        if variable == 'temp1' or variable == 'temp2':
            predicted_temps = scaler.inverse_transform(np.concatenate([train_predictions_2d, test_predictions_2d]))
            mean_temp = sum(predicted_temps) / len(predicted_temps)
            return check_for_air_temperature_disease(mean_temp)

        if variable == 'umid':
            predicted_temps = scaler.inverse_transform(np.concatenate([train_predictions_2d, test_predictions_2d]))
            mean_temp = sum(predicted_temps) / len(predicted_temps)
            return check_for_air_humidity_disease(mean_temp)
