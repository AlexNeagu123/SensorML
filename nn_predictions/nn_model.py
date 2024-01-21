import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from utils import check_for_air_temperature_disease, check_for_air_humidity_disease


class NnModel:
    def __init__(self, training_hours, prediction_hours, hidden_units, layer_units,
                 learning_rate, epochs, autoregressive=False):
        self.training_hours = training_hours
        self.prediction_hours = prediction_hours
        self.hidden_units = hidden_units
        self.layer_units = layer_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.autoregressive = autoregressive
        self.scaler = MinMaxScaler()

    def _make_plot(self, dataset, original_variable_data, test_predictions, train_predictions, variable):
        test_predictions_2d = np.array(test_predictions).reshape(-1, 1)
        train_predictions_2d = train_predictions.reshape(-1, 1)

        plt.figure(figsize=(12, 6))
        plt.scatter(dataset['Timestamp'][:self.training_hours],
                    original_variable_data[:self.training_hours].flatten(),
                    color='black', label='Training Data', s=10)

        plt.plot(dataset['Timestamp'][:self.training_hours + self.prediction_hours],
                 self.scaler.inverse_transform(np.concatenate([train_predictions_2d, test_predictions_2d])),
                 color='blue', label='Predicted')

        plt.scatter(dataset['Timestamp'][self.training_hours:self.training_hours + self.prediction_hours],
                    original_variable_data[self.training_hours:self.training_hours + self.prediction_hours],
                    color='red', label='Actual', s=10)

        plt.xlabel('ds')
        plt.ylabel('y')
        plt.title(f"Forecast for {variable}")
        plt.legend()
        plt.xticks(rotation=-90)
        plt.savefig(f"plots/plot.png", bbox_inches='tight', pad_inches=0.1)
        plt.close()

        if variable == 'temp1' or variable == 'temp2':
            predicted_temps = self.scaler.inverse_transform(np.concatenate([train_predictions_2d, test_predictions_2d]))
            mean_temp = sum(predicted_temps) / len(predicted_temps)
            return check_for_air_temperature_disease(mean_temp)

        if variable == 'umid':
            predicted_temps = self.scaler.inverse_transform(np.concatenate([train_predictions_2d, test_predictions_2d]))
            mean_temp = sum(predicted_temps) / len(predicted_temps)
            return check_for_air_humidity_disease(mean_temp)

        return []