import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_squared_error

from utils import *


class ProphetModel:
    def __init__(self, training_hours, prediction_hours):
        self.training_hours = training_hours
        self.prediction_hours = prediction_hours

    def make_predictions(self, dataset, variable):
        variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']

        if variable not in variables_to_plot:
            return

        variable_data = dataset[['Timestamp', variable]].copy()
        variable_data.columns = ['ds', 'y']

        actual_values = variable_data['y'][self.training_hours:self.training_hours+self.prediction_hours].values

        model = Prophet()
        model.fit(variable_data.head(self.training_hours))

        future = model.make_future_dataframe(periods=self.prediction_hours, freq='h')
        forecast = model.predict(future)
        model.plot(forecast)

        predicted_values = forecast['yhat'].tail(self.prediction_hours).tolist()
        testing_error = mean_squared_error(actual_values, predicted_values)

        plt.scatter(variable_data['ds'][self.training_hours:self.training_hours+self.prediction_hours],
                    variable_data['y'][self.training_hours:self.training_hours+self.prediction_hours],
                    color='red', s=10)

        plt.title(f"Forecast for {variable}")
        plt.savefig(f"plots/plot.png")
        plt.close()

        if variable == 'temp1' or variable == 'temp2':
            predicted_temps = forecast['yhat'].tail(self.prediction_hours).tolist()
            mean_temp = sum(predicted_temps) / len(predicted_temps)
            return check_for_air_temperature_disease(mean_temp), testing_error

        if variable == 'umid':
            predicted_temps = forecast['yhat'].tail(self.prediction_hours).tolist()
            mean_temp = sum(predicted_temps) / len(predicted_temps)
            return check_for_air_humidity_disease(mean_temp), testing_error

        return [], testing_error


def cross_validate(dataset, initial, horizon, period):
    variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']
    for variable in variables_to_plot:
        variable_data = dataset[['Timestamp', variable]].copy()
        variable_data.columns = ['ds', 'y']

        model = Prophet()
        model.fit(variable_data)
        dataset_cross_validation = cross_validation(model, initial=f'{initial} hours', period=f'{period} hours',
                                                    horizon=f'{horizon} hours')
        performance = performance_metrics(dataset_cross_validation, rolling_window=0)

        output_file_path = f'{variable}.csv'
        performance.to_csv(output_file_path, index=False)
        print(f'Performance metrics saved to {output_file_path}')