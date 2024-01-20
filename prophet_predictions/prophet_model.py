import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


class ProphetModel:
    def __init__(self, training_hours, prediction_hours):
        self.training_hours = training_hours
        self.prediction_hours = prediction_hours

    def make_predictions(self, dataset):
        variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']
        for variable in variables_to_plot:
            variable_data = dataset[['Timestamp', variable]].copy()
            variable_data.columns = ['ds', 'y']

            model = Prophet()
            model.fit(variable_data.head(self.training_hours))

            future = model.make_future_dataframe(periods=self.prediction_hours, freq='h')
            forecast = model.predict(future)
            model.plot(forecast)

            plt.scatter(variable_data['ds'][self.training_hours:self.training_hours + self.prediction_hours],
                        variable_data['y'][self.training_hours:self.training_hours + self.prediction_hours],
                        color='red', s=10)

            plt.title(f"Forecast for {variable}")
            plt.savefig(f"plots/{variable}.png")
            plt.close()


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