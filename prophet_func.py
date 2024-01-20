import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from globals import TRAINING_HOURS, PREDICTION_HOURS, CV_RESULTS_FOLDER


def create_box_plots(dataset):
    variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']

    n_vars = len(variables_to_plot)
    n_cols = 5
    n_rows = n_vars // n_cols + (n_vars % n_cols > 0)

    plt.figure(figsize=(3 * n_cols, 3 * n_rows))

    for i, variable in enumerate(variables_to_plot, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(y=variable, data=dataset)
        plt.xticks([0], ['Entire Time Frame'])
        plt.title(f'Boxplot of "{variable}"')

    plt.tight_layout()
    plt.show()


def heatmaps(dataset):
    dataset_per_day = dataset.copy()
    dataset_per_day['Timestamp'] = dataset_per_day['Timestamp'].dt.date

    variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']

    for variable in variables_to_plot:
        mean_data = dataset_per_day.groupby('Timestamp')[variable].mean()
        median_data = dataset_per_day.groupby('Timestamp')[variable].median()

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(12, 16))

        sns.heatmap(pd.DataFrame(mean_data), annot=False, cmap='viridis', ax=ax1)
        ax1.set_title(f'Heatmap of Daily Mean Values for {variable}')

        sns.heatmap(pd.DataFrame(median_data), annot=False, cmap='viridis', ax=ax2)
        ax2.set_title(f'Heatmap of Daily Median Values for {variable}')

        plt.show()


def correlation_matrix(dataset):
    dataset = dataset.drop('Timestamp', axis=1)
    corr_matrix = dataset.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5,
                cbar_kws={"shrink": .5})

    plt.title('Correlation Matrix Heatmap')
    plt.show()


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

        output_file_path = f'{CV_RESULTS_FOLDER}/{variable}.csv'
        performance.to_csv(output_file_path, index=False)
        print(f'Performance metrics saved to {output_file_path}')


def make_predictions(dataset):
    variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']
    for variable in variables_to_plot:
        variable_data = dataset[['Timestamp', variable]].copy()
        variable_data.columns = ['ds', 'y']

        model = Prophet()
        model.fit(variable_data.head(TRAINING_HOURS))

        future = model.make_future_dataframe(periods=PREDICTION_HOURS, freq='h')
        forecast = model.predict(future)
        model.plot(forecast)

        plt.scatter(variable_data['ds'][TRAINING_HOURS:TRAINING_HOURS + PREDICTION_HOURS],
                    variable_data['y'][TRAINING_HOURS:TRAINING_HOURS + PREDICTION_HOURS],
                    color='red', s=10)

        plt.title(f"Forecast for {variable}")
        plt.show()
