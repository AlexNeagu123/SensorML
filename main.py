import pandas as pd

from prophet_func import *


def main():
    dataset = pd.read_csv("dataset/SensorMLDataset.csv")
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    # heatmaps(dataset)
    # create_box_plots(dataset)
    # correlation_matrix(dataset)
    make_predictions(dataset)
    # cross_validate(dataset, initial=168, period=24, horizon=48)


if __name__ == '__main__':
    main()
