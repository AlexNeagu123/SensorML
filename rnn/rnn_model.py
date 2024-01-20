import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from globals import TRAINING_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE, RNN_EPOCHS, PREDICTION_HOURS


def data_loader(df_as_np, window=5):
    # df_as_np = df.to_numpy()
    inputs = list()
    expected_outputs = list()

    for i in range(len(df_as_np) - window):
        row = [[a] for a in df_as_np[i:i + window]]
        inputs.append(row)
        expected_outputs.append(df_as_np[i + window])

    return np.array(inputs), np.array(expected_outputs)


def make_predictions(dataset, window):
    variables_to_plot = [col for col in dataset.columns if col != 'Timestamp']
    for variable in variables_to_plot:
        original_variable_data = dataset[variable].copy()
        original_variable_data = original_variable_data.values.reshape(-1, 1)

        scaler = MinMaxScaler()
        variable_data = scaler.fit_transform(original_variable_data)

        inputs, expected_outputs = data_loader(variable_data, window)

        train_inputs, train_outputs = inputs[:TRAINING_HOURS], expected_outputs[:TRAINING_HOURS]
        test_inputs = inputs[TRAINING_HOURS:(TRAINING_HOURS + PREDICTION_HOURS)]
        test_outputs = expected_outputs[TRAINING_HOURS:(TRAINING_HOURS + PREDICTION_HOURS)]

        rnnModel = Sequential()
        rnnModel.add(InputLayer((window, 1)))
        rnnModel.add(LSTM(RNN_HIDDEN_UNITS))
        rnnModel.add(Dense(RNN_LAYER_UNITS, 'relu'))
        rnnModel.add(Dense(1, 'linear'))

        cp = ModelCheckpoint('model1/', save_best_only=True)
        rnnModel.compile(loss=MeanSquaredError(), optimizer=Adam(RNN_LEARNING_RATE), metrics=[RootMeanSquaredError()])
        rnnModel.fit(train_inputs, train_outputs,
                     epochs=RNN_EPOCHS, callbacks=[cp])

        test_predictions = rnnModel.predict(test_inputs).flatten()
        train_predictions = rnnModel.predict(train_inputs).flatten()

        plt.scatter(dataset['Timestamp'][:TRAINING_HOURS], original_variable_data[:TRAINING_HOURS].flatten(),
                    color='black', label='Training Data', s=10)

        test_predictions_2d = test_predictions.reshape(-1, 1)
        train_predictions_2d = train_predictions.reshape(-1, 1)

        plt.plot(dataset['Timestamp'][:TRAINING_HOURS + PREDICTION_HOURS],
                 scaler.inverse_transform(np.concatenate([train_predictions_2d, test_predictions_2d])),
                 color='blue', label='Predicted')

        plt.scatter(dataset['Timestamp'][TRAINING_HOURS:TRAINING_HOURS + PREDICTION_HOURS],
                    scaler.inverse_transform(test_outputs),
                    color='red', label='Actual', s=10)

        plt.xlabel('ds')
        plt.ylabel('y')
        plt.title(f"Forecast for {variable}")
        plt.legend()
        plt.show()
