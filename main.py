import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        sequences.append(seq)
    return np.array(sequences)


def main():
    df = pd.read_csv("dataset/SensorMLDataset.csv")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df.drop('Timestamp', axis=1, inplace=True)
    # df = [col for col in df.columns if col != 'Timestamp']
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    seq_length = 10
    X = create_sequences(df_scaled, seq_length)
    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(seq_length, df.shape[1])))
    model.add(RepeatVector(seq_length))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(df.shape[1])))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.1)

    loss = model.evaluate(X_test, X_test)
    print(f'Mean Squared Error on Test Data: {loss}')

    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, df.shape[1]))
    print(predictions_rescaled)


if __name__ == '__main__':
    main()
