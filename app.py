import pandas as pd
import requests
from data_analysis.analysis import correlation_matrix, create_box_plots
from globals import *
from flask import Flask, send_file

from prophet_predictions.prophet_model import ProphetModel
from rnn.rnn_model import RnnModel
from seq2seq.seq2seq_model import Seq2SeqModel

app = Flask(__name__)
allowed_types = ['prophet', 'rnn_auto', 'rnn', 'seq2seq', 'seq2seq_auto']
allowed_variables = ["Timestamp", 'temp1', 'pres', 'umid', 'temp2', 'V450', 'B500', 'G550', 'Y570', 'O600', 'R650',
                     'temps1', 'temps2', 'lumina']


@app.route('/')
def hello_world():
    return 'ciau'


@app.route('/box-plots', methods=['GET'])
def get_boxplots():
    try:
        dataset = pd.read_csv("dataset/SensorMLDataset.csv")
        dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
        create_box_plots(dataset)
        image_url = get_link_predict('boxplots')

    except Exception as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'link': image_url}


@app.route('/correlation-matrix', methods=['GET'])
def get_correlation_matrix():
    try:
        dataset = pd.read_csv("dataset/SensorMLDataset.csv")
        dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
        correlation_matrix(dataset)
        image_url = get_link_predict('correlation-matrix')

    except Exception as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'link': image_url}


@app.route('/<type_predict>/<variable_name>/<training_hours>/<prediction_hours>', methods=['GET'])
def get_predict(type_predict, training_hours, prediction_hours, variable_name):
    if type_predict not in allowed_types:
        return {'ok': False, 'error': 'Invalid type'}
    if variable_name not in allowed_variables:
        return {'ok': False, 'error': 'Invalid variable name'}
    try:
        diseases = make_predictions(type_predict, int(training_hours), int(prediction_hours), variable_name)
        image_url = get_link_predict("plot")
    except Exception as e:
        return {'ok': False, 'error': str(e)}
    return {'ok': True, 'link': image_url, 'diseases': str(diseases)}


def make_predictions(model_type, training_hours, prediction_hours, variable_name):
    dataset = pd.read_csv("dataset/SensorMLDataset.csv")
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

    if model_type == 'prophet':
        model = ProphetModel(training_hours, prediction_hours)
    elif model_type == 'rnn_auto':
        model = RnnModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
                         RNN_EPOCHS, 5, True)
    elif model_type == 'rnn':
        model = RnnModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
                         RNN_EPOCHS, 5, False)
    elif model_type == 'seq2seq_auto':
        model = Seq2SeqModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
                             RNN_EPOCHS, 5, 3, True)
    elif model_type == 'seq2seq':
        model = Seq2SeqModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
                             RNN_EPOCHS, 5, 3, False)
    else:
        model = ProphetModel(training_hours, prediction_hours)

    return model.make_predictions(dataset, variable_name)


def upload_image(file_path, title, description):
    upload_url = 'https://api.imgur.com/3/upload'
    headers = {'Authorization': 'Client-ID ' + IMGUR_CLIENT_ID}

    with open(file_path, 'rb') as file:
        files = {'image': file}
        data = {'title': title, 'description': description}

        response = requests.post(upload_url, headers=headers, data=data, files=files)

    if response.status_code == 200:
        return response.json()['data']['link']
    else:
        print(f"Image upload failed. Status code: {response.status_code}")
        return None

def get_link_predict(plot_type):
    if plot_type == 'boxplots':
        image_url = upload_image('plots/box_plots.png', 'Box Plots', 'Box Plots')
    elif plot_type == 'correlation-matrix':
        image_url = upload_image('plots/cor_matrix.png', 'Correlation Matrix', 'Correlation Matrix')
    else:
        image_url = upload_image('plots/plot.png', 'Plot', 'Plot')

    return image_url


if __name__ == '__main__':
    app.run()
