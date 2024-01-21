import pandas as pd

from data_analysis.analysis import correlation_matrix, create_box_plots
from globals import *
from flask import Flask, send_file
from imgur_python import Imgur

from prophet_predictions.prophet_model import ProphetModel
from rnn.rnn_model import RnnModel

app = Flask(__name__)
allowed_types = ['prophet', 'rnn']
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


def make_predictions(type, training_hours, prediction_hours, variable_name):
    dataset = pd.read_csv("dataset/SensorMLDataset.csv")
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

    if type == 'prophet':
        model = ProphetModel(training_hours, prediction_hours)
    elif type == 'rnn':
        model = RnnModel(TRAINING_HOURS, PREDICTION_HOURS, RNN_HIDDEN_UNITS, RNN_LAYER_UNITS, RNN_LEARNING_RATE,
                         RNN_EPOCHS, 5)
    else:
        model = ProphetModel(training_hours, prediction_hours)

    return model.make_predictions(dataset, variable_name)


def get_link_predict(type):
    imgur_client = Imgur({'client_id': IMGUR_CLIENT_ID})
    if type == 'boxplots':
        iamge_url = imgur_client.image_upload('plots/box_plots.png', title='Box Plots',
                                              description='Box Plots')['response']['data']['link']
    elif type == 'correlation-matrix':
        iamge_url = imgur_client.image_upload('plots/cor_matrix.png', title='Correlation Matrix',
                                              description='Correlation Matrix')['response']['data']['link']
    else:
        iamge_url = imgur_client.image_upload('plots/plot.png', title='Plot',
                                              description='Plot')['response']['data']['link']
    return iamge_url


if __name__ == '__main__':
    app.run()
