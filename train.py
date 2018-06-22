#!/usr/bin/env python
from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
from dataprocess.source import RealSource
from dataprocess.source import SynSource
from metrics import Metrics
from datetime import timedelta
from time import strftime


# Configuration
PATH = '/Users/kang/Desktop/energydisagg' # multi_group
CHANNELS = ['main','fridge','air conditioner']
HOUSES = [14, 39]
HOUSES_PROB = []
TRAIN_STEP = 10
FREQ_REAL_SYN = 2


def main():
    os.chdir(PATH)
    data_path = os.path.join(PATH, 'data', 'multi_group')
    data_to_memory, house_prob, activation_prob = load_data(HOUSES, data_path)
    print('Get Batch for Training:')
    real_source = RealSource(data_to_memory = data_to_memory, channels = CHANNELS, seq_length=60, 
                        houses = HOUSES, houses_prob  = house_prob, activations_prob = activation_prob)
    syn_source = RealSource(data_to_memory = data_to_memory, channels = CHANNELS, seq_length=60, 
                        houses = HOUSES, houses_prob  = house_prob, activations_prob = activation_prob)

    model = build_model(input_shape=(60,1), channels= CHANNELS[1:])
    for i in range(TRAIN_STEP):
        if i % FREQ_REAL_SYN == 0:
            main, targets = real_source._get_batch()
            while main is None or targets is None:
                main, targets = real_source._get_batch()
        else:
            main, targets = syn_source._get_batch()
            while main is None or targets is None:
                main, targets = syn_source._get_batch()

        train_metrics = model.train_on_batch(x=main,y=[targets[CHANNELS[1]], targets[CHANNELS[2]]]) 

        if i % 100 == 0:
            print('Step : {} , Time : {}\n'.format(i,strftime('%Y-%m-%d_%H_%M')))
            #for i, metrics_name in enumerate(model.metrics_names):
            #    print('{}={:.2f}, '.format(metrics_name, train_metrics[i]))
            #print('\n')

            prediction = model.predict_on_batch(main)
            print('Inference Guess :')
            for item, channel in enumerate(CHANNELS[1:]):
                print(channel, ':')
                metrics = Metrics(state_boundaries=[15], clip_to_zero=True)
                scores = metrics.compute_metrics(prediction[item].flatten(), targets[channel].flatten())
                for valid_type, score in scores.iteritems():
                    for metrics_name, value in score.iteritems():
                        print(metrics_name, ': {:.2f}'.format(value))
                print('\n')

            print('Random Guess :')
            for item, channel in enumerate(CHANNELS[1:]):
                print(channel, ':')
                metrics = Metrics(state_boundaries=[15], clip_to_zero=True)
                target_dim = len(targets[channel].flatten())
                scores = metrics.compute_metrics(np.zeros(target_dim), targets[channel].flatten())
                for valid_type, score in scores.iteritems():
                    for metrics_name, value in score.iteritems():
                        print(metrics_name, ': {:.2f}'.format(value))
                print('\n')


    model_name = strftime('%Y%m%d_%H')
    for item in CHANNELS[1:]:
        model_name = model_name + '_' + item
    print('Saving model ', model_name, '.h5')
    model.save(os.path.join(PATH, 'models', 'config_' + model_name + '.h5'))
                
def load_data(house, path):
    collection = {}
    house_prob = []
    activation_prob = {}
    for item in sorted(house):
        pathfile = os.path.join(path, str(item))
        activation_counts= []
        activation_collection = {}
        activations = os.listdir(pathfile)
        for activation in activations:
            activation_data = pd.read_csv(pathfile + '/' + activation, index_col=0)           
            activation_counts.append(len(activation_data))
            activation_collection[str(activation[:-15])] = activation_data
        house_prob.append(sum(activation_counts))
        collection['house_'+str(item)] =  activation_collection
        activation_prob['house_'+ str(item)] = [i/sum(activation_counts) for i in activation_counts]
    house_prob = [i/sum(house_prob) for i in house_prob]
    return collection, house_prob, activation_prob

def build_model(input_shape, channels):
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Activation, Reshape, Flatten, Conv1D, MaxPooling1D, Dropout, LSTM, TimeDistributed, Bidirectional
    from keras.optimizers import RMSprop
    MODEL_CONV_FILTERS = 32
    MODEL_CONV_KERNEL_SIZE = 18
    MODEL_CONV_STRIDES = 1
    MODEL_CONV_PADDING = 'same'

    seq_length = 60
    # conv * 4
    x = Input(shape=input_shape)
    conv_1 = Conv1D(filters=MODEL_CONV_FILTERS, kernel_size=MODEL_CONV_KERNEL_SIZE, padding=MODEL_CONV_PADDING, activation='relu')(x)
    drop_1 = Dropout(0.12)(conv_1)
    conv_2 = Conv1D(filters=64, kernel_size=12, padding=MODEL_CONV_PADDING, activation='relu')(drop_1)
    drop_2 = Dropout(0.14)(conv_2)
    conv_3 = Conv1D(filters=128, kernel_size=7, padding=MODEL_CONV_PADDING, activation='relu')(drop_2)
    pool_3 = MaxPooling1D(pool_size=2)(conv_3)
    drop_3 = Dropout(0.18)(pool_3)
    conv_4 = Conv1D(filters=128, kernel_size=3, padding=MODEL_CONV_PADDING, activation='relu')(drop_3)
    pool_4 = MaxPooling1D(pool_size=2)(conv_4)
    drop_4 = Dropout(0.2)(pool_4)
    # reshape
    flat_4 = Flatten()(drop_4)
    dense_5 = Dense(1280, activation='relu')(flat_4)
    drop_5 = Dropout(0.16)(dense_5)
    dense_6 = Dense(960, activation='relu')(drop_5)
    drop_6 = Dropout(0.14)(dense_6) 
    dense_7 = Dense(720, activation='relu')(drop_6)
    drop_7 = Dropout(0.12)(dense_7)
    reshape_8 = Reshape(target_shape=(seq_length, 12))(drop_7)
    # Initialization
    outputs_disaggregation = []
    for appliance_name in channels:
        biLSTM_1 = Bidirectional(LSTM(6, return_sequences=True))(reshape_8)
        biLSTM_2 = Bidirectional(LSTM(3, return_sequences=True))(biLSTM_1)
        outputs_disaggregation.append(TimeDistributed(Dense(1, activation='relu'), name=appliance_name.replace(" ", "_"))(biLSTM_2))

    model = Model(inputs=x, outputs=outputs_disaggregation)
    optimizer = RMSprop(lr=0.001, clipnorm=4)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
    return model

    
    

if __name__ == '__main__':
    main()
