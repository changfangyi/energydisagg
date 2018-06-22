#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
from lib import dirs
import importlib
import numpy as np
import pandas as pd
from dataprocess.source import RealSource
from dataprocess.source import SynSource
from metrics import Metrics
from datetime import timedelta
from time import strftime

# Configuration
PATH = '/Users/kang/Desktop/energydisagg' # multi_group
#PATH = '/home/nilm/Desktop/energydisagg' # multi_group
APPLIANCES = None
CHANNELS = None
HOUSES = None
NUM_STEPS = None
FREQ_REAL_SYN = 2

def main():
    os.chdir(PATH)
    data_path = os.path.join(PATH, 'data', 'multi_group')
    parse_args()
    load_config()
    data_to_memory, house_prob, activation_prob = load_data(HOUSES, data_path)
    print('Get Batch for Training:')
    real_source = RealSource(data_to_memory = data_to_memory, channels = CHANNELS, seq_length=60, 
                        houses = HOUSES, houses_prob  = house_prob, activations_prob = activation_prob)
    syn_source = RealSource(data_to_memory = data_to_memory, channels = CHANNELS, seq_length=60, 
                        houses = HOUSES, houses_prob  = house_prob, activations_prob = activation_prob)

    topology_module = importlib.import_module(dirs.TOPOLOGIES_DIR + '.' + 'multi_CLDNN', __name__)
    model = topology_module.build_model(input_shape=(60,1), appliances= CHANNELS[1:])
    for i in range(NUM_STEPS):
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
            #print('Random Guess :')
            #for item, channel in enumerate(CHANNELS[1:]):
            #    print(channel, ':')
            #    metrics = Metrics(state_boundaries=[15], clip_to_zero=True)
            #    target_dim = len(targets[channel].flatten())
            #    scores = metrics.compute_metrics(np.zeros(target_dim), targets[channel].flatten())
            #    for valid_type, score in scores.iteritems():
            #        for metrics_name, value in score.iteritems():
            #            print(metrics_name, ': {:.2f}'.format(value))
            #    print('\n')
    model_name = strftime('%Y%m%d_%H')
    for item in CHANNELS[1:]:
        model_name = model_name + '_' + item
    print('Saving model ', model_name, '.h5')
    model.save(os.path.join(PATH, 'models', 'config_' + model_name + '.h5'))

def parse_args():
    global APPLIANCES, NUM_STEPS
    parser = argparse.ArgumentParser()
     # required
    required_named_arguments = parser.add_argument_group('required named arguments')
    required_named_arguments.add_argument('-a', '--appliancess',
                                          help='FB fridge and bottle warmer',
                                          required=True)
    required_named_arguments.add_argument('-t', '--num-steps',
                                          help='Number of steps.',
                                          type=int,
                                          required=True)
     # start parsing
    args = parser.parse_args()
    APPLIANCES = args.appliancess
    NUM_STEPS = args.num_steps

def load_config():
    global CHANNELS, HOUSES 
    config_module = importlib.import_module(dirs.CONFIG_DIR + '.' + 'train', __name__)
    if APPLIANCES == 'FB':
        HOUSES = config_module.FB
        HOUSES = HOUSES['house']
        CHANNELS = ['main','fridge','bottle warmer']
    elif APPLIANCES == 'FA':
        HOUSES = config_module.FA
        HOUSES = HOUSES['house']
        CHANNELS = ['main','fridge','air conditioner']
    elif APPLIANCES == 'FT':
        HOUSES = config_module.FT
        HOUSES = HOUSES['house']
        CHANNELS = ['main','fridge','television']
    elif APPLIANCES == 'FW':
        HOUSES = config_module.FT
        HOUSES = HOUSES['house']
        CHANNELS = ['main','fridge','washing machine']
        

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

if __name__ == '__main__':
    main()
