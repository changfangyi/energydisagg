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
from dataprocess.validation import Validation
from dataprocess.table_process import load_data
from metrics import Metrics
from datetime import timedelta
from time import strftime

# Configuration
#PATH = '/Users/kang/Desktop/energydisagg' # multi_group
PATH = '/home/nilm/Desktop/energydisagg' # multi_group
APPLIANCES = None
MODEL = None
CHANNELS = None
HOUSES = None
NUM_STEPS = None
SINGLE = None
DATA = None
FREQ_REAL_SYN = 2

def main():
    os.chdir(PATH)
    parse_args()
    data_path = os.path.join(PATH, 'data', DATA)
    load_config()
    data_to_memory, house_prob, activation_prob = load_data(HOUSES, data_path)
    print('Get Batch for Training:')
    real_source = RealSource(data_to_memory = data_to_memory, channels = CHANNELS, seq_length=30, 
                        houses = HOUSES, houses_prob  = house_prob, activations_prob = activation_prob)
    syn_source = RealSource(data_to_memory = data_to_memory, channels = CHANNELS, seq_length=30, 
                        houses = HOUSES, houses_prob  = house_prob, activations_prob = activation_prob)
    topology_module = importlib.import_module(dirs.TOPOLOGIES_DIR + '.' + MODEL, __name__)
    model = topology_module.build_model(input_shape=(30,1), appliances= CHANNELS[1:])

    for i in range(NUM_STEPS):
        if i % FREQ_REAL_SYN == 0:
            main, targets = real_source._get_batch()
            while main is None or targets is None:
                main, targets = real_source._get_batch()
        else:
            main, targets = syn_source._get_batch()
            while main is None or targets is None:
                main, targets = syn_source._get_batch()

        model.train_on_batch(x=main,y=[targets[CHANNELS[item+1]] for item in range(len(CHANNELS[1:]))]) 

        if i % 1000 == 0:
            print('Step : {} , Time : {}\n'.format(i,strftime('%Y-%m-%d_%H_%M')))
            print('Inference Guess :')
            validate = Validation(main, targets, model, CHANNELS, SINGLE)
            validate._plot(os.path.join(PATH, 'fig'))
            validate._model_guess()

    model_name = str(DATA) + '_' + strftime('%Y%m%d_%H') + '_' + str(MODEL)
    for item in CHANNELS[1:]:
        model_name = model_name + '_' + item
    print('Saving model',  model_name, '.h5')
    model.save(os.path.join(PATH, 'models', model_name + '.h5'))

def parse_args():
    global APPLIANCES, NUM_STEPS, SINGLE, MODEL, DATA
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
    required_named_arguments.add_argument('-d', '--data',
                                          help='data name',
                                          required=True)
    required_named_arguments.add_argument('-m', '--model',
                                          help='model name',
                                          required=True)
    # optional
    optional_named_arguments = parser.add_argument_group('optional named arguments')
    optional_named_arguments.add_argument('-s', '--single',
                                          help='Flag to perform a single task',
                                          action='store_true')
     # start parsing
    args = parser.parse_args()
    APPLIANCES = args.appliancess
    NUM_STEPS = args.num_steps
    SINGLE = args.single
    MODEL = args.model
    DATA = args.data

def load_config():
    global CHANNELS, HOUSES 
    config_module = importlib.import_module(dirs.CONFIG_DIR + '.' + DATA, __name__)
    if APPLIANCES == 'FB':
        HOUSES = config_module.FB
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','bottle warmer']
    elif APPLIANCES == 'FA':
        HOUSES = config_module.FA
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','air conditioner']
    elif APPLIANCES == 'FT':
        HOUSES = config_module.FT
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','television']
    elif APPLIANCES == 'FW':
        HOUSES = config_module.FW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','washing machine']
    elif APPLIANCES == 'FK':
        HOUSES = config_module.FK
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','kettle']
    elif APPLIANCES == 'FM':
        HOUSES = config_module.FM
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','microwave']
    elif APPLIANCES == 'FD':
        HOUSES = config_module.FD
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','dish washer']
    elif APPLIANCES == 'BW':
        HOUSES = config_module.BW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','bottle warmer','washing machine']
    elif APPLIANCES == 'BT':
        HOUSES = config_module.BT
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','bottle warmer','television']
    elif APPLIANCES == 'FTB':
        HOUSES = config_module.FTB
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','television', 'bottle warmer']
    elif APPLIANCES == 'DW':
        HOUSES = config_module.DW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','dish washer','washing machine']
    elif APPLIANCES == 'MK':
        HOUSES = config_module.MK
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','microwave','kettle']
    elif APPLIANCES == 'MD':
        HOUSES = config_module.MD
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','microwave','dish washer']
    elif APPLIANCES == 'MW':
        HOUSES = config_module.MW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','microwave','washing machine']

    elif APPLIANCES == 'TW':
        HOUSES = config_module.TW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','television', 'washing machine']
    elif APPLIANCES == 'TA':
        HOUSES = config_module.TA
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','television','air conditioner']
    elif APPLIANCES == 'WA':
        HOUSES = config_module.WA
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','washing machine','air conditioner']
    elif APPLIANCES == 'TW':
        HOUSES = config_module.TW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','television','washing machine']
    elif APPLIANCES == 'TA':
        HOUSES = config_module.TW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','television','air conditioner']
    elif APPLIANCES == 'KW':
        HOUSES = config_module.KW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','kettle','washing machine']
    elif APPLIANCES == 'KD':
        HOUSES = config_module.KD
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','kettle','dish washer']
    elif APPLIANCES == 'F':
        HOUSES = config_module.F
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge']
    elif APPLIANCES == 'B':
        HOUSES = config_module.B
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','bottle warmer']
    elif APPLIANCES == 'A':
        HOUSES = config_module.A
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','air conditioner']
    elif APPLIANCES == 'T':
        HOUSES = config_module.T
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','television']
    elif APPLIANCES == 'W':
        HOUSES = config_module.W
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','washing machine']
    elif APPLIANCES == 'M':
        HOUSES = config_module.M
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','microwave']
    elif APPLIANCES == 'K':
        HOUSES = config_module.K
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','kettle']
    elif APPLIANCES == 'D':
        HOUSES = config_module.D
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','dish washer']

if __name__ == '__main__':
    main()
