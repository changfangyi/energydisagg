#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
from lib import dirs
import importlib
from dataprocess.source import RealSource
from dataprocess.validation import Validation
from keras.models import load_model
from dataprocess.table_process import load_data
#import random

#PATH = '/Users/kang/Desktop/energydisagg' # multi_group
PATH = '/home/nilm/Desktop/energydisagg' # multi_group
MODEL = None
APPLIANCES = None
CHANNELS = None
HOUSES = None
NUM_STEPS = None
SINGLE = None
FREQ_REAL_SYN = 2

def main():
    os.chdir(PATH)
    parse_args()
    load_config()
    model = os.path.join(PATH, 'models', MODEL + '.h5')
    model = load_model(model)
    data_to_memory, house_prob, activation_prob = load_data(HOUSES, os.path.join(PATH, 'data', 'multi_group'))
    real_source = RealSource(data_to_memory = data_to_memory, channels = CHANNELS, seq_length=60, 
                        houses = HOUSES, houses_prob  = house_prob, activations_prob = activation_prob)
    main, targets = real_source._get_batch()
    while main is None or targets is None:
        main, targets = real_source._get_batch()
    # validate
    validate = Validation(main, targets, model, CHANNELS, SINGLE)
    print(validate._zeros_guess())
    print(validate._model_guess())
    validate._plot(os.path.join(PATH, 'fig'))


def parse_args():
    global APPLIANCES, MODEL, SINGLE
    parser = argparse.ArgumentParser()
     # required
    required_named_arguments = parser.add_argument_group('required named arguments')
    required_named_arguments.add_argument('-a', '--appliancess',
                                          help='FB fridge and bottle warmer',
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
    MODEL = args.model
    SINGLE = args.single

def load_config():
    global CHANNELS, HOUSES 
    config_module = importlib.import_module(dirs.CONFIG_DIR + '.' + 'config', __name__)
    if APPLIANCES == 'FB':
        HOUSES = config_module.FB
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','fridge','bottle warmer']
    elif APPLIANCES == 'FA':
        HOUSES = config_module.FA
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','fridge','air conditioner']
    elif APPLIANCES == 'FT':
        HOUSES = config_module.FT
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','fridge','television']
    elif APPLIANCES == 'FW':
        HOUSES = config_module.FW
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','fridge','washing machine']
    elif APPLIANCES == 'F':
        HOUSES = config_module.F
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','fridge']
    elif APPLIANCES == 'B':
        HOUSES = config_module.B
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','bottle warmer']
    elif APPLIANCES == 'A':
        HOUSES = config_module.A
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','air conditioner']
    elif APPLIANCES == 'T':
        HOUSES = config_module.T
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','television']
    elif APPLIANCES == 'W':
        HOUSES = config_module.W
        HOUSES = HOUSES['valid']['house']
        CHANNELS = ['main','washing machine']

if __name__ == '__main__':
    main()
