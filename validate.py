#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
from lib import dirs
import importlib
from dataprocess.source import RealSource
from dataprocess.validation import Validation
from keras.models import load_model
from train import load_data

#PATH = '/Users/kang/Desktop/energydisagg' # multi_group
PATH = '/home/nilm/Desktop/energydisagg' # multi_group
MODEL = None
APPLIANCES = None
CHANNELS = None
HOUSES = None
NUM_STEPS = None
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
    validate = Validation(main, targets, model, CHANNELS)
    validate._zeros_guess
    validate._model_guess
    validate._plot(os.path.join(PATH, 'fig'))


def parse_args():
    global APPLIANCES, MODEL
    parser = argparse.ArgumentParser()
     # required
    required_named_arguments = parser.add_argument_group('required named arguments')
    required_named_arguments.add_argument('-a', '--appliancess',
                                          help='FB fridge and bottle warmer',
                                          required=True)
    required_named_arguments.add_argument('-m', '--model',
                                          help='model name',
                                          required=True)
     # start parsing
    args = parser.parse_args()
    APPLIANCES = args.appliancess
    MODEL = args.model

def load_config():
    global CHANNELS, HOUSES 
    config_module = importlib.import_module(dirs.CONFIG_DIR + '.' + 'config', __name__)
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
        HOUSES = config_module.FW
        HOUSES = HOUSES['house']
        CHANNELS = ['main','fridge','washing machine']

if __name__ == '__main__':
    main()