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

PATH = os.path.join(os.path.expanduser('~') , 'Desktop', 'energydisagg')
MODEL = None
APPLIANCES = None
CHANNELS = None
HOUSES = None
NUM_STEPS = None
DATA = None
FREQ_REAL_SYN = 2

def main():
    SINGLE = False
    os.chdir(PATH)
    parse_args()
    load_config()
    if len(CHANNELS) == 1+1 :
	SINGLE = True
    model = os.path.join(PATH, 'models', MODEL + '.h5')
    model = load_model(model)
    print(model.summary())
    data_to_memory, house_prob, activation_prob = load_data(HOUSES, os.path.join(PATH, 'data', DATA))
    real_source = RealSource(data_to_memory = data_to_memory, channels = CHANNELS, seq_length=60, 
                        houses = HOUSES, houses_prob  = house_prob, activations_prob = activation_prob)
    main, targets = real_source._get_batch()
    while main is None or targets is None:
        main, targets = real_source._get_batch()
    # validate
    validate = Validation(main, targets, model, CHANNELS, SINGLE)
    validate._zeros_guess()
    validate._model_guess()
    validate._plot(os.path.join(PATH, 'fig'))


def parse_args():
    global APPLIANCES, MODEL, DATA
    parser = argparse.ArgumentParser()
     # required
    required_named_arguments = parser.add_argument_group('required named arguments')
    required_named_arguments.add_argument('-a', '--appliancess',
                                          help='FB fridge and bottle warmer',
                                          required=True)
    required_named_arguments.add_argument('-d', '--data',
                                          help='data name',
                                          required=True)
    required_named_arguments.add_argument('-m', '--model',
                                          help='model name',
                                          required=True)
     # start parsing
    args = parser.parse_args()
    APPLIANCES = args.appliancess
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
    elif APPLIANCES == 'AB':
        HOUSES = config_module.AB
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','air conditioner','bottle warmer']
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
    elif APPLIANCES == 'KD':
        HOUSES = config_module.KD
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','kettle','dish washer']
    elif APPLIANCES == 'MD':
        HOUSES = config_module.MD
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','microwave','dish washer']
    elif APPLIANCES == 'MW':
        HOUSES = config_module.MW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','microwave','washing machine']
    elif APPLIANCES == 'KD':
        HOUSES = config_module.KD
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','kettle','dish washer']

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
    elif APPLIANCES == 'FTB':
        HOUSES = config_module.FTB
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','television', 'bottle warmer']
    elif APPLIANCES == 'FTBWA':
        HOUSES = config_module.FTBWA
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','fridge','television', 'bottle warmer','washing machine','air conditioner']
    elif APPLIANCES == 'KW':
        HOUSES = config_module.KW
        HOUSES = HOUSES['train']['house']
        CHANNELS = ['main','kettle','washing machine']

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
