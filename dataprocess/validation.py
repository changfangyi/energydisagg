#!/usr/bin/env python
from __future__ import print_function, division

import os
import numpy as np
import argparse
import importlib
import pandas as pd
import matplotlib.pyplot as plt
from metrics import Metrics
from keras.models import load_model


class Validation(object):
    def __init__(self, main, targets, model, channels):
        self.main = main
        self.targets = targets
        self.model = model
        self.channels = channels

    def _inference(self):
        return self.model.predict_on_batch(self.main)
        
    def _zeros_guess(self):
        print('Zeros Guess:')
        zeros_guess_score = {}
        for channel, values in self.targets.iteritems():
            print(channel, ':')
            target_length = len(values.flatten())
            metrics = Metrics(state_boundaries=[15], clip_to_zero=True)
            scores = metrics.compute_metrics(np.zeros(target_length), self.targets[channel].flatten())
            zeros_guess_score[channel]=scores
            for __, score in scores.iteritems():
                for metrics_name, value in score.iteritems():
                    print(metrics_name, ': {:.2f}'.format(value))
                print('\n')
        return zeros_guess_score 
    
    def _model_guess(self):
        print('Model Guess :')
        model_guess_score = {}
        for item, channel in self.targets.iteritems():
            print(channel, ':')
            metrics = Metrics(state_boundaries=[15], clip_to_zero=True)
            scores = metrics.compute_metrics(self._inference()[item].flatten(), self.targets[channel].flatten())
            model_guess_score[channel]=scores
            for __, score in scores.iteritems():
                for metrics_name, value in score.iteritems():
                    print(metrics_name, ': {:.2f}'.format(value))
                print('\n')
        return model_guess_score

    def _plot(self, savefolder, NUM_SEQ_PER_BATCH=16):
        for item, channel in enumerate(self.channels[1:]):
            print(channel)
            for sample_no in range(NUM_SEQ_PER_BATCH):
                p1 = plt.subplot(131)
                p1.set_title('Input #{}'.format(sample_no + 1))
                p2 = plt.subplot(132, sharey=p1)
                p2.set_title('Target #{}'.format(sample_no + 1))
                p3 = plt.subplot(133, sharey=p1)
                p3.set_title('Prediction #{}'.format(sample_no + 1))
                p1.plot(self.main[sample_no].flatten())
                p2.plot(self.targets[channel][sample_no].flatten())
                p3.plot(self._inference()[item][sample_no].flatten())
                plt.savefig(os.path.join(savefolder, channel, 'Step_{}.png'.format(sample_no + 1)))
                plt.clf()
        
        


    

    
