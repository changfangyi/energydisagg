#!/usr/bin/env python#!/usr/bin/env python
from __future__ import print_function, division
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from time import strftime

class Sequence(object):
    """
    Attributes
    ----------
    input : np.ndarray
    target : np.ndarray
    all_appliances : pd.DataFrame
        Column names are the appliance names.
    metadata : dict
    weights : np.ndarray or None
    """
    def __init__(self, shape, target_channels_in_list):
        self.input = np.zeros(shape, dtype=np.float32)
        self.target = {}
        for target_channel in target_channels_in_list:
            self.target[str(target_channel)] = np.zeros(shape, dtype=np.float32)

# the RealSource, which will replace the original one
class RealSource(object):
    
    def __init__(self, data_to_memory, channels, seq_length, 
                        houses, houses_prob, activations_prob, num_seq_per_batch=32):
        self.data_to_memory = data_to_memory
        self.channels = channels # must be list
        self.seq_length = seq_length
        self.houses = houses
        self.houses_prob = houses_prob
        self.activations_prob = activations_prob
        self.num_seq_per_batch = num_seq_per_batch
    
    def _select_building(self, houses, houses_prob):
        """
        For Example:
        
        _select_building(train_builing, building_prob), where
        train_builing = [house_1, house_14, house_19]
        building_prob = [0.5, 0.2, 0.3]
        """
        return np.random.choice(houses, 1, p=houses_prob)
    
    def _select_activation(self, activations, activations_prob):
        """
        For Example:
        
        _select_activation(range(len(activation_prob['house_1'])), activation_prob['house_1']), where
        range(len(activation_prob['house_1'])) = [0,1,2,3,5]
        activation_prob['house_1'] = [0.1, 0.1, 0.3, 0.2, 0.3]
        """
        return np.random.choice(activations, 1, p=activations_prob)
    
    def _get_seq_and_check(self, data_to_memory, houses, houses_prob, activations_prob):
        """
        get a batch of data
        For Example:
        get_seq_and_check(collection, train_builing, building_prob, activation_prob)
        collection = data
        train_builing = [house_1, house_14, house_19]
        building_prob = [0.5, 0.2, 0.3]
        activation_prob = {'house_1':[0.1, 0.1, 0.3, 0.2, 0.3], 
                            'house_14':[0.1, 0.1, 0.3, 0.2, 0.3],
                            'house_19':[0.1, 0.1, 0.3, 0.2, 0.3]}
        
        Warning:
        ------------------------------------------------------------------------
            Currently, setting max_iter == 120, the gap within select_start and end is self.seq_length*2 points
            If the gap is self.seq_length points, it will not success. The cause needs to be figured out
            In the prototype stage, using main as target
        """
        # Check whether the length of selected activation is larger than self.seq_length
        success_for_enough_data = False
        max_iter_for_enough_data = 0

        while not success_for_enough_data:
            max_iter_for_enough_data +=1
            select_house = self._select_building(houses, houses_prob)[0]
            select_house = 'house_'+str(select_house) 
            activation_prob_for_the_select_building = activations_prob[select_house]
            select_activation = self._select_activation(range(len(activation_prob_for_the_select_building)), 
                                           activation_prob_for_the_select_building)[0]      
            get_seq_before_check = data_to_memory[select_house][str(select_activation)]
            get_seq_before_check.index = pd.to_datetime(get_seq_before_check.index) # double check that the index is datetime format
            if len(get_seq_before_check)>=self.seq_length or  max_iter_for_enough_data >= 32 :
                 success_for_enough_data = True
      
        success_for_large_length = False
        max_iter_success_for_large_length = 0
        while not success_for_large_length:
            max_iter_success_for_large_length +=1
            select_start = get_seq_before_check.sample(n=1).index[0]
            end = select_start + timedelta(seconds = 60*self.seq_length*2) 
            if len(get_seq_before_check[select_start:end])>=self.seq_length or max_iter_success_for_large_length == 32:
                success_for_large_length = True
                get_seq_after_check = get_seq_before_check[select_start:end]
        
        if max_iter_success_for_large_length ==32:
            seq = None

        else:
            del get_seq_before_check
            seq = Sequence(self.seq_length, self.channels)
            seq.input = np.array(get_seq_after_check[self.channels[0]].values[:self.seq_length])
            for target in self.channels[1:]:
                seq.target[str(target)] = np.array(get_seq_after_check[target].values[:self.seq_length])
        return seq
    
    def _get_sequence(self):
        seq = self._get_seq_and_check( data_to_memory = self.data_to_memory, 
                                        houses = self.houses, 
                                        houses_prob = self.houses_prob, 
                                        activations_prob = self.activations_prob)
        return seq
    
    def _get_batch(self):
        """
        Returns
        -------
        A Batch object or None
        """

        input_sequences = []
        target_sequences = {}
        none_happened = False
        for target in self.channels[1:]:
                target_sequences[str(target)] = []

        for i in range(self.num_seq_per_batch):
            seq = self._get_sequence()
            
            if seq is None:
                none_happened = True
            else:
                input_sequences.append(seq.input.reshape(self.seq_length,1))
                for channel in self.channels[1:]:
                    target_sequences[channel].append(seq.target[channel].reshape(self.seq_length,1))
                
        if not none_happened:
            input_sequences = np.asarray(input_sequences).reshape(self.num_seq_per_batch,self.seq_length,1)
            for channel in self.channels[1:]:
                target_sequences[channel] = np.asarray(target_sequences[channel]).reshape(self.num_seq_per_batch,self.seq_length,1)
        else:
            input_sequences = None
            target_sequences = None
            
        return input_sequences, target_sequences

class SynSource(RealSource):
    def __init__(self, data_to_memory, channels, seq_length, 
                        houses, houses_prob, activations_prob, num_seq_per_batch=32):
        self.data_to_memory = data_to_memory
        self.channels = channels # must be list
        self.seq_length = seq_length
        self.houses = houses
        self.houses_prob = houses_prob
        self.activations_prob = activations_prob
        self.num_seq_per_batch = num_seq_per_batch
        
    def _get_sequence(self):
        seq = Sequence(self.seq_length, self.channels)
   
        for channel in self.channels[1:]:
            channel_seq = super(SynSource,self)._get_sequence()
            # for targets
            if channel_seq is None:
                seq.target[channel] = np.array(np.zeros(self.seq_length))
            else:
                seq.target[channel] = np.array(channel_seq.target[channel])
            # for input
            seq.input += seq.target[channel]            
        return seq
    

          