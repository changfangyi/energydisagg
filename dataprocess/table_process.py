#!/usr/bin/env python
from __future__ import print_function, division
import os
import pymysql
import pandas as pd
import nilmtk
import numpy as np
from sys import stdout
from keras.models import load_model
from dataprocess import House_Appliance_info
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

"""
Inculde all functions related to processing table
"""
"""
appliance_in_code:
    main : 0
    television : 2
    fridge : 3
    air conditioner : 4
    bottle warmer : 5
    washinh machine : 6
"""

def activation_extract(df, appliance_in_code, min_on_duration_in_minutes=30, on_power_threshold=10):
    """
        Extract "activations" from table. 
        Any activation lasting less minutes than min_on_duration_in_minutes will be ignored.

        Input
        -----
        df : table
        appliance_in_code: 0, 2, 3, 4, 5, 6

    """
    main = df[df['channelid']==0]
    whether_sensor_for_appliance = _whether_sensor_for_appliance(df)
    target = df[df['channelid']==appliance_in_code]
    sensor_stat = whether_sensor_for_appliance[whether_sensor_for_appliance[str(appliance_in_code)]!=0]['buildingid']
    for house in np.unique(sensor_stat.index):
        # appliance
        df_target = target.groupby(['buildingid']).get_group(house)
        df_target.index = pd.to_datetime(df_target['reporttime'])
        df_target = pd.DataFrame(df_target.w.resample('1min', how='sum'))
        activations = nilmtk.electric.get_activations
        activations = activations(df_target.w, min_on_duration=60*min_on_duration_in_minutes, on_power_threshold=on_power_threshold)
        activations_number = len(activations)
        print ('house ', int(house), ' has ', activations_number, 'activations',end='\n')
        #activations_pooling = pd.DataFrame({'w':[]})
        if activations_number!=0:
            activations_pooling = pd.DataFrame({'w':[]})
            output_dir = os.path.join('data','III_' + str(appliance_in_code), 'house_'+str(int(house)))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print (output_dir, 'made')

            for i in range(activations_number):
                activation_to_append = pd.DataFrame(pd.DataFrame(activations[i]).w)
                activations_pooling = activations_pooling.append(activation_to_append)
            
            activations_pooling.to_csv(os.path.join(output_dir,'channel_2.dat'), header=False)  
            # main
            df_main = main.groupby(['buildingid']).get_group(house)
            df_main.index = pd.to_datetime(df_main['reporttime'])
            df_main = pd.DataFrame(df_main.w.resample('1min', how='sum'))
            df_main = pd.DataFrame(pd.merge(activations_pooling, df_main, right_index=True, left_index=True, how='inner')['w_y'])
            df_main.to_csv(os.path.join(output_dir,'channel_1.dat'), header=False)
    print ('Done !!!')

def append_with_previous_data(previous_data, new_data):
    append_data = previous_data.append(new_data)
    append_data = append_data.drop_duplicates()
    return append_data
    
def load_building_appliance(df, appliance_in_code, buildingid):
    main = df[df['channelid']==0]
    main = main.groupby(['buildingid']).get_group(buildingid)
    main.index = pd.to_datetime(main['reporttime'])
    main = pd.DataFrame(main.w.resample('1min', how='sum'))

    target = df[df['channelid']==appliance_in_code]
    target = target.groupby(['buildingid']).get_group(buildingid)
    target.index = pd.to_datetime(target['reporttime'])
    target = pd.DataFrame(target.w.resample('1min', how='sum'))
    
    merge = pd.merge(target, main, right_index=True, left_index=True, how='inner')
    main = pd.DataFrame(merge['w_y'])
    target = pd.DataFrame(merge['w_x'])

    if len(target)==0:
        print("Invalid Search !!!")
    else:
        return main, target

def _load_csv(load_path):
    df = pd.read_csv(load_path)
    df = df.drop(['Unnamed: 0'], axis=1)
    return df

def _from_SQL(save_path):
    HOST = '223.27.48.230'
    USER = 'tkfc'
    PWD = '1qaz@WSX'
    DBNAME = 'iii_bees_all'
    sql_query = ("SELECT * FROM raw_training_data WHERE reporttime BETWEEN '2017-12-01 00:00:00' AND '2018-08-30 00:00:00' ")
    stdout.flush()
    connection = pymysql.connect(host=HOST,
                             user=USER,
                             password=PWD,
                             port=3306,
                             db=DBNAME)
    stmt = connection.cursor()
    print ('Successfully assess to iii_bees_all database !!!')
    print ('Start to query from the raw_traing_data table !!!')
    stmt.execute(sql_query)
    table = pd.read_sql(sql_query, connection) 
    print ('The summary for the raw_traing_data table shown:')
    print ( table.describe() )
    table.to_csv(save_path)
    print ('The raw_traing_data table saved')

def _find_all_houses(df):
    return np.unique(df['buildingid'])

def _find_all_channels(df_individual_house):
    return np.unique(df_individual_house['channelid']) 

def _whether_sensor_for_appliance(df):
    summary_table = pd.DataFrame({'buildingid':[], '0':[], '1':[], '2':[], '3':[], '4':[], '5':[], '6':[]})
    
    for house in _find_all_houses(df):
        print ('Processing ', house, end='\n')
        df_individual = df.groupby(['buildingid']).get_group(house)
        house_to_append = pd.DataFrame({'buildingid':[house], '0':[0], '1':[0], '2':[0], '3':[0], '4':[0], '5':[0], '6':[0]}) 
        find_channels = _find_all_channels(df_individual) 
    
        for channel in [0,1,2,3,4,5,6]:
            if channel in find_channels:
                house_to_append[str(channel)] = 1
            
        summary_table = summary_table.append(house_to_append)  
    
    summary_table.index = summary_table['buildingid']
    return summary_table

def group_III(table, SAVEFILE, METADATA):
    """ Preparation for converting to hf format

        Input:
        -----
        table : table
        SAVEFILE : where saving these resulting files
        METADATA : where METADATA folder is
    """
    House_Appliance = House_Appliance_info.House_Appliance_info()
    for item, building in enumerate( sorted(table.groupby( ['buildingid'] ).groups)):
        orignal_name = building
        house_id = 'house_' + str(item+1)
        House_Appliance.add_house(orignal_name, house_id)
        building_data = table.groupby(['buildingid']).get_group(building)
        savefolder = SAVEFILE + '/house_' + str(item+1) + '/'
        if not os.path.exists(savefolder):
            print ("Creating", savefolder, "folder")
            os.makedirs(savefolder)        
        for i, channel in  enumerate(_find_all_channels(building_data)):
            if i==0 and channel!=0:
                print('Main Data Missing !!')
            channel_data = building_data[building_data['channelid'] == channel ]
            channel_id = i+1 
            savefile =  savefolder + '/' + 'channel_' + str(channel_id) + '.dat'
            House_Appliance.add_appliance( house_id, channel, str(channel_id) )
            channel_data = channel_data[[ 'reporttime', 'w' ]]
            channel_data.to_csv( savefile, index=False, header=False)
        print( "Creating building", int(item)+1, ".yaml ...")
        house_id_number = int(item+1)
        print(house_id_number)
        House_Appliance.YAML_Creat(house_id_number, METADATA)    
    print ("Creating dataset.yaml and meter_devices ...")
    House_Appliance.dataset_yaml(METADATA, 'Taipei')
    House_Appliance.meter_devices(METADATA)
    print ("Creating readme.txt ...")
    House_Appliance.readme_txt(SAVEFILE)
    print ("Done !" ) 
    
def _drop_nan(table, save_path_folder, resample='1min'):
    for building in sorted(table.groupby(['buildingid']).groups):
        print("Processing ", building)
        building_data = table.groupby(['buildingid']).get_group(building)
        for item, channel in enumerate(_find_all_channels(building_data)):
            if item==0: # initialization
                result = building_data[building_data['channelid'] == channel]
                result = reindex_by_reporttime_extract_w(result, resample=resample)
                missing_rate = result['w'].isnull().sum()/len(result['w'])
                print(rename(channel), ' missing rate : {:.4f}'.format( missing_rate))
                result.columns = [str(rename(channel))]
            else:  
                channel_data = building_data[building_data['channelid'] == channel]
                channel_data = reindex_by_reporttime_extract_w(channel_data , resample=resample)
                missing_rate = channel_data['w'].isnull().sum()/len(channel_data['w'])
                print(rename(channel), ' missing rate : {:.4f}'.format(missing_rate))
                channel_data.columns = [str(rename(channel))]
                result = pd.merge(channel_data, result, right_index=True, left_index=True, how='inner')
        result = result.dropna(axis='index')
        result.to_csv(os.path.join(save_path_folder, str(building)+'_nan_dropped_merged_indivi.csv'))

        for item, channel in enumerate(_find_all_channels(building_data)):
            if item==0 and building==1: # initialization
                result1 = pd.DataFrame({'w':result[str(rename(channel))].values,'reporttime':result[str(rename(channel))].index})
                result1['buildingid'] = building
                result1['channelid'] = channel
            else:  
                result2 = pd.DataFrame({'w':result[str(rename(channel))].values,'reporttime':result[str(rename(channel))].index})
                result2['buildingid'] = building
                result2['channelid'] = channel
                result1 = result1.append(result2)
    result1.to_csv(os.path.join(save_path_folder, 'nan_dropped_append_all.csv'), index=False)
        
def _validation(main, target, model, start='1/1/2018', period= 1, seq_length = 60):
    f1_all = 0
    recall_all = 0
    precision_all = 0
    acc_all = 0
    mse_all = 0
    mae_all = 0
    energy = 0
    count = 0
    for item in pd.date_range(start, periods=period, freq='1d'):
        if len(main[str(item.date())])==1440:
            #print (item.date())
            L = len(main[str(item.date())])//seq_length*seq_length
            df_main = main[str( item.date() )]
            df_target = target[str( item.date() )]
            L = len(df_main)//seq_length*seq_length

            df_main = df_main['w_y'][:L].reshape(L//seq_length,seq_length,1)
            df_target = df_target['w_x'][:L]
            df_target = df_target.fillna(0)
            prediction = model.predict_on_batch((df_main)/1000)
            prediction = prediction.reshape(L)*400
            prediction[np.isnan(prediction)] = 0

            #for i in range(int(L//seq_length)):
            #    try:
            #        start = i
            #        end = i+seq_length-1
            #        metrics_hour = Metrics(state_boundaries=[15], clip_to_zero=True)
            #        score_hour = metrics_hour.compute_metrics(prediction[start:end].flatten(), df_target[start:end].values.flatten())
            #        print ('relative_error_in_total_energy average:', score_hour['regression']['relative_error_in_total_energy']) 
            #        print ('mean_absolute_error:', score_hour['regression']['mean_absolute_error']) 
            #    except:
            #        pass
            
            metrics = Metrics(state_boundaries=[15], clip_to_zero=True)
            score = metrics.compute_metrics(prediction.flatten(), df_target.values.flatten())                       
            energy += score['regression']['relative_error_in_total_energy']
            mse_all += score['regression']['mean_squared_error']
            mae_all += score['regression']['mean_absolute_error']
            f1_all += score['classification_2_state']['f1_score']
            recall_all += score['classification_2_state']['recall_score']
            precision_all += score['classification_2_state']['precision_score']
            acc_all += score['classification_2_state']['accuracy_score']
            count+=1
    print ('relative_error_in_total_energy average: {:.4f}'.format(energy/count) )       
    print ('mse average: {:.4f}'.format(mse_all/count))
    print ('mae average: {:.4f}'.format(mae_all/count))
    print ('acc average: {:.4f}'.format(acc_all/count))
    print ('f1 average: {:.4f}'.format(f1_all/count))
    print ('recall average: {:.4f}'.format(recall_all/count))
    print ('precision average: {:.4f}'.format(precision_all/count))    
    df_main = pd.DataFrame({'w':df_main.reshape(len(df_target))})
    df_main.index = df_target.index        
    return df_main, prediction, df_target

def rename(name):
    if name == 0:
        name = 'main'
    elif name == 1:
        name = 'others'
    elif name == 2:
          name = 'television'
    elif name == 3:
          name = 'fridge'
    elif name == 4:
          name = 'air conditioner'
    elif name == 5:
          name = 'bottle warmer'
    elif name == 6:
          name = 'washing machine'
    elif name == '1002':
          name = 'fridge'
    elif name == '1004':
          name = 'air conditioner'        
    return name

def reindex_by_reporttime_extract_w(table, resample='1min'):
    result = table[[ 'reporttime', 'w' ]]
    result.index = pd.to_datetime(result['reporttime'])
    result = pd.DataFrame(result.w.resample(resample, how='sum'))
    result = result[['w']]            
    return result

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


