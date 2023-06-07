import pickle
import csv
import os
import numpy as np
import yaml

import re
import wave
import numpy as np
import python_speech_features as ps
from sklearn.preprocessing import StandardScaler
import os
import glob
import pickle
import csv

with open('Text_data_No_combine.pickle', 'rb') as file:
    Pre_ALL_data = pickle.load(file)

Data_dir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/Experiment_Data/IEM_Add_noise_HJJ/SNR_-5_hjj_park/'

def Get_fea_new(data_dir_org):
    # 提取声学特征
    traindata = []
    num = 0
    for sess in os.listdir(data_dir_org):
        data_dir = data_dir_org + '/' + sess
        data = {}
        data['id'] = sess[:-4][:-10]
        data['fea_data'] = np.load(data_dir)
        num = num + 1
        traindata.append(data)
    return traindata

def combine_wav_text(Pre_ALL_data, train_ALL_data):
    for i in range(len(train_ALL_data)):
        for j in range(len(train_ALL_data[i])):
            for x in range(len(Pre_ALL_data)):
                if (train_ALL_data[i][j]['id'] == Pre_ALL_data[x]['id']):
                    train_ALL_data[i][j]['Her_wav_data'] = Pre_ALL_data[x]['fea_data']
    return train_ALL_data

#语音特征 time * 144
train_ALL_data = Get_fea_new(Data_dir)
print(len(train_ALL_data))
print('**********************************')
train_fin_data = combine_wav_text(train_ALL_data, Pre_ALL_data)


file = open('train_data_map_-5db_park.pickle', 'wb')
pickle.dump(train_fin_data, file)
file.close()
