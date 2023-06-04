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

Data_dir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/Experiment_Data/IEM_Add_noise_HJJ/5db/'


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

    label_list= [1,2,3,4,5]
    ALL_data = []
    for i in range(len(train_ALL_data)):
        ALL_data_1 = []
        for j in range(len(train_ALL_data[i])):
            print(len(train_ALL_data[i][j]))
            if(len(train_ALL_data[i][j]) == 9):
                if(train_ALL_data[i][j]['label'] in label_list):
                    if(train_ALL_data[i][j]['label'] == 5):
                        train_ALL_data[i][j]['label'] = 2
                    train_ALL_data[i][j]['label'] = train_ALL_data[i][j]['label'] - 1
                    ALL_data_1.append(train_ALL_data[i][j])
        ALL_data.append(ALL_data_1)
    return ALL_data


#语音特征 time * 144
train_ALL_data = Get_fea_new(Data_dir)
print(len(train_ALL_data))
print('**********************************')


train_fin_data = combine_wav_text(train_ALL_data, Pre_ALL_data)

print(len(train_fin_data))

speaker = ['1','2','3','4','5','6','7','8','9','10']
#按照说话人分折

data_1 = []
data_2 = []
data_3 = []
data_4 = []
data_5 = []


for i in range(len(train_fin_data)):
    for j in range(len(train_fin_data[i])):
        if (train_fin_data[i][j]['id'][4] == '1'):
            data_1.append(train_fin_data[i][j])
        if (train_fin_data[i][j]['id'][4] == '2'):
            data_2.append(train_fin_data[i][j])
        if (train_fin_data[i][j]['id'][4] == '3'):
            data_3.append(train_fin_data[i][j])
        if (train_fin_data[i][j]['id'][4] == '4'):
            data_4.append(train_fin_data[i][j])
        if (train_fin_data[i][j]['id'][4] == '5'):
            data_5.append(train_fin_data[i][j])

data = []
data.append(data_1)
data.append(data_2)
data.append(data_3)
data.append(data_4)
data.append(data_5)

file = open('train_data_map_IEM.pickle', 'wb')
pickle.dump(data, file)
file.close()
