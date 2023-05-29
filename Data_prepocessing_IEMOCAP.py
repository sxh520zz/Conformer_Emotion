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

Data_dir = '/home/shixiaohan-toda/Desktop/ASR-SER/Processed_IEMOCAP/'

'''
train_data: 9989
dev_data: 1109
test_data: 2610
'''

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
            if(len(train_ALL_data[i][j]) == 10):
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
data_6 = []
data_7 = []
data_8 = []
data_9 = []
data_10 = []

for i in range(len(train_fin_data)):
    for j in range(len(train_fin_data[i])):
        if (train_fin_data[i][j]['id'][4] == '1'):
            if (train_fin_data[i][j]['id'][-4] == 'F'):
                data_1.append(train_fin_data[i][j])
            if (train_fin_data[i][j]['id'][-4] == 'M'):
                data_2.append(train_fin_data[i][j])
        if (train_fin_data[i][j]['id'][4] == '2'):
            if (train_fin_data[i][j]['id'][-4] == 'F'):
                data_3.append(train_fin_data[i][j])
            if (train_fin_data[i][j]['id'][-4] == 'M'):
                data_4.append(train_fin_data[i][j])
        if (train_fin_data[i][j]['id'][4] == '3'):
            if (train_fin_data[i][j]['id'][-4] == 'F'):
                data_5.append(train_fin_data[i][j])
            if (train_fin_data[i][j]['id'][-4] == 'M'):
                data_6.append(train_fin_data[i][j])
        if (train_fin_data[i][j]['id'][4] == '4'):
            if (train_fin_data[i][j]['id'][-4] == 'F'):
                data_7.append(train_fin_data[i][j])
            if (train_fin_data[i][j]['id'][-4] == 'M'):
                data_8.append(train_fin_data[i][j])
        if (train_fin_data[i][j]['id'][4] == '5'):
            if (train_fin_data[i][j]['id'][-4] == 'F'):
                data_9.append(train_fin_data[i][j])
            if (train_fin_data[i][j]['id'][-4] == 'M'):
                data_10.append(train_fin_data[i][j])

data = []
data.append(data_1)
data.append(data_2)
data.append(data_3)
data.append(data_4)
data.append(data_5)
data.append(data_6)
data.append(data_7)
data.append(data_8)
data.append(data_9)
data.append(data_10)

file = open('train_data_map_IEM.pickle', 'wb')
pickle.dump(data, file)
file.close()
