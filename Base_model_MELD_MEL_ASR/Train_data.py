# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 22:58:02 2019

@author: shixiaohan
"""

import pickle

# reload a file to a variable
with open('../train_data_map_for-Mulit-speaker.pickle', 'rb') as file:
    train_data= pickle.load(file)
with open('../test_data_map_for-Mulit-speaker.pickle', 'rb') as file:
    dev_data = pickle.load(file)
with open('../dev_data_map_for-Mulit-speaker.pickle', 'rb') as file:
    test_data = pickle.load(file)

print(len(train_data[0][0]))
print(len(dev_data))
print(len(test_data))


'''
总的训练数据  9989
总的测试数据  2610
print(len(train_data))
print(len(test_data))
'''

'''
可用的训练数据  8638
可用的测试数据  2147
'''

def emo_change(x):
    if x == 'neutral':
        x = 0
    if x == 'joy':
        x = 1
    if x == 'anger':
        x = 2
    if x == 'sadness':
        x = 3
    return x

def Train_data(train_map):
    train_data_ALL = []
    for i in range(len(train_map)):
        train_data_ALL_1 = []
        for j in range(len(train_map[i])):
            train_data_ALL_1.append(train_map[i][j])
        train_data_ALL.append(train_data_ALL_1)

    label_list= [0,1,2,3]
    num = 0
    traindata_1 = []

    label_0 = 0
    label_1 = 0
    label_2 = 0
    label_3 = 0
    for i in range(len(train_data_ALL)):
        for j in range(len(train_data_ALL[i])):
            a = {}
            if (train_data_ALL[i][j]['Emotion']in label_list):
                if(train_data_ALL[i][j]['Emotion'] == 0):
                    label_0 = label_0 + 1
                if(train_data_ALL[i][j]['Emotion'] == 1):
                    label_1 = label_1 + 1
                if(train_data_ALL[i][j]['Emotion'] == 2):
                    label_2 = label_2 + 1
                if(train_data_ALL[i][j]['Emotion'] == 3):
                    label_3 = label_3 + 1
                a['trad_data_1'] = train_data_ALL[i][j]['Mel_wav_data']
                a['trad_data_2'] = train_data_ALL[i][j]['Her_wav_data']
                a['label_emotion'] = int(train_data_ALL[i][j]['Emotion'])
                a['id'] = train_data_ALL[i][j]['IDs']
                traindata_1.append(a)
                num = num + 1
    print('可用的数据为 ', num)
    print(label_0)
    print(label_1)
    print(label_2)
    print(label_3)
    return traindata_1

ALL_train_data = train_data + dev_data
train_data_fin = Train_data(ALL_train_data)
test_data_fin = Train_data(test_data)

Train_data = []
Train_data.append(train_data_fin)
Train_data.append(test_data_fin)
file = open('Train_data_Multi_Speaker.pickle', 'wb')
pickle.dump(Train_data, file)