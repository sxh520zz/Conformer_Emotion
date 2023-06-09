import pickle
import csv
import os
import numpy as np
import yaml

import re
import wave
import numpy as np
import python_speech_features as ps
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import os
import torch
import pickle
import csv


with open('MELD_features_raw.pkl', 'rb') as file:
    Pre_ALL_data = pickle.load(file)

Data_dir = '/home/shixiaohan-toda/Desktop/Experiment/MELD_Feature/'
rootdir_1 = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/MELD.Raw/'
rootdir_2 = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/MELD/'

from transformers import Wav2Vec2Model
from transformers import Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")    # 用于提取通用特征，768维
model.eval()


'''
train_data: 9989
dev_data: 1109
test_data: 2610
'''

def Get_fea_text(dir, name):
    # 提取ASR的识别结果
    data_dir_org = dir + name + '_wav.txt'
    traindata = []
    with open(data_dir_org,'r') as f:
        line = f.readlines()
        for line_1 in line:
            data = {}
            line_split = line_1.split("\t")
            data['id'] = line_split[0][:-4]
            data['tras'] = line_split[-1][:-2]
            traindata.append(data)
    return traindata


def Get_fea_new(dir, name):
    # 提取声学特征
    data_dir_org = dir + name + '_wav'
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

def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate

def Get_fea_Spec(dir, name):
    filter_num = 40
    train_mel_data = []
    data_dir_org = dir + name + '_wav'
    train_num = 0
    for sess in os.listdir(data_dir_org):
        data_dir = data_dir_org + '/' + sess
        wavname = data_dir.split("/")[-1][:-4]
        data, time, rate = read_file(data_dir)
        mel_spec = ps.logfbank(data, rate, nfilt=filter_num)
        mel_data = []
        one_mel_data = {}
        part = mel_spec
        delta1 = ps.delta(mel_spec, 2)
        delta2 = ps.delta(delta1, 2)
        input_data_1 = np.concatenate((part, delta1), axis=1)
        input_data = np.concatenate((input_data_1, delta2), axis=1)
        mel_data.append(input_data)
        one_mel_data['id'] = wavname
        mel_data = np.array(mel_data)
        one_mel_data['fea_data'] = mel_data
        train_mel_data.append(one_mel_data)
        train_num = train_num + 1
    return train_mel_data

def Get_fea_Wav2vec(dir, name):
    train_mel_data = []
    data_dir_org = dir + name + '_wav'
    train_num = 0
    for sess in os.listdir(data_dir_org):
        mel_data = []
        one_mel_data = {}
        data_dir = data_dir_org + '/' + sess
        wavname = data_dir.split("/")[-1][:-4]
        audio_input, sample_rate = sf.read(data_dir)
        input_values = processor(audio_input, sampling_rate=sample_rate,
                                 return_tensors="pt").input_values
        with torch.no_grad():
            # model inference
            hidden_states = model(input_values, output_hidden_states=True).hidden_states # tuple of (B, T, D)
            feature = torch.stack(hidden_states)[-1].sum(dim=0)  # sum, (B, T, D)
            feature = feature[0].detach().squeeze().cpu().numpy() # (T, D)
            mel_data.append(feature)
            one_mel_data['id'] = wavname
            mel_data = np.array(mel_data)
            one_mel_data['wav2vec_data'] = mel_data
            '''
            wav2vec2 = model(input_values)['last_hidden_state']
            wav2vec2 = wav2vec2.mean(1)
            mel_data.append(wav2vec2.detach().numpy())
            one_mel_data['id'] = wavname
            mel_data = np.array(mel_data)
            one_mel_data['wav2vec_data'] = mel_data
            '''
            train_mel_data.append(one_mel_data)
            train_num = train_num + 1
            print(train_num)
    return train_mel_data

def Get_fea(dir, name):
    # 提取声学特征
    data_dir_org = dir + name + '_data'
    traindata = []
    num = 0
    for sess in os.listdir(data_dir_org):
        data_dir = data_dir_org + '/' + sess
        data_1 = []
        data = {}
        file = open(data_dir, 'r')
        file_content = csv.reader(file)
        for row in file_content:
            if (row[0] != 'file'):
                x = []
                for i in range(3, len(row)):
                    row[i] = float(row[i])
                    b = np.isinf(row[i])
                    # print(b)
                    if b:
                        print(row[i])
                    x.append(row[i])
                row = np.array(x)
                data_1.append(row)
        data['id'] = sess[:-4]
        data_1_1 = np.array(data_1)
        data['fea_data'] = data_1_1
        num = num + 1
        traindata.append(data)
    return traindata

def Divide_train_test_data_from_train(ALL_data,ALL_data_tar):
    num = 0
    train_data = []
    for itm in ALL_data_tar:
        data_1 = []
        if(itm <= 1038):
            for j in range(len(ALL_data[2][itm])):
                data = {}
                data['IDs'] = 'dia' + str(itm) + '_utt' + str(ALL_data[0][itm][j])
                data['Speaker'] = ALL_data[1][itm][j].index(1)
                data['Emotion'] = ALL_data[2][itm][j]
                data['Text'] = ALL_data[3][itm][j]
                data['Audio'] = ALL_data[4][itm][j]
                data['Utterance'] = ALL_data[5][itm][j]
                data['Sentiment'] = ALL_data[8][itm][j]
                data_1.append(data)
                num = num + 1
            train_data.append(data_1)
    print(num)
    return train_data
def Divide_train_test_data_from_dev(ALL_data,ALL_data_tar):
    train_data = []
    num = 0
    for itm in ALL_data_tar:
        data_1 = []
        if(itm >= 1039):
            for j in range(len(ALL_data[2][itm])):
                data = {}
                id_ix = itm - 1039
                data['IDs'] = 'dia' + str(id_ix) + '_utt' + str(ALL_data[0][itm][j])
                data['Speaker'] = ALL_data[1][itm][j].index(1)
                data['Emotion'] = ALL_data[2][itm][j]
                data['Text'] = ALL_data[3][itm][j]
                data['Audio'] = ALL_data[4][itm][j]
                data['Utterance'] = ALL_data[5][itm][j]
                data['Sentiment'] = ALL_data[8][itm][j]
                data_1.append(data)
                num = num + 1
            train_data.append(data_1)
    print(num)
    return train_data
def Divide_train_test_data_from_test(ALL_data,ALL_data_tar):
    train_data = []
    num = 0
    for itm in ALL_data_tar:
        data_1 = []
        if(itm >= 1153):
            for j in range(len(ALL_data[2][itm])):
                data = {}
                id_ix = itm - 1153
                data['IDs'] = 'dia' + str(id_ix) + '_utt' + str(ALL_data[0][itm][j])
                data['Speaker'] = ALL_data[1][itm][j].index(1)
                data['Emotion'] = ALL_data[2][itm][j]
                data['Text'] = ALL_data[3][itm][j]
                data['Audio'] = ALL_data[4][itm][j]
                data['Utterance'] = ALL_data[5][itm][j]
                data['Sentiment'] = ALL_data[8][itm][j]
                data_1.append(data)
                num = num + 1
            train_data.append(data_1)
    print(num)
    return train_data

def Get_label(name):
    # 获取当前脚本所在文件夹路径
    curPath = os.path.dirname(os.path.realpath(__file__))
    # 获取yaml文件路径
    yamlPath = os.path.join(curPath, "datasets.yaml")
    # open方法打开直接读出来
    with open(yamlPath, 'r', encoding='utf-8') as f:
        config = f.read()
    d = yaml.load(config, Loader=yaml.FullLoader)  # 用load方法转字典
    file_content = d[name]
    max_dia_num = 0
    train_label = []
    for row in file_content:
        s_data = {}
        s_data['Id'] = 'dia' + str(file_content[row]['Dialogue_ID']) + '_utt' + str(file_content[row]['Utterance_ID'])
        s_data['id'] = str(file_content[row]['Dialogue_ID']) + '_' + str(file_content[row]['Utterance_ID'])
        s_data['Utterance'] = file_content[row]['Utterance']
        s_data['Speaker'] = file_content[row]['Speaker']
        s_data['Emotion'] = emo_change(file_content[row]['Emotion'])
        s_data['Sentiment'] = file_content[row]['Sentiment']
        s_data['Dialogue_ID'] = file_content[row]['Dialogue_ID']
        s_data['Utterance_ID'] = file_content[row]['Utterance_ID']
        s_data['Group'] = str(file_content[row]['Season']) + '_' + str(file_content[row]['Episode'])
        if (max_dia_num < int(s_data['Dialogue_ID'])):
            max_dia_num = int(s_data['Dialogue_ID'])
        train_label.append(s_data)
    return train_label, max_dia_num

def combine_wav_text(Pre_data,wav_pre_data,wav_pre_data_1, wav_pre_data_11,wav_pre_data_2,label_data):
    for i in range(len(Pre_data)):
        for j in range(len(Pre_data[i])):
            for x in range(len(wav_pre_data)):
                if (Pre_data[i][j]['IDs'] == wav_pre_data[x]['id']):
                    Pre_data[i][j]['Her_wav_data'] = wav_pre_data[x]['fea_data']
    for i in range(len(Pre_data)):
        for j in range(len(Pre_data[i])):
            for x in range(len(wav_pre_data_1)):
                if (Pre_data[i][j]['IDs'] == wav_pre_data_1[x]['id']):
                    Pre_data[i][j]['Mel_wav_data'] = wav_pre_data_1[x]['fea_data']
    for i in range(len(Pre_data)):
        for j in range(len(Pre_data[i])):
            for x in range(len(wav_pre_data_11)):
                if (Pre_data[i][j]['IDs'] == wav_pre_data_11[x]['id']):
                    Pre_data[i][j]['Wav2vec_wav_data'] = wav_pre_data_11[x]['wav2vec_data']

    for i in range(len(Pre_data)):
        for j in range(len(Pre_data[i])):
            for x in range(len(wav_pre_data_2)):
                if (Pre_data[i][j]['IDs'] == wav_pre_data_2[x]['id']):
                    Pre_data[i][j]['Text_data'] = wav_pre_data_2[x]['tras']
    for i in range(len(Pre_data)):
        for j in range(len(Pre_data[i])):
            for x in range(len(label_data)):
                if (Pre_data[i][j]['IDs'] == label_data[x]['Id']):
                    Pre_data[i][j]['Group'] = label_data[x]['Group']
                    Pre_data[i][j]['Dialogue_ID'] = label_data[x]['Dialogue_ID']
                    Pre_data[i][j]['Utterance_ID'] = label_data[x]['Utterance_ID']
    ALL_data = []
    for i in range(len(Pre_data)):
        for j in range(len(Pre_data[i])):
            if(len(Pre_data[i][j]) == 14):
                ALL_data.append(Pre_data[i][j])
    return ALL_data

def remove_nested_list(listt):
    while [] in listt:  # 判断是否有空值在列表中
        listt.remove([])  # 如果有就直接通过remove删除
    return listt

def emo_change(x):
    if x == 'neutral':
        x = 0
    if x == 'joy':
        x = 1
    if x == 'anger':
        x = 2
    if x == 'sadness':
        x = 3
    if x == 'surprise':
        x = 4
    if x == 'fear':
        x = 5
    if x == 'sadness':
        x = 3

    return x

def Re_Define_Group(data):
    group = []
    ALL_Data = [[] for _ in range(500)]
    for i in range(len(data)):
        if(data[i]['Group'] not in group):
            group.append(data[i]['Group'])
    for i in range(len(data)):
        ALL_Data[group.index(data[i]['Group'])].append(data[i])
    fin_data = remove_nested_list(ALL_Data)
    return fin_data

def Class_data(all_data, max_dia_num):
    #按照对话数据分组
    Train_data = [[] for x in range(max_dia_num +1)]
    for data_ind in all_data:
        Train_data[int(data_ind['Dialogue_ID'])].append(data_ind)
    train_data = []
    for i in range(len(Train_data)):
        dia_data = {}
        name_list = []
        for j in range(len(Train_data[i])):
            if (Train_data[i][j]['Speaker'] not in name_list):
                name_list.append(Train_data[i][j]['Speaker'])
        dia_data['Speaker_num'] = len(name_list)
        dia_data['Dia_length'] = len(Train_data[i])
        dia_data['Dia_data'] = Train_data[i]
        train_data.append(dia_data)

    CAN_USE = []
    num = 0
    for i in range(len(train_data)):
        #if (train_data[i]['Dia_length'] >= 7 and train_data[i]['Speaker_num'] >= 2):
        CAN_USE.append(train_data[i]['Dia_data'])
        num = num + train_data[i]['Dia_length']
    print("可用的数据 ",num)
    return CAN_USE




#语音MEL特征 time*120
train_pre_data_mel = Get_fea_Spec(rootdir_1, 'train')
dev_pre_data_mel = Get_fea_Spec(rootdir_1, 'dev')
test_pre_data_mel = Get_fea_Spec(rootdir_1, 'test')
print(len(train_pre_data_mel)+len(dev_pre_data_mel)+len(test_pre_data_mel))
print('**********************************')


#语音ASR特征 time * 144
train_pre_data = Get_fea_new(rootdir_2, 'train')
dev_pre_data = Get_fea_new(rootdir_2, 'dev')
test_pre_data = Get_fea_new(rootdir_2, 'test')
print(len(train_pre_data)+len(dev_pre_data)+len(test_pre_data))
print('**********************************')

#语音WAV2VEC特征 time * 768
train_pre_data_wav2vec = Get_fea_Wav2vec(rootdir_1, 'train')
dev_pre_data_wav2vec = Get_fea_Wav2vec(rootdir_1, 'dev')
test_pre_data_wav2vec = Get_fea_Wav2vec(rootdir_1, 'test')
print(len(train_pre_data_wav2vec)+len(dev_pre_data_wav2vec)+len(test_pre_data_wav2vec))
print('**********************************')

Pre_train_data = Divide_train_test_data_from_train(Pre_ALL_data,Pre_ALL_data[6])
Pre_dev_data = Divide_train_test_data_from_dev(Pre_ALL_data,Pre_ALL_data[6])
Pre_test_data = Divide_train_test_data_from_test(Pre_ALL_data,Pre_ALL_data[7])


#提取文本内容+句级别标签
train_label, max_dia_num_train = Get_label('train')
dev_label, max_dia_num_dev = Get_label('dev')
test_label, max_dia_num_test = Get_label('test')
print(len(train_label)+len(dev_label)+len(test_label))
print('**********************************')

train_pre_data_text = Get_fea_text(rootdir_2, 'train')
dev_pre_data_text = Get_fea_text(rootdir_2, 'dev')
test_pre_data_text = Get_fea_text(rootdir_2, 'test')
print(len(train_pre_data_text)+len(dev_pre_data_text)+len(test_pre_data_text))
print('**********************************')

train_data = combine_wav_text(Pre_train_data, train_pre_data, train_pre_data_mel, train_pre_data_wav2vec, train_pre_data_text, train_label)
dev_data = combine_wav_text(Pre_dev_data, dev_pre_data, dev_pre_data_mel, dev_pre_data_wav2vec, dev_pre_data_text, dev_label,)
test_data = combine_wav_text(Pre_test_data, test_pre_data, test_pre_data_mel, test_pre_data_wav2vec, test_pre_data_text, test_label)

print(len(train_data))
print(len(dev_data))
print(len(test_data))

train_data_org = Class_data(train_data, max_dia_num_train)
dev_data_org = Class_data(dev_data, max_dia_num_dev)
test_data_org = Class_data(test_data, max_dia_num_test)

file = open('train_data_map_for-Mulit-speaker.pickle', 'wb')
pickle.dump(Re_Define_Group(train_data), file)
file.close()

file = open('dev_data_map_for-Mulit-speaker.pickle', 'wb')
pickle.dump(Re_Define_Group(dev_data), file)
file.close()

file = open('test_data_map_for-Mulit-speaker.pickle', 'wb')
pickle.dump(Re_Define_Group(test_data), file)
file.close()



'''
def is_UTF_8(str):
    remain = 0         #剩余byte数
    for x in range(len(str)):
        if remain == 0:
            if (ord(str[x]) & 0x80) == 0x00:
                remain = 0
            elif (ord(str[x]) & 0xE0) == 0xC0:
                remain = 1
            elif (ord(str[x]) & 0xF0) == 0xE0:
                remain = 2
            elif(ord(str[x]) & 0xF8) == 0xF0:
                remain = 3
            else:
                return False
        else:
            if not ((ord(str[x]) & 0xC0) == 0x80):
                return False
            remain = remain - 1
    if remain == 0:
        return True
    else:
        return False
'''

'''
f = open("a.txt", 'w')
for i in range(len(Pre_train_data)):
    for j in range(len(Pre_train_data[i])):
        f.write(Pre_train_data[i][j]['IDs'] + '.wav')
        f.write('\t')
        x= []
        for w in range(len(Pre_train_data[i][j]['Utterance'])):
            if(is_UTF_8(Pre_train_data[i][j]['Utterance'][w])):
                x.extend(Pre_train_data[i][j]['Utterance'][w])
            else:
                x.extend("'")
        f.write(''.join(x))
        f.write('\n')
f.close()

f = open("b.txt", 'w')
for i in range(len(Pre_dev_data)):
    for j in range(len(Pre_dev_data[i])):
        f.write(str(Pre_dev_data[i][j]['IDs']) + '.wav')
        f.write('\t')
        x= []
        for w in range(len(Pre_dev_data[i][j]['Utterance'])):
            if(is_UTF_8(Pre_dev_data[i][j]['Utterance'][w])):
                x.extend(Pre_dev_data[i][j]['Utterance'][w])
            else:
                x.extend("'")
        f.write(''.join(x))
        f.write('\n')
f.close()

f = open("c.txt", 'w')
for i in range(len(Pre_test_data)):
    for j in range(len(Pre_test_data[i])):
        f.write(str(Pre_test_data[i][j]['IDs']) + '.wav')
        f.write('\t')
        x= []
        for w in range(len(Pre_test_data[i][j]['Utterance'])):
            if(is_UTF_8(Pre_test_data[i][j]['Utterance'][w])):
                x.extend(Pre_test_data[i][j]['Utterance'][w])
            else:
                x.extend("'")
        f.write(''.join(x))
        f.write('\n')
f.close()
'''
