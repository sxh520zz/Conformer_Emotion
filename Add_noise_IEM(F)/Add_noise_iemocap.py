#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jan  9 20:32:28 2018

@author: shixiaohan
"""
import re
import wave
import numpy as np
import python_speech_features as ps
import os
import glob
import pickle
import csv
import librosa
import soundfile as sf

Data_dir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/journal_Data'
rootdir = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/IEMOCAP_full_release/'
path = '/media/shixiaohan-toda/70e9f22b-13a4-429e-acad-a8786e1f1143/DataBase/Experiment_Data/IEM_Add_noise/'

snr = "no_noise_"
def emo_change(x):
    if x == 'xxx' or x == 'oth':
        x = 0
    if x == 'neu':
        x = 1
    if x == 'hap':
        x = 2
    if x == 'ang':
        x = 3
    if x == 'sad':
        x = 4
    if x == 'exc':
        x = 5
    if x == 'sur':
        x = 6
    if x == 'fea':
        x = 7
    if x == 'dis':
        x = 8
    if x == 'fru':
        x = 9
    return x

def awgn(audio,snr):
    audio_power = audio **2
    audio_average_power = np.mean(audio_power)
    audio_average_db = 10* np.log10(audio_average_power)
    noise_average_db = audio_average_db - snr
    noise_average_power = 10 **(noise_average_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_average_power), len(audio))
    return audio+noise

def read_file(filename):
    file = wave.open(filename, 'r')
    params = file.getparams()
    nchannels, sampwidth, framerate, wav_length = params[:4]
    str_data = file.readframes(wav_length)
    wavedata = np.fromstring(str_data, dtype=np.short)
    time = np.arange(0, wav_length) * (1.0 / framerate)
    file.close()
    return wavedata, time, framerate

def Read_IEMOCAP_Spec():
    filter_num = 40
    train_num = 0
    train_mel_data = []
    for speaker in os.listdir(rootdir):
        if (speaker[0] == 'S'):
            sub_dir = os.path.join(rootdir, speaker, 'sentences/wav')
            for sess in os.listdir(sub_dir):
                if (sess[7] in ['i','s']):
                    file_dir = os.path.join(sub_dir, sess, '*.wav')
                    files = glob.glob(file_dir)
                    for filename in files:
                        wavname = filename.split("/")[-1][:-4]
                        new_name =  wavname + ".wav"
                        data, time, rate = read_file(filename)
                        #data_1 = awgn(data,-5)
                        new_path = path + snr + 'db/'
                        new_path_data = new_path + new_name
                        sf.write(new_path_data, data,rate)
                        train_num = train_num +1
                print(train_num)
    return train_mel_data

if __name__ == '__main__':
    train_data_spec = Read_IEMOCAP_Spec()
