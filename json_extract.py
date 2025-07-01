import json
import os
import numpy as np
import random

def json_ext(path, flag):
    if flag == 'train':
        dataset_path = r"Y:\dataset_low_snr\noisy_trainset_28spk_wav"
        #dataset_path = os.path.join(path, 'train', 'mix')# read Train mixed speech name
        json_path = r"C:\Users\Admin\Downloads\codes\DB_CPT"
        json_path = os.path.join(json_path, 'Json', 'train')
        os.makedirs(json_path, exist_ok=True)
    else:
        dataset_path = r"Y:\dataset_low_snr\noisy_testset_wav"
        #dataset_path = os.path.join(path, 'cv', 'mix')# read CV mixed speech name 
        json_path = r"C:\Users\Admin\Downloads\codes\DB_CPT"
        json_path = os.path.join(json_path, 'Json', 'cv')
        os.makedirs(json_path, exist_ok=True)
    data_dir = os.listdir(dataset_path)
    file_num = len(data_dir)
    random.shuffle(data_dir)
    data_list = []

    for i in range(file_num):
        file_name = data_dir[i]
        file_name = os.path.splitext(file_name)[0]
        data_list.append(file_name)

    with open(os.path.join(json_path, 'files.json'), 'w') as f :
        json.dump(data_list, f, indent=4)

file_path = '/content/drive/MyDrive/NetworkHakim/train/clean_trainset_28spk_wav'
json_ext(file_path, flag='train')
#json_ext(file_path, flag='cv')
