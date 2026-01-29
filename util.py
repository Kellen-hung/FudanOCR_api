# util.py
import torch
import os
import shutil
import pickle as pkl
import torch.nn as nn
import copy

from data.datasetReader import lmdbDataset, resizeNormalize, InferenceDataset
from config import config
from shutil import copyfile

mse_loss = nn.MSELoss()
alphabet_character_file = open(config['alpha_path'])                # 開啟字典檔
alphabet_character = list(alphabet_character_file.read().strip())   # 把字典做成 list
alphabet_character_raw = ['START']

for item in alphabet_character:
    alphabet_character_raw.append(item)

alphabet_character_raw.append('END')
alphabet_character = alphabet_character_raw # "list head 多 START" 跟 "list tail 多 END"

# 做 map
alp2num_character = {} 

for index, char in enumerate(alphabet_character):
    alp2num_character[char] = index

def get_inference_data(img_dir):
    print(f"\033[32m[info] Detect inference dataset: {img_dir}\033[0m")
    inference_dataset = []
    transform = resizeNormalize((config['imageW'], config['imageH']))
    dataset = InferenceDataset(img_dir, transform=transform)
    # print(dataset)
    _ = torch.utils.data.DataLoader(
        dataset, batch_size=config['batch'], shuffle=False, num_workers=16, pin_memory=True
    )
    
    inference_dataset.append(dataset)
    inference_dataset_total = torch.utils.data.ConcatDataset(inference_dataset)
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=4, 
    )
    return inference_dataloader

def get_dataloader(root,shuffle=False):
    if root.endswith('pkl'):
        print(f"\033[32m[info] Detect dataset: {root}\033[0m")
        f = open(root,'rb')
        dataset = pkl.load(f)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], shuffle=shuffle, num_workers=16, pin_memory=True, prefetch_factor=4,
        )
    else:
        # for training and validation
        print(f"\033[32m[info] Detect dataset: {root}\033[0m")
        dataset = lmdbDataset(root,resizeNormalize((config['imageW'],config['imageH'])))
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config['batch'], shuffle=shuffle, num_workers=16, pin_memory=True, prefetch_factor=4,
        )
    return dataloader, dataset

def get_data_package():
    train_dataset = []
    for dataset_root in config['train_dataset'].split(','):
        _ , dataset = get_dataloader(dataset_root,shuffle=True)
        train_dataset.append(dataset)
    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=16, pin_memory=True, prefetch_factor=4, 
    )

    validation_dataset = []
    for dataset_root in config['validation_dataset'].split(','):
        _ , dataset = get_dataloader(dataset_root,shuffle=True)
        validation_dataset.append(dataset)
    validation_dataset_total = torch.utils.data.ConcatDataset(validation_dataset)

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=16, pin_memory=True, prefetch_factor=4, 
    )
    return train_dataloader, validation_dataloader

# radical list
r2num = {}
alphabet_radical = []
alphabet_radical.append('PAD')
lines = open(config['radical_path'], 'r').readlines()
for line in lines:
    alphabet_radical.append(line.strip('\n'))
alphabet_radical.append('$')
# r2num = map (一個字一個編號 1,2,3 ...)
for index, char in enumerate(alphabet_radical):
    r2num[char] = index

# 做 IDS 的 map
dict_file = open(config['decompose_path'], 'r').readlines()
char_radical_dict = {}
for line in dict_file:
    line = line.strip('\n')
    char, r_s = line.split(':')
    char_radical_dict[char] = r_s.split(' ') 

# 把對應部首轉換成數字 (用 r2num 的 key-value)
def convert_char(label):
    # r_label = 沒有答案的 IDS (每個 element 的 tail 有 $, 每個 element 都是一個字的 decompose)
    r_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])
        r_tmp.append('$')
        r_label.append(r_tmp)
    
    # 把每個部首（或 $）轉成對應的數字 (map) 存進 text_tensor
    # 數字是 0 的代表留白 (用不到 30 個 tensor element)
    text_tensor = torch.zeros(batch, 30).long().cuda() # 二維 list, text_tensor[batch][30]
    for i in range(batch):
        tmp = r_label[i]
        for j in range(len(tmp)):
            '''
            # 跳過沒有在 radical 字典裡的字 (懶得動資料集, 後面應該要改)
            if tmp[j] not in r2num:
                continue
            '''
            text_tensor[i][j] = r2num[tmp[j]]
    return text_tensor

def get_radical_alphabet():
    return alphabet_radical

def converter(label):

    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            text_input[i][j + 1] = alp2num[label[i][j]]

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])

    return length, text_input, text_all, string_label

def get_alphabet():
    return alphabet_character

def tensor2str(tensor):
    alphabet = get_alphabet()
    string = ""
    for i in tensor:
        if i == (len(alphabet)-1):
            continue
        string += alphabet[i]
    return string

# 檢測程式是否在 Screen 裡運行
def must_in_screen():
    text = os.popen('echo $STY').readlines()
    string = ''
    for line in text:
        string += line
    if len(string.strip()) == 0:
        print("run in the screen!")
        exit(0)

# 建立實驗紀錄資料夾，並複製當前程式碼檔案
def saver():
    try:
        shutil.rmtree('./history/{}'.format(config['exp_name']))
    except:
        pass
    os.mkdir('./history/{}'.format(config['exp_name']))

    import time

    print('\033[33m**** Experiment Name: {} ****\033[0m'.format(config['exp_name']))

    localtime = time.asctime(time.localtime(time.time()))
    f = open(os.path.join('./history', config['exp_name'], str(localtime)),'w+')
    f.close()

    src_folder = './'
    exp_name = config['exp_name']
    dst_folder = os.path.join('./history', exp_name)

    file_list = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    for file_name in file_list:
        src = os.path.join(src_folder, file_name)
        dst = os.path.join(dst_folder, file_name)
        copyfile(src, dst)