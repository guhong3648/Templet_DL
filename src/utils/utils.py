import os

import cv2
import time
import glob
import h5py
import random
import natsort
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from copy import deepcopy
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, Dataset
from sklearn.model_selection import train_test_split


def make_dir(path):
    if not os.path.exists(path): 
        os.mkdir(path)
        os.chmod(path, 0o777)

def globsort(path):
    return natsort.natsorted(glob.glob(path))

def listdir(path):
    return natsort.natsorted(os.listdir(path))

def minMax_scaling(x, lb=0, m=None, M=None):
    if m==None: m = x.min()
    if M==None: M = x.max()
    if lb==0:
        return (x-m)/(M-m)
    if lb==-1:
        return (2*x-(M+m))/(M-m)

def get_split_idx(ratio=[0.6, 0.2, 0.2], shuffle=True, generate=False):
    if generate: 
        list_patient = listdir('../data')
        print(len(list_patient))
        train, valid = train_test_split(list_patient, train_size=ratio[0], shuffle=shuffle)
        valid, test  = train_test_split(valid       , train_size=ratio[1]/(ratio[1]+ratio[2]), shuffle=shuffle)
        train = natsort.natsorted(train)
        valid = natsort.natsorted(valid)
        test = natsort.natsorted(test)
        
        split_idx = {'train':train, 
                     'valid':valid, 
                     'test':test}
        
        torch.save(split_idx, './utils/split_idx.pt')
    
    else:
        split_idx = torch.load('./utils/split_idx.pt')
        
    return split_idx

class Dataset_Temp(Dataset):
    def __init__(self, hp, phase='train'):
        self.hp = hp
        self.phase = phase
        split_idx = get_split_idx()[phase]
        
        self.list_data = [h5py.File(f'{hp.path_data}/{patient}/data.h5', 'r') for patient in split_idx]
        self.list_n = [len(data['X']) for data in self.list_data]
        self.N = len(self.list_data)
        
    def __len__(self):
        return self.N
    
    def __getitem__(self, idx):
        k = np.random.randint(self.list_n[idx])
        X_data = torch.from_numpy(self.list_data[idx]['X'][k]).to(torch.float32).unsqueeze(0)
        y_data = torch.from_numpy(self.list_data[idx]['y'][k]).to(torch.float32)
        
        return X_data, y_data
    
class Dataset_Temp_inference(Dataset):
    def __init__(self, hp, patient):
        self.hp = hp
        
        self.data = h5py.File(f'{hp.path_data}/{patient}/data.h5', 'r')
        self.X_data = self.data['X'][:]
        self.y_data = self.data['y'][:]
        self.N = len(self.X_data)
        
    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        X_data = torch.from_numpy(self.X_data[idx]).to(torch.float32).unsqueeze(0)
        y_data = torch.from_numpy(self.y_data[idx]).to(torch.float32)
        
        return X_data, y_data
    
class MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p, y, w=None):
        res = torch.abs(p-y).mean()
        
        return res
    
class BCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = 1e-5
        
    def forward(self, p, y, w=None):
        res = -(y*torch.log(p+self.smooth)+(1-y)*torch.log(1-p+self.smooth)).mean()
        
        return res

class ACC(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, p, y):
        p = (p>self.alpha).float()
        res = (p==y).float()
        res = res.mean()
        
        return res

def es(loss, epoch, inverse=False):
    escounter = ['E', '|']
    loss = loss[1:epoch+1, 1]
    loss_best = torch.min(loss, axis=0)[0]
    loss_curr = loss[epoch-1]
    if inverse==False:
        T = loss_best>=loss_curr
    else:
        T = loss_best<=loss_curr
    T = [escounter[i] for i in T]
    
    return T

def get_ylim(loss, title, alpha=2, margin=0.1):
    m = loss.min()
    M = loss.max()
    a = loss.mean()
    
    if M-a>(a*alpha):
        M = a*alpha
        
    m = m*(1-margin)
    M = M*(1+margin)
    
    l = len(loss)//2
    
    if torch.sum(loss[0:10][0])>torch.sum(loss[-10:][0]): 
        plt.axhline(loss[:, 1].min(), color='black', linewidth=0.7, linestyle='--', alpha=0.7)
        epoch = torch.argmin(loss[:, 1])
        plt.axvline(epoch, color='r', linewidth=0.7, linestyle='--', alpha=0.7)
        plt.title(f'{title}: {loss[:, 1].min():0.4f} ({loss[l:, 1].mean():0.4f})')
    else:
        plt.axhline(loss[:, 1].max(), color='black', linewidth=0.7, linestyle='--', alpha=0.7)
        epoch = torch.argmax(loss[:, 1])
        plt.axvline(epoch, color='r', linewidth=0.7, linestyle='--', alpha=0.7)
        plt.title(f'{title}: {loss[:, 1].max():0.4f} ({loss[l:, 1].mean():0.4f})')

    return m, M

def plot_history(hp, epoch=0, loss_titles=None, rate_titles=None):
    history = torch.load(f'../res/{hp.name}/model/history.pt')
    loss_keys = history['loss_keys']
    rate_keys = history['rate_keys']
    if 'Loss_Final' in loss_keys: loss_keys.remove('Loss_Final')
    
    loss = history['loss'][1:]
    rate = history['rate'][1:]
    k = len(loss[:, 1, 0][loss[:, 1, 0]>0])
    loss = loss[:k, :, :]
    rate = rate[:k, :, :]
    
    if loss_titles==None:
        loss_titles = loss_keys
    if rate_titles==None:
        rate_titles = rate_keys
        
    N = len(loss_keys) + len(rate_keys)
    n, m = N//3+1, 3
    
    plt.figure(figsize=(14, 4))
    plt.subplots_adjust(hspace=0.4)
    plt.suptitle(f'{hp.name} ({len(loss)})')
    for i in range(len(loss_titles)):
        plt.subplot(n, m, i+1)
        plt.plot(loss[:, 0, i], color='b', linewidth=0.7, alpha=0.7, label='Train')
        plt.plot(loss[:, 1, i], color='r', linewidth=0.7, alpha=0.7, label='Valid')
        plt.ylim(get_ylim(loss[:, :, i], loss_titles[i], alpha=1.5))
        if i==2: plt.legend(loc='upper right')

    for j in range(len(rate_titles)):
        plt.subplot(n, m, len(loss_titles)+j+1)
        plt.plot(rate[:, 0, j], color='b', linewidth=0.7, alpha=0.7)
        plt.plot(rate[:, 1, j], color='r', linewidth=0.7, alpha=0.7)
        plt.ylim(get_ylim(rate[:, :, j], rate_titles[j], alpha=1.5))

def inference(hp, epoch, batch_size=None, n=0, clear=True):
    print(f'Inference: {hp.name} / epoch={epoch}')
    model = torch.load(f'{hp.path_model}/model_{epoch}.pt', map_location='cpu')
    model = model.to(hp.device).eval()
    make_dir(f'{hp.path_inference}/{epoch}')
    if batch_size==None: batch_size = hp.batch_size_inference
    
    split_idx = get_split_idx()['test']
    if n>0: split_idx = split_idx[:n]
    
    for n, patient in enumerate(split_idx):
        print(f'[{n+1:3.0f}/{len(split_idx):3.0f}]: {patient}')
        test_set = Dataset_Temp_inference(hp, patient)
        test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=batch_size, 
                                 num_workers=2, pin_memory=True)
        
        p_data = []
        for X, y in test_loader:
            with torch.no_grad():
                p = model(X.to(hp.device)).cpu()
                p_data.append(p)
        p_data = torch.concat(p_data, axis=0)
        
        with h5py.File(f'{hp.path_inference}/{epoch}/{patient}.h5', 'w') as f:
            f.create_dataset('p_data', data=p_data)
            f.close()
            
    if clear: clear_output()
    
def evaluation(hp, epoch, alpha=0.5, n=0, clear=True):
    print(f'Evaluation: {hp.name} / epoch={epoch}')
    print(f'{"Patient":<30} {"ACC":>8}')
    make_dir(f'{hp.path_evaluation}/{epoch}')
    ACC_func = ACC()
    list_patient = listdir(f'{hp.path_inference}/{epoch}')
    if n>0: list_patient = list_patient[:n]
    for n, patient in enumerate(list_patient):
        patient = patient[:-3]
        print(f'[{n+1:3.0f}/{len(list_patient):3.0f}]: {patient:<13}', end=' ')
        
        # Loading
        print('[L', end='')
        p_data = h5py.File(f'{hp.path_inference}/{epoch}/{patient}.h5')['p_data'][:]
        y_data = h5py.File(f'{hp.path_data}/{patient}/data.h5')['y'][:]
        p_data = torch.from_numpy(p_data).to(torch.float32)
        y_data = torch.from_numpy(y_data).to(torch.float32)
        
        # Processing (Thresholding)
        print('P', end='')
        p_data = (p_data>alpha).to(torch.float32)
        
        # Scoring
        print('S]', end=' ')
        ACC_res = ACC_func(p_data, y_data)
        
        # Printing
        print(f'{"":<1} {ACC_res:0.4f}')
        
        # Save
        res = {'ACC':ACC_res}
        torch.save(res, f'{hp.path_evaluation}/{epoch}/{patient}.pt')
    
    if clear: clear_output()

def print_total(list_model, list_epoch=None):
    n = 0
    for model in list_model:
        if model=='--':
            print('-'*80)
            continue
        
        if list_epoch==None:
            list_epoch = listdir(f'../res/{model}/evaluation')
        
        for epoch in list_epoch:
            n += 1
            list_ACC_res = []
            list_eval = listdir(f'../res/{model}/evaluation/{epoch}')
            for eval in list_eval:
                res = torch.load(f'../res/{model}/evaluation/{epoch}/{eval}')
                list_ACC_res.append(res['ACC'])
                
            list_ACC_res = torch.stack(list_ACC_res)
            
        if (n==1):
            print(f'[{"epoch":>5}] {"model":<20} {"ACC":>11}')
        print(f'[{epoch:>5}] {model:<20}', end= ' ')
        print(f'{torch.mean(list_ACC_res):0.4f}±{torch.std(list_ACC_res):0.2f}', end=' ')
        print()