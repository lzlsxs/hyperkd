import os
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as Data
from scipy.io import loadmat
from datasets.preprocess_vit import data_preprocss


class My_Dataset(Data.TensorDataset):
    def __init__(self,x_data,y_label):
        super(My_Dataset,self).__init__(x_data,y_label)
        self.transform = None
        self.labels = y_label
        self.images = x_data

    
def My_Loader(my_dataset,batchsize,is_shuffle):
    return Data.DataLoader(my_dataset,batch_size=batchsize,shuffle=is_shuffle)
    
def get_loader(data_path,batchsize,patch, band_patch,is_shuffle,tsk_offset= 0):
    x_train_band, y_train, x_test_band,y_test, num_classes,band = data_preprocss(data_path, patch, band_patch)
    y_train = y_train + tsk_offset
    y_test = y_test + tsk_offset
    # data = loadmat(data_path)
    # x_data = data['data']
    # y_label = data['label'] + tsk_offset
    # num_classes = data['num_classes'][0,0]
    x_data_train=torch.from_numpy(x_train_band).type(torch.FloatTensor) 
    y_label_train=torch.from_numpy(y_train).type(torch.LongTensor)
    x_data_test=torch.from_numpy(x_test_band).type(torch.FloatTensor) 
    y_label_test=torch.from_numpy(y_test).type(torch.LongTensor)
    train_dataset = My_Dataset(x_data_train,y_label_train)
    test_dataset = My_Dataset(x_data_test,y_label_test)
    train_loader = My_Loader(train_dataset,batchsize,is_shuffle)
    test_loader = My_Loader(test_dataset,batchsize,is_shuffle)
    return train_loader,test_loader, num_classes, band

