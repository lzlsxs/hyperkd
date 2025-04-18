from preprocess import data_preprocss
from scipy.io import savemat,loadmat
import torch
import torch.utils.data as Data

x_train_band, y_train, x_test_band,y_test, num_classes = data_preprocss('/Users/lzlsxs/Documents/life-long/my-code/data/Pavia.mat',3,5)
print(y_train)
print(y_train.dtype)
y_t = y_train + 5
print(y_t)
print(y_t.dtype)
'''
train_mat = {'data':x_train_band,'label':y_train,'num_classes':num_classes}
savemat('train.mat',train_mat)
data = loadmat('train.mat')
x_data = data['data']
y_label = data['label']
num_classes1 = data['num_classes']
print(num_classes1[0,0])
'''
