from scipy.io import loadmat
from scipy.io import savemat
import numpy as np
from sklearn.decomposition import PCA


pca_components = 36
def applyPCA(X, numComponents):

    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX
def my_normalize(data):
    input_normalize = np.zeros(data.shape)
    for i in range(data.shape[2]):
        input_max = np.max(data[:,:,i])
        input_min = np.min(data[:,:,i])
        input_normalize[:,:,i] = (data[:,:,i]-input_min)/(input_max-input_min)
    return input_normalize

def choose_train_or_test_point(data,num_classes):
    number_choose =[]
    pos_choose = {}
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(data==(i+1))
        number_choose.append(each_class.shape[0])
        pos_choose[i] = each_class
    total_pos_choose = pos_choose[0]
    for i in range(1, num_classes):
        total_pos_choose = np.r_[total_pos_choose,pos_choose[i]]
    total_pos_choose = total_pos_choose.astype(int)
    return total_pos_choose,number_choose

def choose_true_point(data, num_classes):
    number_choose =[]
    pos_choose = {}
    for i in range(num_classes + 1):
        each_class = []
        each_class = np.argwhere(data== i)
        number_choose.append(each_class.shape[0])
        pos_choose[i] = each_class
    total_pos_choose = pos_choose[0]
    for i in range(1, num_classes+1):
        total_pos_choose = np.r_[total_pos_choose,pos_choose[i]]
    total_pos_choose = total_pos_choose.astype(int)
    return total_pos_choose,number_choose

def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    return mirror_hsi

def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image


def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            #x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            #x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:(nn-i)] = x_train_reshape[:,:,(band-nn+i):]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,(nn-i):] = x_train_reshape[:,:,:(band-nn+i)]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band


def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    #x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    '''
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)
    '''
    '''
    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")
    '''
   
    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    x_train_band = x_train_band.transpose(0,2,1)
    x_test_band = x_test_band.transpose(0,2,1)

    # x_train_band = x_train.transpose(0,3,1,2)
    # x_test_band = x_test.transpose(0,3,1,2)
    
    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    #print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")

    return x_train_band, x_test_band


def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    #y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    '''
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    '''
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    #y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    #print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test



def data_preprocss(data_path, patch, band_patch):
    data = loadmat(data_path)['input']
    data = applyPCA(data, numComponents=pca_components)
    TR = loadmat(data_path)['TR']
    TE = loadmat(data_path)['TE']
    TRE = TR + TE
    num_classes = np.max(TRE)

    data__normalize = my_normalize(data)
    height, width, band = data.shape
    print("height={0},width={1},band={2},num_classes={3}".format(height, width, band,num_classes))
    train_point , train_num = choose_train_or_test_point(TR,num_classes)
    test_point, test_num = choose_train_or_test_point(TE, num_classes)
    #true_point, true_num = choose_true_point(TRE, num_classes)
    true_point =[]
    true_num=[]
    data_mirror = mirror_hsi(height,width,band,data__normalize,patch)
    x_train_band,x_test_band= train_and_test_data(data_mirror,band,train_point,test_point,true_point,patch,band_patch)
    y_train ,y_test = train_and_test_label(train_num, test_num,true_num,num_classes)
    return x_train_band, y_train, x_test_band,y_test, num_classes,band