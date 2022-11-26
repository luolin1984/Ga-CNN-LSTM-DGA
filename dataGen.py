import copy
import pandas as pd
import numpy as np
import random


def read_data():
    df = pd.read_csv('DGA2.csv')
    data = np.array(df)
    random.shuffle(data) #随机打乱
    #取前70%为训练集
    allurl_fea = [d[0] for d in data]
    data_train = data[:int(0.8*len(allurl_fea))]
    ddata_train = data_train[:,:5]
    llab_train = data_train[:, 5].astype('int64')
    #剩余百分之30为测试集
    data_test = data[int(0.8*len(allurl_fea)):]
    ddata_test = data_test[:, :5]
    llab_test = data_test[:, 5].astype('int64')
    return ddata_train, llab_train, ddata_test, llab_test

def gen_seq_data(n_samples):
    d_train, l_train, d_test, l_test = read_data()
    data_train,label_train,data_test,label_test = [],[],[],[]
    seq_data_train, seq_data_test = [], []
    length_train = d_train.shape[0] - n_samples + 1
    length_test = d_test.shape[0] - n_samples + 1
    for j in range(n_samples):
        data_train.append(d_train[j : j + length_train])
        #label_train.append(l_train[j + length_train -1])
    data_train = np.hstack(data_train)
    seq_data_train.append(data_train)
    label_train.append(l_train[:-n_samples+1])

    for i in range(n_samples):
        data_test.append(d_test[i : i + length_test])
        #label_test.append(l_test[i + length_test -1])
    data_test = np.hstack(data_test)
    seq_data_test.append(data_test)
    label_test.append(l_test[:-n_samples+1])
    return np.vstack(seq_data_train), np.array(label_train), np.vstack(seq_data_test), np.array(label_test)


def output_encodes(yhat):
    # 形式三@：各类概率值
    y_score_pro = copy.copy(yhat) # 注意：python里，b=a之后，a值也是变的

    # 形式二@：one-hot值
    yhat_one = copy.copy(yhat)
    #yhat_one[yhat_one < 0.5] = 0
    #yhat_one[yhat_one > 0.5] = 1
    # 形式一@：原始值（0或1或2）
    y_score = [np.argmax(yhat_ones) for yhat_ones in yhat_one]   # one_hot编码转为np.array
    y_score = np.array(y_score)
    return yhat_one, y_score_pro, y_score
