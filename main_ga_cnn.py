import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from dataGen import gen_seq_data,output_encodes
from GaCNN import init_net,fitness,selection,crossover,mutate
import scikitplot as skplt
from matplotlib import pyplot as plt

target = [0,1,2,3,4,5]
TIME_STEPS = 5  
n_classes = 6
BATCH_SIZE, EPOCHS = 32, 100 

data_train, label_train, data_test, label_test  = gen_seq_data(TIME_STEPS)

# Preprocess: zero mean and unit standard variance
scaler = StandardScaler().fit(data_train)
train_data = scaler.transform(data_train)
test_data = scaler.transform(data_test)

# Reshape the training dataset
train_data_seq = train_data.reshape(train_data.shape[0], TIME_STEPS, train_data.shape[1] // TIME_STEPS)
test_data_seq = test_data.reshape(test_data.shape[0], TIME_STEPS, test_data.shape[1] // TIME_STEPS)
label_train_reshape, label_test_reshape = [], []
for i in label_train:
    label_train_reshape = np.append(label_train_reshape, i)
for j in label_test:
    label_test_reshape = np.append(label_test_reshape, j)
train_labels_one_hot = to_categorical(label_train_reshape, n_classes)
test_labels_one_hot = to_categorical(label_test_reshape, n_classes)

# model 1: CNN-LSTM
train_data_cnn_lstm = copy.copy(train_data).reshape(train_data_seq.shape[0], train_data_seq.shape[1], train_data_seq.shape[2], 1)
test_data_cnn_lstm = copy.copy(test_data).reshape(test_data_seq.shape[0], test_data_seq.shape[1], test_data_seq.shape[2], 1)

n_sensors = train_data_cnn_lstm.shape[2]
input_shape = (TIME_STEPS, n_sensors, 1)
print(input_shape)

train_size = len(train_data_cnn_lstm)
test_size = len(test_data_cnn_lstm)

P = 3  # Population
G = 3  # Generation
B = 32  # Batch size
C = n_classes  # Class number
T = 0.994  # Threshold
N = init_net(p=P)  # Create population number networks

accuracy_list = []
yhat_list = []
for g in range(G):
    print('Generation {}'.format(g + 1))
    N = fitness(n = N,    n_c = C,  i_shape = input_shape,
                x = train_data_cnn_lstm[:train_size],
                y = train_labels_one_hot[:train_size],
                b = B,
                x_test = test_data_cnn_lstm[:test_size],
                y_test = test_labels_one_hot[:test_size])
    N = selection(n=N)
    N = crossover(n=N)
    N = mutate(n=N)

for q in N:
    accuracy_list.append(q.ac * 100)
    yhat_list.append(q.yhat)
    if q.ac > T:
        print('Threshold satisfied')
        print(q.init_params())
        print('Best accuracy: {}%'.format(q.ac * 100))
        exit(code=0)

print("The best accuracy so far {}%".format(max(accuracy_list)))


max_acc = [i for i in accuracy_list if max(accuracy_list) in accuracy_list]
yhat = [i for i in yhat_list if max(accuracy_list) in accuracy_list]

index_max_acc = accuracy_list.index(max(accuracy_list))
hat_y = np.array(yhat)
hat_y_max = hat_y[index_max_acc]

yhat_one_cnn, y_score_pro_cnn, y_score_cnn = output_encodes(hat_y_max)

ax_cnn = skplt.metrics.plot_roc(label_test_reshape, y_score_pro_cnn) # ROC Curve
handles, labels = plt.gca().get_legend_handles_labels()
x = [c.replace('class 1','class '+str(target[1])) for c in labels]
plt.legend(handles,x)
plt.show()

skplt.metrics.plot_precision_recall_curve(label_test_reshape, y_score_pro_cnn, cmap='nipy_spectral') # Precision-Recall Curve
handles, labels = plt.gca().get_legend_handles_labels()
x = [c.replace('class 1','class '+str(target[1])) for c in labels]
plt.legend(handles,x)
plt.show()
