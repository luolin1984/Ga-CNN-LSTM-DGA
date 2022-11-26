from numpy.random import randint
from random import choice
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from numpy.random import uniform

class Net:
    def __init__(self):
        self.ep = randint(1, 3)               # epoch
        self.f1 = randint(30, 34)             # filter size 1
        self.f2 = randint(62, 66)             # filter size 2
        self.u1 = randint(126, 130)           # unit
        self.k1 = choice([(1, 3), (3, 3)])    # kernel size 1
        self.k2 = choice([(1, 3), (3, 3)])    # kernel size 2
        self.d1 = choice([0.25, 0.5])         #   dropout 1
        self.d2 = choice([0.25, 0.5])         #   dropout 2
        self.a1 = 'relu'                      # activation 1
        self.a2 = 'relu'                      # activation 2
        self.a3 = 'relu'                      # activation 3
        self.a4 = 'softmax'                   # activation 4
        self.lf = 'categorical_crossentropy'  # loss function
        self.op = 'Adam'                      # optimization
        self.ac = 0                           # accuracy
        self.yhat = list([0,0,0,0,0])         # yhat

    def init_params(self):
        params = {'epochs': self.ep,
                  'filter1': self.f1,
                  'kernel1': self.k1,
                  'activation1': self.a1,
                  'filter2': self.f2,
                  'kernel2': self.k2,
                  'activation2': self.a2,
                  'pool_size': (1, 1),
                  'dropout1': self.d1,
                  'unit1': self.u1,
                  'activation3': self.a3,
                  'dropout2': self.d2,
                  'activation4': self.a4,
                  'loss': self.lf,
                  'optimizer': self.op}
        return params


def init_net(p):
    return [Net() for _ in range(p)]


def fitness(n, n_c, i_shape, x, y, b, x_test, y_test):
    for cnt, i in enumerate(n):
        p = i.init_params()
        ep = p['epochs']
        f1 = p['filter1']
        f2 = p['filter2']
        k1 = p['kernel1']
        k2 = p['kernel2']
        d1 = p['dropout1']
        d2 = p['dropout2']
        ps = p['pool_size']
        u1 = p['unit1']
        a1 = p['activation1']
        a2 = p['activation2']
        a3 = p['activation3']
        a4 = p['activation4']
        lf = p['loss']
        op = p['optimizer']

        try:                                # Parameter name    # Suggested value
            m = net_model(ep=ep,            # epoch number             12
                          f1=f1,            # filter size 1            32
                          f2=f2,            # filter size 2            64
                          k1=k1,            # kernel 1               (3, 3)
                          k2=k2,            # kernel 2               (3, 3)
                          a1=a1,            # activation 1           'relu'
                          a2=a2,            # activation 2           'relu'
                          a3=a3,            # activation 3           'relu'
                          a4=a4,            # activation 4           'softmax'
                          d1=d1,            # dropout 1                0.25
                          d2=d2,            # dropout 2                0.5
                          u1=u1,            # neuron number            128
                          ps=ps,            # pool size               (2, 2)
                          op=op,            # optimizer               'adadelta'
                          lf=lf,            # loss function           'categorical crossentropy'
                          n_c=n_c,          # number of channel
                          i_shape=i_shape,  # input shape
                          x=x,              # train data
                          y=y,              # train label
                          b=b,              # bias value
                          x_test=x_test,    # test data
                          y_test=y_test)    # test label

            s = m.evaluate(x=x_test, y=y_test, verbose=0)
            yhat = m.predict(x_test, batch_size=ep)
            mp = "./model_save/model"+str(round(s[1],6))+".h5"
            m.save(mp)
            i.ac = s[1]
            i.yhat = yhat
            print('Accuracy: {}'.format(i.ac * 100))
        except Exception as e:
            print(e)
    return n


def net_model(ep, f1, f2, k1, k2, a1, a2, a3, a4, d1, d2, u1, ps, op, lf, n_c, i_shape, x, y, b, x_test, y_test):
    model = Sequential()
    model.add(layer=Conv2D(filters=f1, kernel_size=k1, activation=a1, input_shape=i_shape))
    model.add(layer=Conv2D(filters=f2, kernel_size=k2, activation=a2))
    model.add(layer=MaxPooling2D(pool_size=ps))
    model.add(layer=Dropout(rate=d1))
    model.add(layer=Flatten())
    model.add(layer=Dense(units=u1, activation=a3))
    model.add(layer=Dropout(rate=d2))
    model.add(layer=Dense(units=n_c, activation=a4))
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.compile(optimizer=op, loss=lf, metrics=['accuracy'])
    model.fit(x=x, y=y, batch_size=b, epochs=ep, verbose=0, validation_data=(x_test, y_test))
    return model


def selection(n):
    n = sorted(n, key=lambda j: j.ac, reverse=True)
    n = n[:int(len(n))]
    return n


def crossover(n):
    offspring = []
    p1 = choice(n)
    p2 = choice(n)
    c1 = Net()
    c2 = Net()
    c1.ep = int(p2.ep) + 2
    c2.ep = int(p1.ep) + 2
    offspring.append(c1)
    offspring.append(c2)
    n.extend(offspring)
    return n


def mutate(n):
    for i in n:
        if uniform(0, 1) <= 0.1:
            i.ep += randint(0, 5)
            i.u1 += randint(0, 5)
    return n