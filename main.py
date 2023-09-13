import numpy as np
import tensorflow as tf

'''미분연산 및 학습'''

from tensorflow import keras
from keras import Input
from keras.models import Model
from keras.layers import Dense
import matplotlib.pyplot as plt
import math

def test_system(x):
    # return 0.01 * x * x + 2.0
    return math.sin(0.06*x)

def get_model():
    tf.random.set_seed(1000)

    input = tf.keras.Input(shape=(1,), name="Input")
    hidden = Dense(1024, activation="tanh", name="Hidden")(input)
    hidden = Dense(1024, activation="tanh", name="Hidden1")(hidden)
    hidden = Dense(1024, activation="relu", name="Hidden2")(hidden)
    hidden = Dense(1024, activation="tanh", name="Hidden3")(hidden)
    hidden = Dense(128, activation="relu", name="Hidden4")(hidden)
    output = Dense(1, activation="tanh", name="Output")(hidden)
    '''
    input = Input(shape=(1,), name="input")
    output = Dense(1, activation ="linear", name="Output")(input)
    '''
    model = Model(inputs=[input], outputs=[output])

    opt = keras.optimizers.Adam(learning_rate=0.0025)
    model.compile(loss='mse', optimizer=opt, metrics=['mse', 'mae'])
    model.summary()
    return model

if __name__=='__main__':
    x_datas = np.array(range(0, 300+1, 2))
    y_datas = []
    for x in x_datas:
        y_datas.append(test_system(x))
    y_datas = np.array(y_datas)

    plt.plot(x_datas, y_datas)
    plt.show()

    model = get_model()

    history = model.fit(x_datas, y_datas, epochs=2000, shuffle=True)
    plt.plot(history.history['loss'], 'b', label='loss')
    plt.show()

    x_test = np.array(range(1, 301, 2))
    result = model.predict(x_test)


    plt.plot(x_datas, y_datas, 'b')
    plt.plot(x_test, result, 'r')
    plt.show()

    weights = model.get_weights()
    print("W : ", weights[0], " b : ", weights[1])













