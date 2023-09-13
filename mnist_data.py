import os
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

path_dir = './images/training/'
label_list = os.listdir(path_dir)
label_list.sort()

all_files = []
for i in label_list:
    path_dir = './images/training/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    all_files.append(file_list)

# target = 3
#
# img = Image.open('./images/training/{0}/'.format(target) + all_files[target][0])
# img_arr = np.array(img)
# print(img_arr)
# print(img_arr.shape)

x_train_data = []
y_train_data = []
for num in label_list:
    for number in all_files[int(num)]:
        img_path = ('./images/training/{0}/{1}'.format(num, number))
        # print("load : " + img_path)
        img = Image.open(img_path)
        img_arr = np.array(img) / 255.0
        img_arr = np.reshape(img_arr, newshape=(784, 1))
        x_train_data.append(img_arr)
        y_tmp = np.zeros(shape=(10))
        y_tmp[int(num)] = 1         # label -> one-hot vector
        y_train_data.append(y_tmp)  # ont-hot vector save to list
        # x_train_data.append(np.array(img))
        # y_train_data.append(int(num))

print(len(x_train_data))

eval_files = []
for i in label_list:
    path_dir = './images/testing/{0}'.format(i)
    file_list = os.listdir(path_dir)
    file_list.sort()
    eval_files.append(file_list)
x_test_data = []
y_test_data = []
for num in label_list:
    for nmbers in eval_files[num]:
        img_path = './images/testing/{0}/{1}'.format(num, numbers)
        img = Image.open(img_path)
        img_arr = np.array(img)
        img_arr = np.reshape(img_arr, newshape=(784, 1))
        x_test_data.append(img_arr)
        y_tmp = np.zeros(shape=(10))
        y_tmp[int(num)] = 1
        y_test_data.append(y_tmp)

print(y_test_data)

x_train_datas = np.reshape(x_train_data, newshape = (-1, 784))
y_train_datas = np.reshape(y_train_data, newshape = (-1, 10))
x_test_datas = np.reshape(x_test_data, newshape = (-1, 784))
y_test_datas = np.reshape(y_test_data, newshape = (-1, 10))

input = tf.keras.Input(shape = (784), name = "Input")
hidden = tf.keras.layers.Dense(512, activation = "relu", name = "Hidden1")(input)
output = tf.keras.layers.Dense(10, activation="softmax", name = "Output")(hidden)

model = tf.keras.Model(inputs = [input], outputs = [output])
opt = tf.keras.optimizers.Adem(learning_rate = 0.001)
model.compile(loss = "cateforical_crossentropy", optimizer = opt, metrics = ['accuracy'])
model.summary()

model.fit(x_train_datas, y_train_datas, epochs = 5, shuffle = True)    # 학습

test_loss, test_acc = model.evaluate(x_test_datas, y_test_datas)        # 평가
print('테스트 정확도: ', test_acc)

history = model.fit(x_train_datas, y_train_datas, epochs = 20, shuffle = True,
                    validation_data = (x_test_datas, y_test_datas))

plt.plot(history.history['loss'], 'b')
plt.plot(history.history['val_accuracy'], 'r')
plt.show()