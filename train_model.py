# Импорты
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.applications import vgg16 
from PIL import Image
from keras import backend as K
from keras.layers import Activation, Input, Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

import gc
import time

study_epochs = 20
batch_size=128*2
total_sample_size = 1_000
image_shape_1 = 90
image_shape_2 = 300
image_path = "Input"

indCount = [388, 157, 268, 1007, 54, 38, 100, 594, 551, 328, 651, 1308, 1186, 359, 601, 188, 715, 1263, 140, 119, 481, 1040, 106, 308, 589, 269, 184, 160, 396, 715, 179, 454, 551, 108, 463, 212, 117, 304, 944, 586, 1071, 674, 263, 256, 115, 128, 205, 260, 257, 112, 427, 251, 88, 178, 79, 191, 40, 354, 285, 142, 251, 161, 23, 73, 356, 289, 200, 168, 48, 148, 196, 33, 222, 597, 221, 342, 44, 257, 369, 113, 77, 89, 174, 100, 452, 360, 132, 71, 253, 116, 186, 1583, 42, 72, 566, 119, 437, 264, 464, 137, 148, 134]

def wait_and_collect(wait_t, collect_t):
  for i in range(collect_t):
    gc.collect()
    time.sleep(wait_t)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def create_base_network(input_shape):
    seq = Sequential()
    nb_filter = [6, 12]
    kernel_size = 3

    # Свёрточный слой 1
    seq.add(Conv2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shape, padding="valid", data_format="channels_first"))
    seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(.25))

    # Свёрточный слой 2
    seq.add(Conv2D(nb_filter[1], kernel_size, kernel_size, padding="valid", data_format="channels_first"))
    seq.add(Activation("relu"))
    #seq.add(MaxPooling2D(pool_size=(2, 2), data_format="channels_first"))
    seq.add(Dropout(.25))

    # Выходной слой
    seq.add(Flatten())    
    seq.add(Dense(128, activation="relu"))
    seq.add(Dropout(.1))
    seq.add(Dense(50, activation="relu"))
    
    return seq

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def read_image(filename, byteorder=">"):
    mat = cv2.imread(filename)     
    gray_img = cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)   
    return np.frombuffer(gray_img, dtype="u1", count=image_shape_1*image_shape_2).reshape(image_shape_1, image_shape_2)

def prepareImg(imgName, height=60, width=200):
    """Возвращает изображение в формате, подходящем для идентификации в нашей сети.

    Args:
        img (string): адрес изображения.
        height (int): итоговая высота изображения.
        width (int): итоговая ширина изображения
    """
    dim = (width, height)
    img = cv2.imread(imgName)
    resImg = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    grayResImg = cv2.cvtColor(resImg, cv2.COLOR_BGR2GRAY)
    fImg = np.frombuffer(grayResImg, dtype="u1", count=image_shape_1*image_shape_2).reshape(image_shape_1, image_shape_2)
    fImg = fImg[::size, ::size]

    #return grayResImg
    return fImg

def print_res(pred):
    if (pred < .5):
        print("+")
    else:
        print("-")

size = 2

def get_data(size, total_sample_size):
    image = read_image(image_path + "/" + str(1) + "/" + "1_0" + ".jpg", "rw+")
    image = image[::size, ::size]
    
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_genuine = np.zeros([total_sample_size, 1])

    for i in range(40):
        for j in range(int(total_sample_size/40)):
            ind1 = 0
            ind2 = 0

            while ind1 == ind2:
                ind1 = np.random.randint(indCount[i])
                ind2 = np.random.randint(indCount[i])

            img1 = read_image(image_path + "/" + str(i+1) + "/" + str(i+1)+"_"+str(ind1) + ".jpg", "rw+")
            img2 = read_image(image_path + "/" + str(i+1) + "/" + str(i+1)+"_"+str(ind2) + ".jpg", "rw+")
            
            
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]
            
            
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2

            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])

    for i in range(int(total_sample_size/10)):
        for j in range(10):
            while True:
                ind1 = np.random.randint(1, 102)
                ind2 = np.random.randint(1, 102)
                if ind1 != ind2:
                    break

            img1 = read_image(image_path + "/" + str(ind1) + "/" + str(ind1) + "_" + str(np.random.randint(indCount[ind1-1])) + ".jpg", "rw+")
            img2 = read_image(image_path + "/" + str(ind2) + "/" + str(ind2) + "_" + str(np.random.randint(indCount[ind2-1])) + ".jpg", "rw+")
            
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2

            y_imposite[count] = 0
            count += 1

    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0)/255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y

X, Y = get_data(size, total_sample_size)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = .25)
input_dim = x_train.shape[2:]

wait_and_collect(1, 1)

base_network = create_base_network(input_dim)
img_a = Input(shape=input_dim)
img_b = Input(shape=input_dim)

processed_a = base_network(img_a)
processed_b = base_network(img_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])
model = Model(inputs=[img_a, img_b], outputs=distance)

wait_and_collect(1, 1)

rms = RMSprop()
model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])
history = model.fit([x_train[:, 0], x_train[:, 1]], y_train, validation_split=.25, batch_size=batch_size, verbose=2, epochs=study_epochs)

wait_and_collect(1, 1)

y_pred = model.predict([x_train[:, 0], x_train[:, 1]])
train_acc = accuracy(y_train, y_pred)
y_pred = model.predict([x_test[:, 0], x_test[:, 1]])
test_acc = accuracy(y_test, y_pred)

print("train_acc:", train_acc)
print("test_acc:", test_acc)

tf.keras.models.save_model(model, f"model_{train_acc}-{test_acc}", save_format="tf", overwrite=True)