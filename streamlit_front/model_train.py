import cv2 as cv
import easyocr
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

# DIR = '.\data\img2.jpg'
# DIR = '.\data\img4.jpg'
# DIR = '.\data\tough.jpg'
# DIR = '.\data\crap_image.jpg'
DIR = '.\data\\2.jpeg'


def get_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    if os.path.isdir('./my_model'):
        model = keras.models.load_model('./my_model/')
        print("MODEL LOADED")
    else:
        model.fit(x_train, y_train, batch_size=128,
                  epochs=15, verbose=1, validation_split=0.1)
        model.save('my_model')

    return model


def get_cropped_pincode(img):
    print("getting cropped image")
    reader = easyocr.Reader(['en'], gpu=True)
    result = reader.readtext(img)
    print(result)
    # print(result)
    # img = cv.imread(img)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    lap = cv.Laplacian(gray, cv.CV_64F)
    lap = np.uint8(np.absolute(lap))

    plt.imshow(cv.cvtColor(lap, cv.COLOR_BGR2RGB))

    digit = []   # filtered numbers, probabilities and co-ordinated
    cropped = []  # cropped image
    for i in result:
        num = i[1].replace(" ", "")
        if num.isnumeric() and len(num) == 6:
            digit.append(i)
            cropped.append(img[i[0][0][1]:i[0][2][1], i[0][0][0]:i[0][1][0]])

    return digit, cropped


def evaluate(digit_info, cropped_images):
    print("eval digits")
    if len(cropped_images) == 0:
        return ["FOUND NOTHING"]
    for num in digit_info:
        # cv.rectangle(img, (num[0][0][0], num[0][0][1]),
        #              (num[0][2][0], num[0][2][1]), (0, 255, 0), thickness=2)
        num[1].replace(" ", "")

    plt.imshow(cv.cvtColor(cropped_images[0], cv.COLOR_BGR2RGB))
    return digit_info


def predict_ocr(cropped_images):
    print("predict ocr")
    pred = []
    reader = easyocr.Reader(['en'], gpu=True)
    for i in cropped_images:
        x = -1
        y = -1
        ans = reader.readtext(i, batch_size=1000)

        for j in ans[0][0]:
            x = max(x, j[0])
            y = max(y, j[1])
        interval = x//6

        for l in range(0, x, interval):
            if (l + interval > x):
                break
            # cv.line(i, (l, 0), (l, y), (0, 255, 0), 2)
            pred.append(i[0: y, l:l + interval])

        plt.imshow(cv.cvtColor(i, cv.COLOR_BGR2RGB))

    return pred


def get_pincode(model, inter_pred):
    print("use ocr")
    pincode = []
    gray_pred = []

    fig = plt.figure(figsize=(10, 7))

    p = 1
    for i in inter_pred:
        final = cv.cvtColor(i, cv.COLOR_BGR2GRAY)

        final = final.astype('float32')
        final = 1 - cv.resize(final, dsize=(28, 28),
                              interpolation=cv.INTER_CUBIC)/255
        final = np.reshape(final,  (1, 784))
        for i in range(len(final[0])):
            final[0][i] = 1 if final[0][i] >= 0.5 else 0
        final = np.reshape(final,  (1, 28, 28, 1))
        gray_pred.append(final)

    p = 1
    for i in gray_pred:
        fig.add_subplot(2, 3, p)
        plt.imshow(np.reshape(i, (28, 28, 1)), cmap=plt.cm.binary)
        r = model.predict(i).argmax(axis=-1)
        pincode.append(r[0])
        # print(r[0])
        plt.title("label: {}".format(r[0]))
        p += 1

    print("finished ocr")
    return pincode


if __name__ == "__main__":
    print(DIR)
    model = get_model()
    img = cv.imread(DIR)
    print(os.getcwd())
    info, crop = get_cropped_pincode(img)
    pre_evaluation = evaluate(info, crop)
    pred_test = predict_ocr(crop)
    final_pred = get_pincode(model, pred_test)
    print(final_pred)

plt.show()
