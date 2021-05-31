import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import isdigit
import pandas as pd
import keras


imgs = ['.\data\\1.jpeg', '.\data\\2.jpeg', '.\data\\3.jpeg',
        '.\data\\4.jpeg', '.\data\\5.jpeg', '.\data\\6.jpeg', '.\data\\test_img.jpg', '.\data\\test_img3.jpg']


map_images = pd.read_csv(
    ".\streamlit_front\emnist-balanced-mapping.txt", header=None)

ascii_map = []
for i in map_images.values:
    ascii_map.append(int(i[0].split()[1]))


model1 = keras.models.load_model('./my_model/emnist_1.h5')  # best
model2 = keras.models.load_model('./my_model/emnist_2.h5')  # 1s are wrong
model3 = keras.models.load_model('./my_model/mnist.h5')


def ans(img):
    pin = []
    # image = cv2.imread(img, cv2.IMREAD_COLOR)
    image = img
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray,  100, 10, 7, 21)
    ret, th = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(
        ctr)[2] + (cv2.boundingRect(ctr)[1] + cv2.boundingRect(ctr)[3]))

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if h < 40:
            continue

        digit = th[y:y + h, x:x + w]

        resized_digit = cv2.resize(digit, (18, 18))

        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)),
                              "constant", constant_values=0)

        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0

        pred1 = model1.predict([digit])[0]
        pred2 = model2.predict([digit])[0]

        final_pred1 = np.argmax(pred1)
        final_pred2 = np.argmax(pred2)

        final_pred = final_pred2 if max(
            pred1)*1.5 < max(pred2) else final_pred1

        pred = int(max(max(pred1), max(pred2)) * 100)

        if isdigit(chr(ascii_map[int(final_pred)])) or chr(ascii_map[int(final_pred)]) in ['O', 'o', 'I', '|', "T"]:

            num = chr(ascii_map[int(final_pred)])
            num = '0' if num in ['O', 'o'] else num
            num = '1' if num in ['|', 'I'] else num
            num = '7' if num in ["T"] else num

            data = num + " " + str(int(pred)) + '%'
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            color = (255, 0, 0)
            thickness = 1

            cv2.putText(image, data, (x, y - 5), font,
                        fontScale, color, thickness)

            pin.append(num)

            # print(data)

        # cv2.imshow('Predictions', image)
        # cv2.waitKey(0)

    return pin[-6::]


if __name__ == "__main__":
    for img in imgs:
        image = cv2.imread(img)
        p = ans(image)
        print(p)
