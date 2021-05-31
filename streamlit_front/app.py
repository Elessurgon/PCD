from ocr import *
import csv_account
import cv2 as cv
from numpy import asarray
import streamlit as st
import pandas as pd
from PIL import Image


st.write("""
Testing pincode app
""")

img = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if img is not None:
    # print(type(img))
    image = Image.open(img)
    # print(type(image))
    image = asarray(image)
    # print(type(image))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    final_pred = ans(image)
    test = ""
    if len(final_pred) == 0:
        st.warning("Sorry, found nothing")
    else:
        for i in final_pred:
            test += str(i)
        csv_account.add(test)
        st.success(test)
