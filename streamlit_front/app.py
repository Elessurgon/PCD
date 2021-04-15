import cv2 as cv
from model_train import *
from numpy import asarray
import streamlit as st
import pandas as pd
from PIL import Image


st.write("""
Testing pincode app
""")

img = st.file_uploader("Choose an image...", type="jpg")

if img is not None:
    print(type(img))
    image = Image.open(img)
    print(type(image))
    image = asarray(image)
    print(type(image))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    model = get_model()
    info, crop = get_cropped_pincode(image)
    pre_evaluation = evaluate(info, crop)
    pred_test = predict_ocr(crop)
    final_pred = get_pincode(model, pred_test)
    test = ""
    if len(final_pred) == 0:
        st.warning("Sorry, found nothing")
    else:
        for i in final_pred:
            test += str(i)
        st.success(test)
