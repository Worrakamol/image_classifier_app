# -- coding: utf-8 --
"""
Created on Sun Apr 20 13:52:44 2025

@author: LAB
"""

import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# โหลดโมเดล MobileNetV2 ที่ฝึกจาก ImageNet
model = MobileNetV2(weights="imagenet")

# หัวข้อ
st.title("🖼️ Image Classification with MobileNetV2")
st.caption("by Worrakamol Nantipatpanya")

# อัปโหลดรูป
upload_file = st.file_uploader("📤 Upload image:", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    # แสดงรูป
    img = Image.open(upload_file)
    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    # แปลงภาพให้เป็น RGB หากยังไม่ใช่
    if img.mode != "RGB":
        img = img.convert("RGB")

    # เตรียมภาพสำหรับโมเดล
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # ทำนายผล
    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]

    # แสดงผลลัพธ์
    st.subheader("🔍 Predictions:")
    for i, pred in enumerate(top_preds):
        st.write(f"{i+1}. **{pred[1]}** — {round(pred[2]*100, 2)}%")

