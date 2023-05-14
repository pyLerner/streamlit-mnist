#!/usr/bin/env python3
import os
import re
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.saving import load_model


def img_preparing(img_file):
    
    img = Image.open(img_file)
    img = img.resize((28, 28)).convert('L')
    img = np.array(img).astype('float')
    img = (255 - img)
    
    img[np.where(img >= 85)] = 255
    img[np.where(img < 85)] = 0
    
    return img.reshape(-1, 28, 28, 1)

st.subheader('Домашнее задание уровня Pro')
st.title('Модель распознавания рукописных цифр')
st.subheader('"запилил" Вадим Лернер')

model = load_model('model_fmr_all.h5')

upload_file = st.file_uploader(
    'Здесь можно загрузить что-то свое', 
    type=['png', 'jpg', 'jpeg']
    )

file_list = []

digits = 'digits'
list_dir = os.listdir(digits)

for file in list_dir:
    r = re.match(r'\d\d\d.png', file)
    if r:
        file_list.append(r[0])

file_list.sort()

side_file =  st.sidebar.selectbox(
    'Выбор изображений',
    file_list
    )

if not upload_file is None:
    img_file = upload_file
else:
    img_file = f'{digits}/{side_file}'
    
col1, col2 = st.columns([1, 4])

with col1:
    st.image(img_file, width=120)

img = img_preparing(img_file)
pred = model.predict(img)
vanga = np.argmax(pred)

with col2:
    st.text('Выходные нейроны')
    st.write(pred)
    st.subheader(f'Ответ сети: цифра {vanga}')
    

show_code = st.sidebar.checkbox('Показать код:')

if show_code:
    
    st.text('Мой скромный код')
    PATH = os.path.abspath(__file__)
    
    with open(PATH, 'r') as file:
        file = file.read()
        st.code(file, language='python')
        