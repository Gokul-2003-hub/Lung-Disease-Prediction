import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import numpy as np
model=load_model("E:/lungs project/model file.h5")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
st.title("Lung Disease Prediction")
upload_file=st.file_uploader("upload")
button=st.button("Predict")
if button==True:
    img=image.load_img(upload_file,target_size=(150,150,1),color_mode='grayscale')
    image_arr=image.img_to_array(img)
    img_array=np.expand_dims(image_arr,axis=0)
    source_file=img_array/255.0
    predictions=model.predict(source_file)
    if predictions < 0.5:
      st.write("negative")
    else:
      st.error("positive")
 

