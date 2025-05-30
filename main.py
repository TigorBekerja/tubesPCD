import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests

from operation import modify_image_brightness
from operation import modify_image_contrast
from operation import modify_image_invert

if "default_image" not in st.session_state:
    st.session_state.default_image = Image.open(
        requests.get("https://picsum.photos/200/120", stream=True).raw
    ).convert("RGB")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    a = image.size
else:
    image = st.session_state.default_image.copy()
    a = "tidak ada gambar"



edges = cv2.Canny(np.array(image), 100, 200)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Detected edges", "Original", "kecerahan", "contrast", "invert image"])

with tab1:
    tab1.image(image, use_container_width=True)
    st.markdown(a)

with tab2:
    tab2.image(edges, use_container_width=True)

with tab3:
    number = st.slider("tambahkan atau kurangi kecerahan", -255, 255, 0)
    number = int(number)
    modified_image = modify_image_brightness(image.copy(), number)
    tab3.image(modified_image, use_container_width=True)
    st.markdown(image.load()[0, 0])
    st.markdown(modified_image.load()[0, 0])

with tab4:
    number = st.slider("tambahkan atau kurangi tingkat kontrast (dalam persen)", -100, 100, 0)
    number =  int(number)
    modified_image = modify_image_contrast(image.copy(), number)
    tab4.image(modified_image, use_container_width = True)

with tab5:
    modified_image = modify_image_invert(image.copy())
    tab5.image(modified_image, use_container_width = True)