import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests

from operation import modify_image_brightness
from operation import modify_image_contrast
from operation import modify_image_invert
from operation import sobel_kernel
from operation import erosion
from operation import dilation
from operation import opening
from operation import closing

if "default_image" not in st.session_state:
    st.session_state.default_image = Image.open(
        requests.get("https://drive.usercontent.google.com/u/0/uc?id=1VHhsod8v4o6LUItSXp7SdfJXNK33Tr1G&export=download", stream=True).raw
    ).convert("RGB")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    a = image.size
else:
    image = st.session_state.default_image.copy()
    a = "tidak ada gambar"



edges = cv2.Canny(np.array(image), 100, 200)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Original", "kecerahan", "contrast", "invert image", "sobel"])

with tab1:
    tab1.image(image, use_container_width=True)
    st.markdown(a)


with tab2:
    number = st.slider("tambahkan atau kurangi kecerahan", -255, 255, 0)
    number = int(number)
    modified_image = modify_image_brightness(image.copy(), number)
    tab2.image(modified_image, use_container_width=True)
    st.markdown(image.load()[0, 0])
    st.markdown(modified_image.load()[0, 0])

with tab3:
    number = st.slider("tambahkan atau kurangi tingkat kontrast (dalam persen)", -100, 100, 0)
    number =  int(number)
    modified_image = modify_image_contrast(image.copy(), number)
    tab3.image(modified_image, use_container_width = True)

with tab4:
    modified_image = modify_image_invert(image.copy())
    tab4.image(modified_image, use_container_width = True)

with tab5:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("sobel"):
            tab5.image(sobel_kernel(image.copy()), use_container_width = True)
    with col2:
        if st.button("erosion"):
            tab5.image(erosion(image.copy()), use_container_width = True)
    with col3:
        if st.button("dilation"):
            tab5.image(dilation(image.copy()), use_container_width = True)
    with col4:
        if st.button("opening"):
            tab5.image(opening(image.copy()), use_container_width = True)
    with col5:
        if st.button("closing"):
            tab5.image(closing(image.copy()), use_container_width = True)