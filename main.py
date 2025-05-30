import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from streamlit_extras.stateful_button import button

from operation import modify_image_brightness
from operation import modify_image_contrast
from operation import modify_image_invert
from operation import edge_detection
from operation import erode
from operation import dilation

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
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Original", "Detected edges", "kecerahan", 
    "contrast", "invert image", "Erode",
    "Dilation", "Opening", "Closing"])

last_edge_type = 2  # Default to Sobel

with tab1:
    tab1.image(image, use_container_width=True)
    st.markdown(a)

with tab2:
    type = st.selectbox("Pilih tipe deteksi tepi", ["Canny", "Sobel", "Prewitt"])
    if type == "Canny":
        modified_image = edge_detection(image.copy().convert("L"), type=1)
        last_edge_type = 1
    elif type == "Sobel":
        modified_image = edge_detection(image.copy().convert("L"), type=2)
        last_edge_type = 2
    elif type == "Prewitt":
        modified_image = edge_detection(image.copy().convert("L"), type=3)
        last_edge_type = 3
    
    tab2.image(modified_image, use_container_width=True)
    modified_image = modified_image.convert("RGB")
    img_arr = np.array(modified_image).astype(np.uint8)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    st.download_button(
        label="Download Detected Edges",
        data=cv2.imencode('.png', img_arr)[1].tobytes(),
        file_name='detected_edges.png',
        mime='image/png'
    )

with tab3:
    number = st.slider("tambahkan atau kurangi kecerahan", -255, 255, 0)
    number = int(number)
    modified_image = modify_image_brightness(image.copy(), number)
    tab3.image(modified_image, use_container_width=True)
    st.markdown(image.load()[0, 0])
    st.markdown(modified_image.load()[0, 0])
    modified_image = modified_image.convert("RGB")
    img_arr = np.array(modified_image).astype(np.uint8)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    st.download_button(
        label="Download Brightness Modified Image",
        data=cv2.imencode('.png', img_arr)[1].tobytes(),
        file_name='brightness_modified_image.png',
        mime='image/png'
    )

with tab4:
    number = st.slider("tambahkan atau kurangi tingkat kontrast (dalam persen)", -100, 100, 0)
    number =  int(number)
    modified_image = modify_image_contrast(image.copy(), number)
    tab4.image(modified_image, use_container_width = True)
    modified_image = modified_image.convert("RGB")
    img_arr = np.array(modified_image).astype(np.uint8)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    st.download_button(
        label="Download Contrast Modified Image",
        data=cv2.imencode('.png', img_arr)[1].tobytes(),
        file_name='contrast_modified_image.png',
        mime='image/png'
    )

with tab5:
    modified_image = modify_image_invert(image.copy())
    tab5.image(modified_image, use_container_width = True)
    modified_image = modified_image.convert("RGB")
    img_arr = np.array(modified_image).astype(np.uint8)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    st.download_button(
        label="Download Inverted Image",
        data= cv2.imencode('.png', img_arr)[1].tobytes(),
        file_name='inverted_image.png',
        mime='image/png'
    )
    
with tab6:
    modified_image = edge_detection(image.copy().convert("L"), type=last_edge_type)
    
    kernelButton = [st.columns(3), st.columns(3), st.columns(3)]
    
    kernel = [[0,0,0],
              [0,0,0],
              [0,0,0]]
    
    for i in range(3):
        for index, col in enumerate(kernelButton[i]):
            with col:
                btn = button(f"{i}{index}", key=f"erode_{i}{index}", use_container_width=True)
                if btn:
                    kernel[i][index] = 1 if kernel[i][index] == 0 else 0
    
    modified_image = erode(modified_image, kernel)
    tab6.image(modified_image, use_container_width=True)
    modified_image = modified_image.convert("RGB")
    img_arr = np.array(modified_image).astype(np.uint8)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    st.download_button(
        label="Download Eroded Image",
        data=cv2.imencode('.png', img_arr)[1].tobytes(),
        file_name='eroded_image.png',
        mime='image/png'
    )

with tab7:
    modified_image = edge_detection(image.copy().convert("L"), type=last_edge_type)
    
    kernelButton = [st.columns(3), st.columns(3), st.columns(3)]
    
    kernel = [[0,0,0],
              [0,0,0],
              [0,0,0]]
    
    for i in range(3):
        for index, col in enumerate(kernelButton[i]):
            with col:
                btn = button(f"{i}{index}", key=f"dialate_{i}{index}", use_container_width=True)
                if btn:
                    kernel[i][index] = 1 if kernel[i][index] == 0 else 0
    
    modified_image = dilation(modified_image, np.array(kernel))
    tab7.image(modified_image, use_container_width=True)
    modified_image = modified_image.convert("RGB")
    img_arr = np.array(modified_image).astype(np.uint8)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    st.download_button(
        label="Download Dilated Image",
        data=cv2.imencode('.png', img_arr)[1].tobytes(),
        file_name='dilated_image.png',
        mime='image/png'
    )

with tab8:
    modified_image = edge_detection(image.copy().convert("L"), type=last_edge_type)
    
    kernelButton = [st.columns(3), st.columns(3), st.columns(3)]
    
    kernel = [[0,0,0],
              [0,0,0],
              [0,0,0]]
    
    for i in range(3):
        for index, col in enumerate(kernelButton[i]):
            with col:
                btn = button(f"{i}{index}", key=f"open_{i}{index}", use_container_width=True)
                if btn:
                    kernel[i][index] = 1 if kernel[i][index] == 0 else 0
    
    modified_image = erode(modified_image, np.array(kernel))
    modified_image = dilation(modified_image, np.array(kernel))
    tab8.image(modified_image, use_container_width=True)
    modified_image = modified_image.convert("RGB")
    img_arr = np.array(modified_image).astype(np.uint8)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    st.download_button(
        label="Download Opened Image",
        data=cv2.imencode('.png', img_arr)[1].tobytes(),
        file_name='opened_image.png',
        mime='image/png'
    )

with tab9:
    modified_image = edge_detection(image.copy().convert("L"), type=last_edge_type)
    
    kernelButton = [st.columns(3), st.columns(3), st.columns(3)]
    
    kernel = [[0,0,0],
              [0,0,0],
              [0,0,0]]
    
    for i in range(3):
        for index, col in enumerate(kernelButton[i]):
            with col:
                btn = button(f"{i}{index}", key=f"close_{i}{index}", use_container_width=True)
                if btn:
                    kernel[i][index] = 1 if kernel[i][index] == 0 else 0
    
    modified_image = dilation(modified_image, np.array(kernel))
    modified_image = erode(modified_image, np.array(kernel))
    tab9.image(modified_image, use_container_width=True)
    modified_image = modified_image.convert("RGB")
    img_arr = np.array(modified_image).astype(np.uint8)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    st.download_button(
        label="Download Closed Image",
        data=cv2.imencode('.png', img_arr)[1].tobytes(),
        file_name='closed_image.png',
        mime='image/png'
    )