import streamlit as st
#import cv2
from pipeline import model1
from PIL import Image
import numpy as np
#import glob
#import pandas as pd

"""
Upload 2D floorplan image as input.
Select pixel length for calibaration factor calculation.
Perform object detection to detect wall, door, window, floor, column, staircase.
Perform text extraction to obtain dimensions of the elements.
"""

st.title("AI based 2D QTO")
st.subheader('Using Deep Learning and Computer Vision techniques 2D floorplans are analyzed.')
st.header("Upload 2D floorplan for auto takeoff generation")


# upload 2d floor plan image .
upload= st.file_uploader(" Choose 2D floorplan..", type=['png', 'jpg'])
if upload is not None:
    image = Image.open(upload)
    img= np.asarray(image)
    st.image(image, caption='Uploaded 2D Floorplan image.', use_column_width=True)
    df, count = model1(image, img)
    st.header("Measurement Sheet")
    st.dataframe(df)

# Drop down to select floorplan element for all measurement.
    option= 'None'
    st.header('Construction elements Measurements')
    option = st.selectbox( 'Select Element',
                        ('None','wall','column','staircase','door','window','floor')
                        )
    if option == 'None':
        pass
    else:
        df1 = df[df["Label"]== option]
        st.dataframe(df1)

# Select parameters to be measured.
    st.header('Select measurement parameters')
    option1= 'None'
    option1 = st.selectbox( 'Select parameter to be measured',
                        ('None','area','perimeter','width','height')
                        )
    if option1 == 'None':
        pass
    if option1 == 'area':
        dfa= df[['Label', 'area_px', 'Actual_area']]
        st.dataframe(dfa)
    if option1 == 'perimeter':
        dfa= df[['Label', 'perimeter_px', 'Actual_perimeter']]
        st.dataframe(dfa)
    if option1 == 'height':
        dfa= df[['Label', 'height_pixelx', 'Actual_height']]
        st.dataframe(dfa)
    if option1 == 'width':
        dfa= df[['Label', 'width_pixel', 'Actual_width']]
        st.dataframe(dfa)

# Display number of each elements detected.
    st.header('Count of elements')
    st.dataframe(count)
