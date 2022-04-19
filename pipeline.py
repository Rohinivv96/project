import cv2
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy.spatial import distance as dist
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import imutils
import glob
import os
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import sys


import pytesseract
import urllib
import re
import difflib
import streamlit as st

pytesseract.pytesseract.tesseract_cmd ='./model/Tesseract-OCR/tesseract.exe'

########################################################################################
def pixel_measure(px,Actual_unit, Actual_dimensions):
    detected_px_len= int(Actual_dimensions)/px
    #print(detected_px_len, Actual_unit+"/px")
    st.write('Calibration factor is ',detected_px_len, Actual_unit+"/px")
    print(detected_px_len)
   
    return detected_px_len
    

##############################################################################################
emptyDict = {}

def extract(PATH):
    comb=[]
    result1 = []
    custom_config = r'-l eng --oem 3 --psm 4'
    data=(pytesseract.image_to_string(PATH))
    #print(data)
    text = data.split('\n')
    #text = pd.DataFrame([x.split(';') for x in data.split('\n')])
    #new = []
    #for i in text[0]:
    #    new.append(i)
    #print(new)
    possibilities = ["Toilet","Bedroom", "Hall", "Kitchen", "Balcony", "Dining","Sitout","SITOUT","HALL","KITCHEN","BEDROOM","HALL","LOBBY","DRESSING ROOM"]
    n = 3
    cutoff = 0.5

    for w in text:
        if difflib.get_close_matches(w, possibilities, n, cutoff) and w!="":
            i = text.index(w)
            if text[i+1] !='':
                emptyDict[w] = text[i+1]

    return emptyDict
    #print(tabulate(result1, headers = 'keys', tablefmt = 'psql'))
        
        
        



#################################################################################################
def detect(img,factor,OUT_PATH):
    frame=[]
    num=0
    # To read darknet module using config file and yolov4 weights
    net = cv2.dnn.readNet("./model/custom-yolov4-detector_8000.weights", 
                          "./model/custom-yolov4-detector.cfg")

    #Reading class names
    classes = ['wall','column','staircase','door','window','floor']

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i  - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    #img= cv2.imread(r"D:\Winjit_training\floorplan\floorplan.png")
    height, width,_ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)

    # give images input to darknet module
    net.setInput(blob)

    #  Get the output layers of model
    out_names= net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(out_names)

    # Generating detection in the form of bounding box, confidence and class id
    boxes=[]
    confidences=[]
    class_ids=[]



    for out in layerOutputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                centerX= int(detection[0]* width)
                centerY= int(detection[1]* height)
                w= int(detection[2]* width)
                h= int(detection[3]* height)
                x = int(centerX - (w/ 2))
                y = int(centerY - (h/ 2))

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes= cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    indexes= np.array(indexes)
    font= cv2.FONT_HERSHEY_PLAIN
    colors= np.random.uniform(0, 255, size=(len(boxes),3))

    for i in indexes.flatten():
        x, y, w, h= boxes[i]
        label= str(classes[class_ids[i]])
        color= colors[i]
        confidence= str(round(confidences[i],2))
        cv2.rectangle(img, (x, y), (x+w,y+h),color, 2)
        obj= img[y: y+h, x: x+w]
        cv2.imwrite(OUT_PATH+'\crop{0}.jpg'.format(i), obj)
        frame.append(area_1(obj, factor, label))
   
    cv2.imwrite('./result/object_detection.jpg',img)
    st.image(img)
    return frame
   
#########################################################################################################
def area_1(crop, factor,label):
    #crop= cv2.imread(img_path)
   # heigh = int(input("enter your height: "))
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    cnts = cv2.findContours(gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    

  # Sort contours from left to right as leftmost contour is reference object
    (cnts, _) = contours.sort_contours(cnts)

    for cnt in cnts:
        
        # area calculation
        area= cv2.contourArea(cnt)
        area_px= area
        area_mm=area_px*(factor*factor)
        

        if area >100:
            cv2.drawContours(crop, cnt, -1, (0,255,0), 1)
            
            # perimeter calculation
            peri= cv2.arcLength(cnt, True)
            perimeter_px=peri
            perimeter_mm=peri*factor
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
            
            # width and height calculation
            box = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            (tl, tr, br, bl) = box
            width_pixel=euclidean(tl, tr)
            width_mm=euclidean(tl, tr)*factor
            height_pixel=euclidean(tr, br)
            height_mm=euclidean(tr, br)*factor
            
            return label,area_px,area_mm,perimeter_px, perimeter_mm,width_pixel,width_mm,height_mm,height_pixel

#########################################################################################################



def model1(image, pic):
    """
       Model combining calibration factor calcultion, object detection, text extraction. 

        Parameters
        ----------
        image: 
            image 
        pic:
            image converted to array
        

        Returns
        -------
        str
            done or incomplete
        """

    img= image
    img1 = pic
    
    # path to save cropped images
    OUT_PATH=r'./result/detect_out'
    
    st.header('Calibration factor calculation.')
    st.subheader('Select number of pixels from image and enter its actual length.')
    
    # Canvas to draw line for pixel length measurement
    width, height= img.size
    canvas_result = st_canvas(drawing_mode= "line",
                               background_image=img,
                               height= height,
                               width= width,
                               stroke_width= 5,
                                stroke_color='red'
                              )

    global point
    point =0
    if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        if len(canvas_result.json_data["objects"]) == 1:
            objects = pd.json_normalize(canvas_result.json_data["objects"])
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            st.dataframe(objects)
            if objects['width'][0]> objects['height'][0]:
                point= objects['width'][0]
            else:
                point= objects['height'][0]
            st.write('Selected Pixel length', point )

    #select unit and actual length
    Actual_unit = st.radio( "Select your unit(m, mm, cm, inch): ",
                            ('m','mm', 'cm', 'inch'))
    Actual_dimensions = st.number_input('Insert actual length')

    if Actual_dimensions != 0.00 and point != 0:
        factor= pixel_measure(point,  Actual_unit, Actual_dimensions )
      

        # perform column detection.
        st.header('Output of the object detection model.')
        frame= detect(img1 ,factor,OUT_PATH)
        
        
        # text extraction
        for name in glob.glob(OUT_PATH+"/*"): 
            text_out=extract(str(name))
            
        
        new_dict={}
        for i, s in text_out.items():
            dim=re.findall('\d*\.?\d+',s)
            new_dict[i]=dim
        
        
        result1= pd.DataFrame.from_dict(new_dict, orient='index')
        result1.to_csv("./result/text_extract.csv")
        result1.columns = ['length','width']
        st.header('Extracted Measurements')
        st.subheader('Using OCR floorplan dimensions are extracted.')
        st.json(new_dict)
        st.dataframe(result1)

        # Adding download button to download csv
        csv1= result1.to_csv().encode('utf-8')

        st.download_button(
        label="Download data as CSV",
        data=csv1,
        file_name='extract.csv',
        mime='text/csv',
        )
    

        #Calculate area and save in csv format
        df= pd.DataFrame(frame, columns=['Label','area_px','Actual_area','perimeter_px',
                                'Actual_perimeter','width_pixel','Actual_width','Actual_height','height_pixel'])

        df = df.sort_values("Label")
        df.to_csv('./result/dimensions_construction_elements.csv')

        # Adding download button to download csv
        csv= df.to_csv().encode('utf-8')

        st.download_button(
        label="Download CSV",
        data=csv,
        file_name='elements.csv',
        mime='text/csv',
        )


        count = df["Label"].value_counts()
        count = pd.DataFrame(count)
        count.to_csv("./result/Final_count.csv") 
        return 'done'
    else:
        return 'incomplete' 

    
   
