import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorflow import keras
# from keras_preprocessing.image import load_img
from keras.utils import load_img
# from keras.preprocessing.image import img_to_array
from keras.utils import img_to_array
from keras.models import load_model

filepath = 'D:\Cotton Disease Detection\cotton_infection_dir_ready\cotton_disease.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

cotton_plant = cv2.imread('D:\Cotton Disease Detection\cotton_infection_dir_ready\Dataset\curl315.jpg')
test_image = cv2.resize(cotton_plant, (128,128)) # load image 
  
test_image = img_to_array(test_image)/255 # convert image to np array and normalize
test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
result = model.predict(test_image) # predict diseased plant or not
  
pred = np.argmax(result, axis=1)
print(pred)
if pred==0:
    print( "Cotton - Bacterial Blight Disease")
       
elif pred==1:
    print("Cotton - Curl Virus Disease")
        
elif pred==2:
    print("Cotton - Healthy and Fresh")
        
elif pred==3:
    print("Cotton - Fussarium Wilt Disease")
       
