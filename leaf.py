#Import necessary libraries
from flask import Flask, render_template, request

from tensorflow import keras
import numpy as np
import os

from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import load_model

filepath = 'D:\Cotton Disease Detection\cotton_infection_dir_ready\cotton_disease.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_cotton_diease(cotton_plant):
  test_image = load_img(cotton_plant, target_size = (128, 128)) # load image 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image) # predict diseased palnt or not
  print('@@ Raw result = ', result)
  
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
    

# Create flask instance
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fetch input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join(r"D:\Cotton Disease Detection\cotton_infection_dir_ready\Dataset\test\bacterial_blight",filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page  = pred_cotton_diease(cotton_plant=file_path)      
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.debug = True
    app.run(threaded=False,port=8080) 
    
    
