from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2
import imutils
import tensorflow as tf
import matplotlib.pyplot as plt
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/cnn_basico.h5'
print (MODEL_PATH)
# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(file_path, model):
    img = tf.keras.preprocessing.image.load_img(file_path)
    img = cv2.imread(file_path)

    #PREPROCESO!!!
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (600,400))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    print ('******Realiza Resize de la Imagen y las Convierte a Grises******')
    
    img_blurred = cv2.GaussianBlur(img_gray, ksize=(3, 3), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(
    		img_blurred, 
            maxValue=255.0,
    		adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    		thresholdType=cv2.THRESH_BINARY_INV, 
    		blockSize=19, 
    		C=9)
    
    print('****Realiza suavizado y binarizacion de la imagen******')
        
    keypoints = cv2.findContours(img_thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    
    location=None
    for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

    print('****Localiza las coordenadas del contorno de la placa*****')
    print (location)
    
    mask = np.zeros(img_gray.shape, np.uint8)
    new_img = cv2.drawContours(mask, [location], 0,255,-1)
    new_img = cv2.bitwise_and(img, img, mask=mask)

    (x,y) = np.where (mask==255)
    (x1,y1) = (np.min(x), np.min(y))
    (x2,y2) = (np.max(x), np.max(y))
    cropped_img = img[x1:x2+1, y1:y2+1]

    char_test = segment_characters(cropped_img)
    
    def fix_dimension(img): 
        new_img = np.zeros((64,64,3))
        for i  in range(3):
            new_img[:,:,i] = img
        return new_img
    
    dic = {}
    caracteres = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    
    for i,c in enumerate (caracteres):
        dic [i] = c
    
    output = []
    for  i,ch in enumerate(char_test): #iteraccion sobre los cada uno de los caracteres
         img_ = cv2.resize(ch, (64,64), interpolation=cv2.INTER_AREA)
         img= fix_dimension(img_)
         img = img.reshape(1,64,64,3) #Prepera la imagen para ingresar al modelo

         preds = model.predict(img)[0] # Realiza la clasificación
         preds = np.argmax(preds,axis=0)
         
         caracter = dic [preds]
         output.append(caracter)
    #preds = model.predict(img)
    return output
    
#ENCUENTRA CARACTER DE IMAGEN RECORTADA!!!!
def segment_characters(image) :

    print('*****Preprocesa la imagen recortada de la placa*****')
    # Preprocesa la imagen cortada de la placa con binarización
    img = cv2.resize(image, (333, 75))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Transformación morfologica para cerrar el shape de los digitos
    img_binary_lp = cv2.dilate(img_binary_lp, (2,2))
    kernel = np.ones((2,2), np.uint8)
    img_binary_lp = cv2.morphologyEx(img_binary_lp, cv2.MORPH_CLOSE, kernel)

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Genera bordes blancos
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estima el tamaño de los contornos de caracteres de las placas
    dimensions = [LP_WIDTH/5, LP_WIDTH/1, LP_HEIGHT/17, 2*LP_HEIGHT/3]
   
    # Obtiene los cortornos de la imagen de la placa
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

def find_contours(dimensions, img) :
    
    print('*****Identifica los contornos de cada caracter*****')
 
    # Encontrar todos los contornos de la imagen
    cntrs = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Recuperar las dimensiones
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]

    M = []
    cX = []
    cY = []
   
    # Verifica los 15 contornos mas grandes para identificar el contorno de los caracteres
    cntrs = imutils.grab_contours(cntrs)
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = img

    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :

        # Detecta el contorno en coordenadas binarias y devuelve las coordenadas del rectangulo
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)

        # comprobar las dimensiones del contorno para filtrar los caracteres por tamaño del contorno
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :

            x_cntr_list.append(intX) # Almacenas las coordenadas del contorno

            # Identificación de los centroides

            M = cv2.moments(cntr)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #print(cX)
            
            char_copy = np.zeros((44,24))
            
            #Extrae cada caracter usando coordenadas del rectangulo
            
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
                      
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (0,255,0), 2)
           

            # Resultado formateado para clasificacion
           #char = cv2.subtract(255, char)

            # Cambiar el tamaño de la imagen a 24x44 con borde negro
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) # Lista que almacena la imagen binaria del caracter
            
    # Devuelve caracteres en orden ascendente con respecto a la coordenada x
            
    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# Almacena las imagenes del caracter de acuerdo a su indice
    img_res = np.array(img_res_copy)

    return img_res

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        # Hacer la prediccion
        output = model_predict(file_path, model)
        
        result = "".join([str(n) for n in output])

        return result
    
    return None

if __name__ == '__main__':
    app.run(debug=True)
