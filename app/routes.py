from flask import render_template, request
from app import app
import cv2
import numpy as np
import pickle

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    scaler, pca, classifier = pickle.load(model_file)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    equalized = cv2.equalizeHist(resized)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    return blurred

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            preprocessed_img = preprocess_image(img).flatten().reshape(1, -1)
            scaled_img = scaler.transform(preprocessed_img)
            pca_img = pca.transform(scaled_img)
            prediction = classifier.predict(pca_img)
            category = ['Buildings', 'Forests', 'Glaciers', 'Mountains', 'Seas', 'Streets'][prediction[0]]
            return render_template('result.html', prediction=category)
    return render_template('index.html')
