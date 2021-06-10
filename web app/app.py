from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
import io
import re
import math

img_size = 224

app = Flask(__name__) 

model = load_model('model/predict-covid-19.h5')


def preprocess(img):
	img = np.array(img)
	if(img.ndim == 3):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	else:
		gray = img

	resized = cv2.resize(gray, (img_size, img_size))
	reshaped = resized.reshape(1, img_size, img_size, 1)
	return reshaped

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO = io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image = Image.open(dataBytesIO)

	test_image = preprocess(image)

	prediction = model.predict(test_image)

	covid_prob = round(prediction[0][0]*100, 3)
	normal_prob = round(prediction[0][1]*100, 3)
	pneumonia_prob = round(prediction[0][2]*100, 3)

	print(covid_prob, normal_prob, pneumonia_prob)
	print(prediction)
	result1 = f"{covid_prob}%"
	result2 = f"{normal_prob}%"
	result3 = f"{pneumonia_prob}%"
	response = {'prediction': {'result1': result1, 'result2': result2, 'result3': result3}}

	return jsonify(response)

if __name__ == '__main__':
	app.run(debug=True)

