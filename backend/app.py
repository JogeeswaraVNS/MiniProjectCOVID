import tensorflow as tf
from flask import Flask,jsonify,request,Response
from flask_cors import CORS
import os
import json



app=Flask(__name__)

CORS(app)





UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)






@app.route('/predict',methods=['POST'])    
def predict_disease():
    return "Hell"


if __name__ == '__main__':
    app.run(debug=True)