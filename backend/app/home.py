from flask import Flask, request
from flask_cors import CORS
import numpy as np
import cv2
from processImage import processImage;
from errors import *
import os
import pandas as pd
from io import BytesIO

app = Flask(__name__)
CORS(app)

app.register_error_handler(BadRequest, handle_error)
app.register_error_handler(PageNotFound, handle_error)
app.register_error_handler(InternalServerError, handle_error)
app.register_error_handler(UnsupportedFileFormat, handle_error)

@app.route('/')
def welcome():
    raise PageNotFound("Oops, looks like the page you have requested is not available.")

@app.route('/uploadImage', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        raise BadRequest("Image data not available!")
    try:
         image_data = request.files['image'].read()
         image = extractImage(image_data)
         data = processImage(image)
         return data.toJson(), 200
    except:
        raise InternalServerError("Uh oh, the request could not be processed.")
    
@app.route('/uploadFile', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        raise BadRequest("File not available")
    file = request.files['file']
    data = file.read()
    fileExtension = os.path.splitext(file)[1]
    if fileExtension not in ['csv', 'xslx']:
        raise UnsupportedFileFormat("Only csv and excel files are supported.")
    try:
         df = readData(data, fileExtension)
         df = processDataFrame(df)
    except:
        raise InternalServerError("Uh oh, the request could not be processed.")

def extractImage(data):
    buffer_data = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(buffer_data, cv2.IMREAD_COLOR)
    return image

def readData(data, fileType):
    if fileType == 'csv':
        df = pd.read_csv(BytesIO(data))
    else:
        df = pd.read_excel(BytesIO(data))
    df.dropna(how='all')
    return df

def processDataFrame(df):
    df.dropna(how = 'all')
    df.columns = [x.lower for x in df.columns]
    # Remove rows that have total as 0 in them
    if 'total' in df.columns:
        df = df[df['total'] > 0]
    return df    


if __name__ == '__main__':
    app.run()




    