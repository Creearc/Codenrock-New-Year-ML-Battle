import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import random
import time
import tensorflow as tf
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import detect_image
from yolov3.configs import *

yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
yolo.load_weights(f"./checkpoints/yolov4_custom_Tiny")

ALLOWED_EXTENSIONS = set(['jpg', 'png'])
path = 'files'  # path for files to save

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = path

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            saved_file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(saved_file)
            r = filename.split('.')
            print(r)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                      '{}_n.{}'.format(r[0], r[1]))
            
            detect_image(yolo, saved_file, image_path,
                         input_size=YOLO_INPUT_SIZE, show=False,
                         CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))
            out = filename
            return redirect(url_for('uploaded_file',
                                    filename=out))
    return '''
    <!doctype html>
    <title>upload_and_download_files</title>
    <h1>Загрузите файл</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Загрузить>
    </form>
    '''



from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):  
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
