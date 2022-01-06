import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

assert float(tf.__version__[:3]) >= 2.3
import tensorflow.lite as tflite
import os
import numpy as np
import cv2
from PIL import Image
import time

def display_result(top_result, frame, labels):
    r"""Display top K result in top right corner"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (0, 0, 0)  # Blue color
    thickness = 4

    h, w = frame.shape[:2]
    k = 640 / w
    frame = cv2.resize(frame, (int(w * k), int(h * k)))

    for idx, score in enumerate(top_result):
        # print('{} - {:0.4f}'.format(label, score))
        x = 12
        y = 24 * idx + 24
        cv2.putText(frame, '{} - {:0.4f}'.format(labels[idx], score),
                    (x, y), font, size, color, thickness)
        cv2.putText(frame, '{} - {:0.4f}'.format(labels[idx], score),
                    (x, y), font, size, (255, 255, 255), thickness-2)
    return frame

def detect_image(input_img_path, output_img_path):
    global model, labels, input_index, height, width
    
    img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)

    img_n = cv2.resize(img, (width, height))
    img_n = np.expand_dims(img_n, 0)

    t = time.time()
    y = model.predict(img_n)
    print(time.time() - t)
    print(y)
    img = display_result(y[0], img.copy(), labels)
    cv2.imwrite(output_img_path, img)


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
            
            detect_image(saved_file, image_path)
            
            out = image_path.split('/')[-1]
            return redirect(url_for('uploaded_file',
                                    filename=out))
    return '''
    <!doctype html>
    <title>upload_and_download_files</title>
    <h1>Загрузите файл</h1>
    <h3>Принимаются изображения в форматах jpg или png</h3>
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
    model = tf.keras.models.load_model('results/nikita_2.h5')
    labels = ['Nobody', 'Father Frost', 'Santa']

    height, width = 456, 456
    
    app.run(host='0.0.0.0', port=8000, debug=not True)
