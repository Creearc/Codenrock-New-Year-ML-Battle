import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import tensorflow.lite as tflite
import os
import numpy as np
import cv2
from PIL import Image

def load_labels(label_path):
    r"""Returns a list of labels"""
    with open(label_path, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_model(model_path):
    r"""Load TFLite model, returns a Interpreter instance."""
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def process_image(interpreter, image, input_index, k=3):
    r"""Process an image, Return top K result in a list of 2-Tuple(confidence_score, label)"""
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)  # (1, 1001)
    output_data = np.squeeze(output_data)

    # Get top K result
    top_k = output_data.argsort()[-k:][::-1]  # Top_k index
    result = []
    for i in top_k:
        score = float(output_data[i] / 255.0)
        result.append((i, score))

    return result

def display_result(top_result, frame, labels):
    r"""Display top K result in top right corner"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    size = 0.6
    color = (255, 0, 0)  # Blue color
    thickness = 1

    for idx, (i, score) in enumerate(top_result):
        # print('{} - {:0.4f}'.format(label, score))
        x = 12
        y = 24 * idx + 24
        cv2.putText(frame, '{} - {:0.4f}'.format(labels[i], score),
                    (x, y), font, size, color, thickness)
    return frame

def detect_image(input_img_path, output_img_path):
    global interpreter, labels, input_index, height, width
    
    img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)

    img_n = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_n = img_n.resize((width, height))

    
    top_result = process_image(interpreter, img_n, input_index)
    img = display_result(top_result, img, labels)
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
    interpreter = load_model('1_q.tflite')
    labels = load_labels('frost_labels.txt')

    input_details = interpreter.get_input_details()

    input_shape = input_details[0]['shape']
    height = input_shape[1]
    width = input_shape[2]
    input_index = input_details[0]['index']
    print(input_shape)
    app.run(host='0.0.0.0', port=8000, debug=not True)
