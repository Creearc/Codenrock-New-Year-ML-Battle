import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
import cv2
from PIL import Image

from sklearn.metrics import f1_score, accuracy_score


class full_net():
  def __init__(self, path, img_shape=(456, 456)):
    self.model = tf.keras.models.load_model(path)
    self.img_shape = img_shape

  def run(self, img):
    img_n = img.copy()
    img_n = cv2.resize(img_n, self.img_shape)
    img_n = np.expand_dims(img_n, 0)

    return self.model.predict_classes(img_n)[0]


class lite_net():
  def __init__(self, path, img_shape=(448, 448)):
    interpreter = load_model(path)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    self.img_shape = (input_shape[2], input_shape[1])
    input_index = input_details[0]['index'] 

  def run(self, img):
    img_n = img.copy()
    img_n = Image.fromarray(cv2.cvtColor(img_n, cv2.COLOR_BGR2RGB))
    img = img.resize(self.img_shape)
