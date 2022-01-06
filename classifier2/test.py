import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

dataset_path = '/home/alexandr/datasets/santas_2'

model = tf.keras.models.load_model('results/0.8357388427607679__512_5_1_2|0|0_64_0.2.h5')

labels = ['Nobody', 'Father Frost', 'Santa']
height, width = 456, 456
height, width = 512, 512

predictions = []
labels = []

for folder in os.listdir(dataset_path):
  for file in os.listdir('{}/{}'.format(dataset_path, folder)):
    
    img = cv2.imread('{}/{}/{}'.format(dataset_path, folder, file),
                     cv2.IMREAD_COLOR)
    
    img_n = cv2.resize(img, (width, height))
    img_n = np.expand_dims(img_n, 0)

    y = model.predict_classes(img_n)[0]
    predictions.append(str(y))
    labels.append(str(folder))


accuracy = accuracy_score(labels, predictions)
print('Result accuracy: {}'.format(accuracy))
score = f1_score(labels, predictions, average='weighted')
print('Result F1: {}'.format(score))
