import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import tensorflow.lite as tflite
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

    return self.model.predict(img_n)[0]


class lite_net():
  def __init__(self, path, img_shape=(448, 448)):
    self.interpreter = tflite.Interpreter(model_path=path)
    self.interpreter.allocate_tensors()
    input_details = self.interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    self.img_shape = (input_shape[2], input_shape[1])
    self.input_index = input_details[0]['index'] 

  def run(self, img):
    img_n = img.copy()
    img_n = Image.fromarray(cv2.cvtColor(img_n, cv2.COLOR_BGR2RGB))
    img_n = img_n.resize(self.img_shape)

    input_data = np.expand_dims(img_n, axis=0)
    self.interpreter.set_tensor(self.input_index, input_data)
    self.interpreter.invoke()

    output_details = self.interpreter.get_output_details()
    output_data = self.interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data)

    return [i / 255.0 for i in output_data]



dataset_path = '/home/alexandr/datasets/santas_2'

nets = []
nets.append(lite_net('results/1_q.tflite'))
nets.append(lite_net('results/2_q.tflite'))
nets.append(lite_net('results/m_5_q.tflite'))


predictions = [[] for i in range(len(nets))]
labels = []

output_files = [open('{}_net.csv'.format(i), 'w') for i in range(len(nets))]

for folder in os.listdir(dataset_path):
  for file in os.listdir('{}/{}'.format(dataset_path, folder)):
    
    img = cv2.imread('{}/{}/{}'.format(dataset_path, folder, file),
                     cv2.IMREAD_COLOR)
    labels.append(folder)

    for i in range(len(nets)):
      results = nets[i].run(img)
      predictions[i].append(np.argmax(results))
      output_files[i].write('{};{}\n'.format(folder, ';'.join([str(r) for r in results])))
      #print(results, np.argmax(results), folder)


for i in range(len(nets)):
  output_files[i].close()


for i in range(len(nets)):
  print('Net {}'.format(i))
  accuracy = accuracy_score(labels, predictions[i])
  print('  Accuracy: {}'.format(accuracy))
  score = f1_score(labels, predictions[i], average='weighted')
  print('  F1: {}'.format(score))
  
