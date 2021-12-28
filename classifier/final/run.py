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
    input_data = np.expand_dims(image, axis=0)  # expand to 4-dim

    # Process
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()

    # Get outputs
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = np.squeeze(output_data)

    # Get top K result
    top_k = output_data.argsort()[-k:][::-1]  # Top_k index
    result = []
    ind, mx = 0, 0
    
    for i in top_k:
        score = float(output_data[i] / 255.0)
        result.append((i, score))
        if score > mx:
            mx = score
            ind = i

    return labels[ind]

if __name__ == "__main__":
  

  interpreter = load_model('data/weight/1_q.tflite')
  labels = [0, 1, 2]

  input_details = interpreter.get_input_details()

  input_shape = input_details[0]['shape']
  height = input_shape[1]
  width = input_shape[2]
  input_index = input_details[0]['index']  

  dataset_path = '/home/alexandr/datasets/santas'
  
  f = open('data/out/submission.csv', 'w')
  f.write('image_name	class_id\n')
  for file in os.listdir(dataset_path):
      img = cv2.imread('{}/{}'.format(dataset_path, file))

      img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      img = img.resize((width, height))
    
      result = process_image(interpreter, img, input_index)
      f.write('{}	{}\n'.format(file, result))
  f.close()
      

    
