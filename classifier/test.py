import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import tensorflow.lite as tflite

import os
import numpy as np
import cv2
from PIL import Image

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


dataset_path = '/home/alexandr/datasets/santas_2'

IMAGE_SIZE = 224
BATCH_SIZE = 32

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

### Create the base model from the pre-trained MobileNet V2
##base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
##                                              include_top=False, 
##                                              weights='imagenet')
##
##base_model.trainable = False
##
##model = tf.keras.Sequential([
##  base_model,
##  tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
##  tf.keras.layers.Dropout(0.2),
##  tf.keras.layers.GlobalAveragePooling2D(),
##  tf.keras.layers.Dense(units=3,
##                        activation='softmax')
##])
##
##model.compile(optimizer='adam', 
##              loss='categorical_crossentropy', 
##              metrics=['accuracy'])
##
##model.summary()
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

if __name__ == "__main__":

  interpreter = load_model('mobilenet_v2_1.0_224_quant.tflite')
  labels = load_labels('frost_labels.txt')

  input_details = interpreter.get_input_details()

  input_shape = input_details[0]['shape']
  height = input_shape[1]
  width = input_shape[2]
  input_index = input_details[0]['index']
  

  for folder in os.listdir(dataset_path):
    for file in os.listdir('{}/{}'.format(dataset_path, folder)):
      img = cv2.imread('{}/{}/{}'.format(dataset_path, folder, file))

      image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
      image = image.resize((width, height))

    
      top_result = process_image(interpreter, img, input_index)
      print(top_result)

    
