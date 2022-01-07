import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

######################################################################
'''                           PARAMETERS                           '''
######################################################################
dataset_path = '/home/alexandr/datasets/santas_2'

BATCH_SIZE = 16

K_PARTS = 5
NO_K_TRAINING = not True

DROPOUT = 0.2

UNFREEZE_CONFIG = [(3, 1e-2),
                   (6, 1e-4),
                   (50, 1e-6),
                   (7, 1e-8),]


OUTPUT_FILE_NAME = 'dobrynya_1'

LOAD = None
#LOAD = 'results/0.48255649759799135__dobrynya_1.h5'
#LOAD = 'results/0.5133689820328083__dobrynya_1.h5'

######################################################################

classes_paths = os.listdir(dataset_path)
CLASSES_NUM = len(classes_paths)

# Read data
data = dict()
for i in range(len(classes_paths)):
  class_name = classes_paths[i]
  data[class_name] = []
  for img in os.listdir('{}/{}'.format(dataset_path, classes_paths[i])):
    data[class_name].append('{}/{}/{}'.format(dataset_path, classes_paths[i], img))

# Split data
data_parts = [dict() for i in range(K_PARTS)]
for key in data.keys():
  tmp = np.array_split(data[key], K_PARTS)
  for i in range(K_PARTS):
    data_parts[i][key] = tmp[i]


def k_fold_cross_val(data_parts, K_PARTS):
  for k in range(K_PARTS):
    train_data_generator = pd.DataFrame(data={"image_name" : [],
                                              'class_id' : []})
    test_data_generator = pd.DataFrame(data={"image_name" : [],
                                             'class_id' : []})
    for i in range(K_PARTS):
      for key in data_parts[i].keys():
        for label in data_parts[i][key]:
          if i == k:
            test_data_generator = test_data_generator.append({"image_name" : label,
                                                              'class_id' : str(key)},
                                                             ignore_index=True)
          else:
            train_data_generator = train_data_generator.append({"image_name" : label,
                                                                'class_id' : str(key)},
                                                               ignore_index=True)

    yield k, train_data_generator, test_data_generator
    
######################################################################
'''                            TRAINING                            '''
######################################################################


# Model
if LOAD is None:
  from models import dobrinia

  model = dobrinia.Model(CLASSES_NUM)

  IMAGE_SIZE = model.IMAGE_SIZE
  IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

  model = model.model
else:
  IMAGE_SIZE = 416
  IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
  model = tf.keras.models.load_model(LOAD)
  
model.summary()

for UNFREEZE_EPOCHS, LR in UNFREEZE_CONFIG:    
  for k, training_data, validation_data in k_fold_cross_val(data_parts, K_PARTS):
    training_data = training_data.sample(frac=1)
    validation_data = validation_data.sample(frac=1)

    idg = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                          rotation_range=15,
                                                          width_shift_range=0.2,
                                                          height_shift_range=0.2,
                                                          zoom_range=[0.8, 1.2],
                                                          rescale=1./255)

    train_data = idg.flow_from_dataframe(training_data,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                         x_col = "image_name",
                                         y_col = 'class_id',
                                         batch_size=BATCH_SIZE, 
                                         shuffle = False)
              
    test_data = idg.flow_from_dataframe(validation_data,
                                        target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                        x_col = "image_name",
                                        y_col = 'class_id',
                                        batch_size=BATCH_SIZE, 
                                        shuffle = False)

    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    history = model.fit(train_data,
                            steps_per_epoch=len(train_data),
                            epochs=UNFREEZE_EPOCHS,
                            validation_data=test_data,
                            validation_steps=len(test_data))
    if NO_K_TRAINING:
      break


######################################################################
'''                           VALIDATION                           '''
######################################################################

predictions = []
labels = []

for folder in os.listdir(dataset_path):
  for file in os.listdir('{}/{}'.format(dataset_path, folder)):
    
    img = cv2.imread('{}/{}/{}'.format(dataset_path, folder, file),
                     cv2.IMREAD_COLOR)
    
    img_n = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_n = np.expand_dims(img_n, 0)

    y = np.argmax(model.predict(img_n))
    predictions.append(str(y))
    labels.append(str(folder))


accuracy = accuracy_score(labels, predictions)
print('Result accuracy: {}'.format(accuracy))
score = f1_score(labels, predictions, average='weighted')
print('Result F1: {}'.format(score))

model.save('results/{}__{}.h5'.format(score, OUTPUT_FILE_NAME))
