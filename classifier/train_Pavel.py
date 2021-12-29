import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
from sklearn.model_selection import StratifiedKFold as NIKITA
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


dataset_path = '/home/alexandr/datasets/santas_2'

###################################
IMAGE_SIZE = 448
BATCH_SIZE = 32

VALIDATION_SPLIT = 0.2

DROPOUT_CONFIG = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

UNFREEZE_EPOCHS_CONFIG = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]

LR_CONFIG = [1e-4, 1e-5, 1e-6, 1e-7]

FILTERS_CONFIG = [8, 16, 32, 64, 128]

K_PARTS = 5

OUTPUT_FILE = 'm_8.tflite'
OUTPUT_FILE_Q = 'm_8_q.tflite'

CLASSES_NUM = 3 # !!!!!

###################################

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

f = open('log.txt', 'a')
f.close()

classes_paths = os.listdir(dataset_path)
CLASSES_NUM = len(classes_paths)

num_2_vector = lambda x : np.asarray([1.0 if i == int(x) else 0.0 for i in range(CLASSES_NUM)]).astype(np.float32)

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
                                                              'class_id' : num_2_vector(key)},
                                                             ignore_index=True)
          else:
            train_data_generator = train_data_generator.append({"image_name" : label,
                                                                'class_id' : num_2_vector(key)},
                                                               ignore_index=True)
    yield k, train_data_generator, test_data_generator


idg = ImageDataGenerator(rescale=1./255)

for DROPOUT in DROPOUT_CONFIG:
  for LR in LR_CONFIG:
    for UNFREEZE_EPOCHS in UNFREEZE_EPOCHS_CONFIG:
      for FILTERS in FILTERS_CONFIG:
        results = []
        for k, training_data, validation_data in k_fold_cross_val(data_parts, K_PARTS):
          OUTPUT_FILE = '{}_{}_{}_{}'.format(DROPOUT, UNFREEZE_EPOCHS, LR, FILTERS)
          OUTPUT_FILE_Q = '{}_q.tflite'.format(OUTPUT_FILE)
          OUTPUT_FILE = '{}.tflite'.format(OUTPUT_FILE)

          model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Conv2D(filters=FILTERS, kernel_size=3, activation='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=CLASSES_NUM,
                                  activation='softmax')
          ])

          base_model.trainable = True
          fine_tune_at = 100

          # Freeze all the layers before the `fine_tune_at` layer
          for layer in base_model.layers[:fine_tune_at]:
            layer.trainable =  False

          model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    
          train_data = idg.flow_from_dataframe(training_data,
                                               target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                               x_col = "image_name",
                                               y_col = 'class_id', # classes
                                               class_mode = 'raw',
                                               shuffle = True)
          
          test_data = idg.flow_from_dataframe(validation_data,
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                              x_col = "image_name",
                                              y_col = 'class_id', # classes
                                              class_mode = 'raw',
                                              shuffle = True)
##          print(np.unique(labels[train_index]), np.unique(labels[val_index]))
##          train_data = idg.flow(images[train_index], labels[train_index],
##                                batch_size=BATCH_SIZE, subset='training')
##          test_data = idg.flow(images[val_index], labels[val_index],
##                               batch_size=BATCH_SIZE, subset='validation')


          history_fine = model.fit(train_data,
                                   steps_per_epoch=len(train_data), 
                                   epochs=UNFREEZE_EPOCHS, 
                                   validation_data=test_data,
                                   validation_steps=len(test_data))
          
          scores = model.evaluate(valid_data_generator)
          scores = dict(zip(model.metrics_names, scores))['accuracy']
          
          f = open('log.txt', 'a')
          f.write('{} {}_{}_{}_{}:  {}\n'.format(k, DROPOUT, UNFREEZE_EPOCHS, LR, FILTERS,
                                            scores))
          f.close()
          results.append(scores)
          
        f = open('log.txt', 'a')
        f.write('       res :{}'.format(sum(results) / len(results)))
        f.close()

        tf.keras.backend.clear_session()


