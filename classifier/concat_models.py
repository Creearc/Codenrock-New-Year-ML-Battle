import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

###################################
K_PARTS = 5

MODEL_NAMES = ['0.9843802107650238__448_5_0_10|5|5_64_0.0',
               '0.9766045379811267__448_5_0_10|5|5_64_0.0'                     
               ]

OUTPUT_FILE_NAME = '448_5_0_10|5|5_64_0.0.h5'
###################################

models = []

for MODEL in MODEL_NAMES:
  tmp = tf.keras.models.load_model('results/{}'.format(MODEL))
  if not (tmp is None):
    models.append(tmp)
  else:
    print('{} is empty'.format(MODEL))

model =  tf.keras.layers.concatenate(models, axis = 3)


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

    return k, train_data_generator, test_data_generator

    
k, training_data, validation_data = k_fold_cross_val(data_parts, K_PARTS)

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


predictions = model.predict_classes(test_data, verbose=0)
labels = validation_data['class_id'].to_numpy()
labels = labels.astype(np.int)

accuracy = accuracy_score(labels, predictions)
print('Result accuracy: {}'.format(accuracy))
score = f1_score(labels, predictions, average='weighted')
print('Result F1: {}'.format(score))

model.save('results/{}__{}'.format(score, OUTPUT_FILE_NAME))
