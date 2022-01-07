import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


dataset_path = '/home/alexandr/datasets/santas_2'

###################################
tf.random.set_seed(8)
np.random.seed(8)

IMAGE_SIZE = 448
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 2

FILTERS = 64
DROPOUT = 0.0

K_PARTS = 3
VALIDATION_SPLIT = 0.0

FREEZE_EPOCHS = 3
UNFREEZE_CONFIG = [(10, 1e-5),
                   (2, 1e-8)]

args = [IMAGE_SIZE, K_PARTS, FREEZE_EPOCHS,
        '|'.join([str(i[0]) for i in UNFREEZE_CONFIG]),
        FILTERS, DROPOUT]

OUTPUT_FILE_NAME = '{}'.format('_'.join([str(i) for i in args]))

LOAD_MODEL = not True
MODEL_NAME = '0.8654310907491829__448_3_0_10|10_64_0.0'

EVAL_ONLY = not True

OUTPUT_FILE = '{}.tflite'.format(OUTPUT_FILE_NAME)
OUTPUT_FILE_Q = '{}_q.tflite'.format(OUTPUT_FILE_NAME)

###################################

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

if LOAD_MODEL:
  model = tf.keras.models.load_model('results/{}'.format(MODEL_NAME))

else:
  # Create the base model from the pre-trained MobileNet V2
  base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                                 include_top=False,
                                                 weights='imagenet')

  base_model.trainable = False

  model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(filters=FILTERS, kernel_size=3,
                           activation='relu'),
    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units=CLASSES_NUM,
                          activation='softmax')
  ])

for UNFREEZE_EPOCHS, LR in UNFREEZE_CONFIG:
  for period in range(5):    
    for k, training_data, validation_data in k_fold_cross_val(data_parts, K_PARTS):
      training_data = training_data.sample(frac=1)
      validation_data = validation_data.sample(frac=1)

      idg = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                                            rotation_range=45,
                                                            width_shift_range=[-0.3, 0.3],
                                                            height_shift_range=[-0.3, 0.3],
                                                            zoom_range=[0.4, 1.3],
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

      if not EVAL_ONLY :
        if FREEZE_EPOCHS > 0:
          model.compile(optimizer='adam', 
                        loss='categorical_crossentropy', 
                        metrics=['accuracy'])

          history = model.fit(train_data,
                              steps_per_epoch=len(train_data),
                              epochs=FREEZE_EPOCHS,
                              validation_data=test_data,
                              validation_steps=len(test_data))

        
          base_model.trainable = True
##          fine_tune_at = 100
##
##          # Freeze all the layers before the `fine_tune_at` layer
##          for layer in base_model.layers[:fine_tune_at]:
##            layer.trainable =  False

        
        model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])


        history_fine = model.fit(train_data,
                              steps_per_epoch=len(train_data),
                              epochs=UNFREEZE_EPOCHS,
                              validation_data=test_data,
                              validation_steps=len(test_data))


for folder in os.listdir(dataset_path):
  for file in os.listdir('{}/{}'.format(dataset_path, folder)):
    
    img = cv2.imread('{}/{}/{}'.format(dataset_path, folder, file),
                     cv2.IMREAD_COLOR)
    
    img_n = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img_n = np.expand_dims(img_n, 0)

    y = model.predict_classes(img_n)[0]
    predictions.append(str(y))
    labels.append(str(folder))


accuracy = accuracy_score(labels, predictions)
print('Result accuracy: {}'.format(accuracy))
score = f1_score(labels, predictions, average='weighted')
print('Result F1: {}'.format(score))

model.save('results/{}__{}'.format(score, OUTPUT_FILE_NAME))
tf.keras.backend.clear_session()




