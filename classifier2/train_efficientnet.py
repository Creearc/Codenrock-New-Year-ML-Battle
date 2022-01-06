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
tf.random.set_seed(42)
np.random.seed(42)

IMAGE_SIZE = 456
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

BATCH_SIZE = 4

FILTERS = 64
DROPOUT = 0.0

K_PARTS = 5
VALIDATION_SPLIT = 0.0

FREEZE_EPOCHS = 0
UNFREEZE_CONFIG = [(10, 1e-5),
                   (0, 1e-6),
                   (0, 1e-7)]

args = [IMAGE_SIZE, K_PARTS, FREEZE_EPOCHS,
        '|'.join([str(i[0]) for i in UNFREEZE_CONFIG]),
        FILTERS, DROPOUT]

OUTPUT_FILE_NAME = '{}'.format('_'.join([str(i) for i in args]))
OUTPUT_FILE = '{}.tflite'.format(OUTPUT_FILE_NAME)
OUTPUT_FILE_Q = '{}_q.tflite'.format(OUTPUT_FILE_NAME)

###################################

LOAD_MODEL = not  True
MODEL_NAME = '0.48279620350804014__448_3_0_5_1e-05_16_0.1.h5'

EVAL_ONLY = not True

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




if LOAD_MODEL:
  model = tf.keras.models.load_model('results/{}'.format(MODEL_NAME))

else:
  # VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
  conv_base = tf.keras.applications.EfficientNetB5(include_top=False,
                                                 weights='imagenet',
                                                 input_shape=IMG_SHAPE)
  #model = tf.keras.Model(inputs, outputs)
  model = tf.keras.Sequential([
    conv_base,
    tf.keras.layers.GlobalMaxPooling2D(name="gap"),
    tf.keras.layers.Dropout(DROPOUT),
    tf.keras.layers.Dense(units=CLASSES_NUM,
                          activation='softmax')
  ])

  conv_base.trainable = False

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

  conv_base.trainable = True

  for UNFREEZE_EPOCHS, LR in UNFREEZE_CONFIG:
    model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.RMSprop(lr=LR),
    metrics=["acc"],
    )


    history_fine = model.fit(train_data,
                          steps_per_epoch=len(train_data),
                          epochs=UNFREEZE_EPOCHS,
                          validation_data=test_data,
                          validation_steps=len(test_data))

predictions = model.predict_classes(test_data, verbose=0)
labels = validation_data['class_id'].to_numpy()
labels = labels.astype(np.int)

accuracy = accuracy_score(labels, predictions)
print('Result accuracy: {}'.format(accuracy))
score = f1_score(labels, predictions, average='weighted')
print('Result F1: {}'.format(score))

model.save('results/{}__{}.h5'.format(score, OUTPUT_FILE_NAME))
tf.keras.backend.clear_session()





