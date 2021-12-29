import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
from sklearn.model_selection import KFold

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

###################################


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

X, y = next(val_generator)

image_batch, label_batch = next(val_generator)
image_batch.shape, label_batch.shape


labels = '\n'.join(sorted(val_generator.class_indices.keys()))

with open('frost_labels.txt', 'w') as f:
  f.write(labels)
  

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

f = open('log.txt', 'a')
f.close()

kfold = KFold(n_splits=K_PARTS, shuffle=True)

for DROPOUT in DROPOUT_CONFIG:
  for LR in LR_CONFIG:
    for UNFREEZE_EPOCHS in UNFREEZE_EPOCHS_CONFIG:
      for FILTERS in FILTERS_CONFIG:
        i = 0
        results = []
        for train, test in kfold.split(X, y):
          OUTPUT_FILE = '{}_{}_{}_{}'.format(DROPOUT, UNFREEZE_EPOCHS, LR, FILTERS)
          OUTPUT_FILE_Q = '{}_q.tflite'.format(OUTPUT_FILE)
          OUTPUT_FILE = '{}.tflite'.format(OUTPUT_FILE)

          model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.Conv2D(filters=FILTERS, kernel_size=3, activation='relu'),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(units=len(train_generator.class_indices),
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

          model.summary()

          print('Number of trainable weights = {}'.format(len(model.trainable_weights)))

          history_fine = model.fit(X[train],
                                   steps_per_epoch=len(X[train]), 
                                   epochs=UNFREEZE_EPOCHS, 
                                   validation_data=y[train],
                                   validation_steps=len(y[train]))
          
          scores = model.evaluate(X[test], y[test], verbose=0)
          
          f = open('log.txt', 'a')
          f.write('{} {}_{}_{}_{}:  {}\n'.format(i, DROPOUT, UNFREEZE_EPOCHS, LR, FILTERS,
                                            result))
          f.close()
          results.append(result)
          i += 1
          
        f = open('log.txt', 'a')
        f.write('       res :{}'.format(sum(result) / K_PARTS))
        f.close()


