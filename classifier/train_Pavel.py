import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def log(text):
  f = open('log.txt', 'a')
  f.write(text)
  f.close()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


dataset_path = '/home/alexandr/datasets/santas_2'

###################################
tf.random.set_seed(42)
np.random.seed(42)

IMAGE_SIZE = 224*2
BATCH_SIZE = 32

DROPOUT_CONFIG = [0.0]
#DROPOUT_CONFIG = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

FREEZE_EPOCHS = 0
UNFREEZE_EPOCHS_CONFIG = [10]
#UNFREEZE_EPOCHS_CONFIG = [70, 75, 80, 90]

OPTIMIZER_CONFIG = [tf.keras.optimizers.Adam,
                    tf.keras.optimizers.Adamax]

OPTIMIZER_CONFIG = [tf.keras.optimizers.Adam]
LR_CONFIG = [1e-5]

FILTERS_CONFIG = [64]
#FILTERS_CONFIG = [32, 16, 8]

K_PARTS = 5

###################################

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)


log('_______________________________________________________________\n')

classes_paths = os.listdir(dataset_path)
CLASSES_NUM = len(classes_paths)

num_2_vector = lambda x : [1 if i == int(x) else 0 for i in range(CLASSES_NUM)]

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


idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

for UNFREEZE_EPOCHS in UNFREEZE_EPOCHS_CONFIG:
  for DROPOUT in DROPOUT_CONFIG:
    for LR in LR_CONFIG:
      for FILTERS in FILTERS_CONFIG:
        for OPTIMIZER in OPTIMIZER_CONFIG:
          results = []
          
          log('epochs: {} filters: {} drop: {} lr: {} optimizer: {}\n'.format(UNFREEZE_EPOCHS, FILTERS,
                                                                              DROPOUT, LR,
                                                                              OPTIMIZER))
          args = [IMAGE_SIZE, K_PARTS, FREEZE_EPOCHS, UNFREEZE_EPOCHS, LR, FILTERS, DROPOUT]
          OUTPUT_FILE = '{}.h5'.format('_'.join([str(i) for i in args]))
          
          for k, training_data, validation_data in k_fold_cross_val(data_parts, K_PARTS):
            base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                include_top=False, 
                                                weights='imagenet')

            

            model = tf.keras.Sequential([
              base_model,
              tf.keras.layers.Conv2D(filters=FILTERS, kernel_size=3, activation='relu'),
              tf.keras.layers.Dropout(DROPOUT),
              tf.keras.layers.GlobalAveragePooling2D(),
              tf.keras.layers.Dense(units=CLASSES_NUM,
                                    activation='softmax')
            ])
            
            for i in range(20):
              training_data = training_data.sample(frac=1)
              validation_data = validation_data.sample(frac=1)

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

              

              if FREEZE_EPOCHS > 0:
                base_model.trainable = False
                
                model.compile(optimizer='adam', 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

                history = model.fit(train_data,
                                         steps_per_epoch=len(train_data), 
                                         epochs=FREEZE_EPOCHS, 
                                         validation_data=test_data,
                                         validation_steps=len(test_data))
              

              base_model.trainable = True
              fine_tune_at = 100

              # Freeze all the layers before the `fine_tune_at` layer
              for layer in base_model.layers[:fine_tune_at]:
                layer.trainable =  False
                
              model.compile(optimizer=OPTIMIZER(LR),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])


              history_fine = model.fit(train_data,
                                       steps_per_epoch=len(train_data), 
                                       epochs=UNFREEZE_EPOCHS, 
                                       validation_data=test_data,
                                       validation_steps=len(test_data))
            
            

            scores = model.evaluate(test_data)
            predictions = model.predict_classes(test_data, verbose=0)
            labels = validation_data['class_id'].to_numpy()
            labels = labels.astype(np.int)
  ##          for i in range(len(labels)):
  ##            print(labels[i], predictions[i])

            accuracy = accuracy_score(labels, predictions)
            score = f1_score(labels, predictions, average='weighted')
            
            #model.load_weights('./checkpoints/my_checkpoint')
            log('{} |   F1: {} accuracy: {}\n'.format(k, score, accuracy))
            if score > 0.98:
              model.save('results/{}__{}'.format(score, OUTPUT_FILE))

            results.append(score)
            
          log('Average result: {}\n\n'.format(sum(results) / len(results)))

          tf.keras.backend.clear_session()


