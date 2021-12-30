import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


dataset_path = '/home/alexandr/datasets/santas_2'

###################################
IMAGE_SIZE = 448
BATCH_SIZE = 32

K_PARTS = 3

FREEZE_EPOCHS = 0
UNFREEZE_EPOCHS = 5

LR = 1e-5
FILTERS = 16
DROPOUT = 0.1

args = [IMAGE_SIZE, K_PARTS, FREEZE_EPOCHS, UNFREEZE_EPOCHS, LR, FILTERS, DROPOUT]
OUTPUT_FILE = '{}.h5'.format('_'.join([str(i) for i in args]))

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

    
k, training_data, validation_data in k_fold_cross_val(data_parts, K_PARTS)

idg = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_data = idg.flow_from_dataframe(training_data,
                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                     x_col = "image_name",
                                     y_col = 'class_id',
                                     batch_size=BATCH_SIZE, 
                                     shuffle = True)
          
test_data = idg.flow_from_dataframe(validation_data,
                                    target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                    x_col = "image_name",
                                    y_col = 'class_id',
                                    batch_size=BATCH_SIZE, 
                                    shuffle = True)


IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(filters=FILTERS, kernel_size=3, activation='relu'),
  tf.keras.layers.Dropout(DROPOUT),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(units=len(train_generator.class_indices),
                        activation='softmax')
])

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
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model.compile(optimizer=tf.keras.optimizers.Adam(UNFREEZE_ADAM_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history_fine = model.fit(train_generator,
                         steps_per_epoch=len(train_generator), 
                         epochs=UNFREEZE_EPOCHS, # <--------------------------------------
                         validation_data=val_generator,
                         validation_steps=len(val_generator))

scores = model.evaluate(test_data)
predictions = model.predict_classes(test_data, verbose=0)
labels = validation_data['class_id'].to_numpy()
labels = labels.astype(np.int)
for i in range(len(labels)):
  print(labels[i], predictions[i])


score = f1_score(labels, predictions, average='weighted')
print('Result F1: {}'.format(score))

model.save('results/{}__{}'.format(score, OUTPUT_FILE))
tf.keras.backend.clear_session()





