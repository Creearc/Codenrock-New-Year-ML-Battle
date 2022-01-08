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
IMAGE_SIZE = 512
BATCH_SIZE = 2

VALIDATION_SPLIT = 0.2

FREEZE_EPOCHS = 1
UNFREEZE_EPOCHS = 10

UNFREEZE_ADAM_LR = 1e-5
DROPOUT = 0.1

OUTPUT_FILE_NAME = 'test_res'

###################################


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, 
    validation_split=VALIDATION_SPLIT)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='training')

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE, 
    subset='validation')


image_batch, label_batch = next(val_generator)
image_batch.shape, label_batch.shape


labels = '\n'.join(sorted(train_generator.class_indices.keys()))

with open('frost_labels.txt', 'w') as f:
  f.write(labels)

IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

# Create the base model from the pre-trained MobileNet V2
base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                              include_top=False, 
                                              weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'),
  tf.keras.layers.Dropout(DROPOUT),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(units=len(train_generator.class_indices),
                        activation='softmax')
])

if FREEZE_EPOCHS > 0:
  model.compile(optimizer='adam', 
                loss='categorical_crossentropy', 
                metrics=['accuracy'])

  model.summary()

  print('Number of trainable weights = {}'.format(len(model.trainable_weights)))

  history = model.fit(train_generator,
                      steps_per_epoch=len(train_generator), 
                      epochs=FREEZE_EPOCHS, # <--------------------------------------
                      validation_data=val_generator,
                      validation_steps=len(val_generator))


  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']

  loss = history.history['loss']
  val_loss = history.history['val_loss']

print("Number of layers in the base model: ", len(base_model.layers))

base_model.trainable = True
##fine_tune_at = 100
##
### Freeze all the layers before the `fine_tune_at` layer
##for layer in base_model.layers[:fine_tune_at]:
##  layer.trainable =  False

model.compile(optimizer=tf.keras.optimizers.Adam(UNFREEZE_ADAM_LR),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print('Number of trainable weights = {}'.format(len(model.trainable_weights)))

history_fine = model.fit(train_generator,
                         steps_per_epoch=len(train_generator), 
                         epochs=UNFREEZE_EPOCHS, # <--------------------------------------
                         validation_data=val_generator,
                         validation_steps=len(val_generator))


predictions = []
labels = []


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


