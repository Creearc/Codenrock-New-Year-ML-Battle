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

IMAGE_SIZE = 448
BATCH_SIZE = 32

FILTERS = 32
DROPOUT = 0.2

K_PARTS = 7
VALIDATION_SPLIT = 0.2

FREEZE_EPOCHS = 10
UNFREEZE_CONFIG = [(30, 1e-5),
                   (10, 1e-6),
                   (5, 1e-7)]

args = [IMAGE_SIZE, K_PARTS, FREEZE_EPOCHS,
        '|'.join([str(i[0]) for i in UNFREEZE_CONFIG]),
        FILTERS, DROPOUT]

OUTPUT_FILE_NAME = '{}'.format('_'.join([str(i) for i in args]))

LOAD_MODEL = not  True
MODEL_NAME = '0.48279620350804014__448_3_0_5_1e-05_16_0.1.h5'
MODEL_NAME = 'base.h5'

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


IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

if LOAD_MODEL:
  model = tf.keras.models.load_model('results/{}'.format(MODEL_NAME))

else:
  # Create the base model from the pre-trained MobileNet V2
  base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
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

  if not LOAD_MODEL:
    base_model.trainable = True
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
      layer.trainable =  False

  for UNFREEZE_EPOCHS, LR in UNFREEZE_CONFIG:
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


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

model.save('results/{}__{}'.format(score, OUTPUT_FILE_NAME))
tf.keras.backend.clear_session()



converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('results/{}'.format(OUTPUT_FILE), 'wb') as f:
  f.write(tflite_model)

# A generator that provides a representative dataset
def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files(dataset_path + '/*/*')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image / 255., tf.float32)
    image = tf.expand_dims(image, 0)
    yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
# This enables quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# This sets the representative dataset for quantization
converter.representative_dataset = representative_data_gen
# This ensures that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# For full integer quantization, though supported types defaults to int8 only, we explicitly declare it for clarity.
converter.target_spec.supported_types = [tf.int8]
# These set the input and output tensors to uint8 (added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

with open('results/{}'.format(OUTPUT_FILE_Q), 'wb') as f:
  f.write(tflite_model)

batch_images, batch_labels = next(test_data)

logits = model(batch_images)
prediction = np.argmax(logits, axis=1)
truth = np.argmax(batch_labels, axis=1)

keras_accuracy = tf.keras.metrics.Accuracy()
keras_accuracy(prediction, truth)

print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

def set_input_tensor(interpreter, input):
  input_details = interpreter.get_input_details()[0]
  tensor_index = input_details['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  # Inputs for the TFLite model must be uint8, so we quantize our input data.
  # NOTE: This step is necessary only because we're receiving input data from
  # ImageDataGenerator, which rescaled all image data to float [0,1]. When using
  # bitmap inputs, they're already uint8 [0,255] so this can be replaced with:
  #   input_tensor[:, :] = input
  scale, zero_point = input_details['quantization']
  input_tensor[:, :] = np.uint8(input / scale + zero_point)

def classify_image(interpreter, input):
  set_input_tensor(interpreter, input)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = interpreter.get_tensor(output_details['index'])
  # Outputs from the TFLite model are uint8, so we dequantize the results:
  scale, zero_point = output_details['quantization']
  output = scale * (output - zero_point)
  top_1 = np.argmax(output)
  return top_1

interpreter = tf.lite.Interpreter('results/{}'.format(OUTPUT_FILE_Q))
interpreter.allocate_tensors()

# Collect all inference predictions in a list
batch_prediction = []
batch_truth = np.argmax(batch_labels, axis=1)

for i in range(len(batch_images)):
  prediction = classify_image(interpreter, batch_images[i])
  batch_prediction.append(prediction)

# Compare all predictions to the ground truth
tflite_accuracy = tf.keras.metrics.Accuracy()
tflite_accuracy(batch_prediction, batch_truth)
print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))


