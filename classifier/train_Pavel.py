import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
import numpy as np
import pandas as pd


def log(text):
  f = open('log.txt', 'a')
  f.write(text)
  f.close()

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


dataset_path = '/home/alexandr/datasets/santas_2'

###################################
IMAGE_SIZE = 448
BATCH_SIZE = 32

DROPOUT_CONFIG = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

UNFREEZE_EPOCHS_CONFIG = [5, 10, 20]

LR_CONFIG = [1e-5, 1e-6, 1e-7]

FILTERS_CONFIG = [8, 16, 32, 64]

K_PARTS = 3

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
        results = []
        for k, training_data, validation_data in k_fold_cross_val(data_parts, K_PARTS):
##          OUTPUT_FILE = '{}_{}_{}_{}'.format(DROPOUT, UNFREEZE_EPOCHS, LR, FILTERS)
##          OUTPUT_FILE_Q = '{}_q.tflite'.format(OUTPUT_FILE)
##          OUTPUT_FILE = '{}.tflite'.format(OUTPUT_FILE)

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
                                               y_col = 'class_id', 
                                               shuffle = True)
          
          test_data = idg.flow_from_dataframe(validation_data,
                                              target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                              x_col = "image_name",
                                              y_col = 'class_id',
                                              shuffle = True)


          history_fine = model.fit(train_data,
                                   steps_per_epoch=len(train_data), 
                                   epochs=UNFREEZE_EPOCHS, 
                                   validation_data=test_data,
                                   validation_steps=len(test_data))
          
          converter = tf.lite.TFLiteConverter.from_keras_model(model)
          tflite_model = converter.convert()

          with open(OUTPUT_FILE, 'wb') as f:
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

          with open(OUTPUT_FILE_Q, 'wb') as f:
            f.write(tflite_model)

          batch_images, batch_labels = next(val_generator)

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

          interpreter = tf.lite.Interpreter(OUTPUT_FILE_Q)
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
          
          f = open('log.txt', 'a')
          f.write('{} epochs: {} filters: {} drop: {}  lr: {}  score: {}\n'.format(k, UNFREEZE_EPOCHS, FILTERS, DROPOUT, LR, 
                                            scores))
          f.close()
          results.append(scores)
          
        f = open('log.txt', 'a')
        f.write('Average result: {}\n'.format(sum(results) / len(results)))
        f.close()

        tf.keras.backend.clear_session()


