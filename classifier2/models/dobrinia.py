import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras import datasets, layers, models, losses


class Model:
  def __init__(self):
    self.IMAGE_SIZE = 416
    self.IMG_SHAPE = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
    
    input_layer = layers.Input(shape=self.IMG_SHAPE)

    conv = layers.Conv2D(filters=16, kernel_size=3,
                                strides=(2, 2),
                                padding='same',
                                activation='relu')(input_layer)

    conv_1x1 = layers.Conv2D(filters=16, kernel_size=1,
                                strides=(1, 1),
                                padding='same',
                                activation='relu')(conv)
    conv_1x1 = layers.BatchNormalization(momentum=0.99)(conv_1x1)

    conv_3x3 = layers.Conv2D(filters=16, kernel_size=1,
                                strides=(1, 1),
                                padding='same',
                                activation='relu')(conv)
    conv_3x3 = layers.Conv2D(filters=32, kernel_size=3,
                                strides=(1, 1),
                                padding='same',
                                activation='relu')(conv_3x3)
    conv_3x3 = layers.BatchNormalization(momentum=0.99)(conv_3x3)

    conv_5x5 = layers.Conv2D(filters=16, kernel_size=1,
                                strides=(1, 1),
                                padding='same',
                                activation='relu')(conv)

    conv_5x5 = layers.Conv2D(filters=32, kernel_size=5,
                                strides=(1, 1),
                                padding='same',
                                activation='relu')(conv_5x5)
    conv_5x5 = layers.BatchNormalization(momentum=0.99)(conv_5x5)

    pool_proj = layers.Conv2D(filters=32, kernel_size=1,
                                strides=(1, 1),
                                padding='same',
                                activation='relu')(conv)
    pool_proj = layers.BatchNormalization(momentum=0.99)(pool_proj)

    conc = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)

    conc = layers.Conv2D(filters=32, kernel_size=3,
                              strides=(2, 2),
                              padding='same',
                              activation='tanh')(conc)
        
    for i in range(4):

      conc = layers.Conv2D(filters=32, kernel_size=3,
                              strides=(1, 1),
                              padding='same',
                              activation='tanh')(conc)

      conc = layers.AveragePooling2D(2)(conc)
      conc = layers.Conv2D(filters=64, kernel_size=3,
                              strides=(2, 2),
                              padding='same',
                              activation='tanh')(conc)
      conc = layers.BatchNormalization(momentum=0.99)(conc)

    conc = layers.Dropout(DROPOUT)(conc)
    conc = layers.GlobalAveragePooling2D()(conc)
    conc = layers.Dense(CLASSES_NUM, activation='softmax')(conc)

    self.model = tf.keras.Model(input_layer, conc)
