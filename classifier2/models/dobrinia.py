import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras import datasets, layers, models, losses

def nikita_layer(conc,
                 filters_1=32,
                 filters_2=64,
                 name=None):
  conc = layers.Conv2D(filters=filters_1, kernel_size=3,
                              strides=(1, 1),
                              padding='same',
                              activation='tanh')(conc)

  conc = layers.AveragePooling2D(2)(conc)
  conc = layers.Conv2D(filters=filters_2, kernel_size=3,
                              strides=(2, 2),
                              padding='same',
                              activation='tanh')(conc)
  conc = layers.BatchNormalization(momentum=0.99,
                                   name=name)(conc)
  return conc

def inception_module(conc,
                     filters_1x1=64,
                     filters_3x3_reduce=96,
                     filters_3x3=128,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=32,
                     name='inception_3a'):

  conv_1x1 = layers.Conv2D(filters=filters_1x1, kernel_size=1,
                              strides=(1, 1),
                              padding='same',
                              activation='relu')(conc)

  conv_3x3 = layers.Conv2D(filters=filters_3x3_reduce, kernel_size=1,
                              strides=(1, 1),
                              padding='same',
                              activation='relu')(conc)
  conv_3x3 = layers.Conv2D(filters=filters_3x3, kernel_size=3,
                              strides=(1, 1),
                              padding='same',
                              activation='relu')(conv_3x3)

  conv_5x5 = layers.Conv2D(filters=filters_5x5_reduce, kernel_size=1,
                              strides=(1, 1),
                              padding='same',
                              activation='relu')(conc)
  conv_5x5 = layers.Conv2D(filters=filters_5x5, kernel_size=5,
                              strides=(1, 1),
                              padding='same',
                              activation='relu')(conv_5x5)

  pool_proj = layers.Conv2D(filters=filters_pool_proj, kernel_size=1,
                              strides=(1, 1),
                              padding='same',
                              activation='relu')(conc)

  conc = layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
  return conc

def depthwise_conv(conc,
                   filters=64,
                   strides=(1, 1)):
  conc = layers.BatchNormalization(momentum=0.99)(conc)
  conc = layers.ReLU()(conc)
  conc = layers.Conv2D(filters=filters, kernel_size=1,
                              strides=strides,
                              padding='same',
                              activation='tanh')(conc)
  conc = layers.BatchNormalization(momentum=0.99)(conc)
  conc = layers.ReLU()(conc)
  return conc
  

class Model:
  def __init__(self, CLASSES_NUM):
    self.IMAGE_SIZE = 416
    self.IMG_SHAPE = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
    
    input_layer = layers.Input(shape=self.IMG_SHAPE)
    
    conc = depthwise_conv(input_layer,
                          filters=16,
                          strides=(2, 2))

    conc = inception_module(conv,
                     filters_1x1=16,
                     filters_3x3_reduce=16,
                     filters_3x3=32,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=16,
                     name='inception_3a')

    conc = layers.BatchNormalization(momentum=0.99)(conc) 

    conc = depthwise_conv(conc,
                          filters=16,
                          strides=(1, 1))
        
    conc = nikita_layer(conc, filters_1=32, filters_2=64)
    conc = nikita_layer(conc, filters_1=32, filters_2=32)

    conc = layers.Dropout(0.2)(conc)
    conc = layers.GlobalAveragePooling2D()(conc)
    conc = layers.Dense(CLASSES_NUM, activation='softmax')(conc)

    self.model = tf.keras.Model(input_layer, conc)












    
