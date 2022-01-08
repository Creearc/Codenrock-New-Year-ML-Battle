import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3
from tensorflow.keras import datasets, layers, models, losses

def swish(x):
    return x * tf.nn.sigmoid(x)

def se_block(conc_1,
             filter_1=16,
             filter_2=16):

  conc = tf.keras.layers.GlobalAveragePooling2D()(conc_1)
  
  conc = tf.expand_dims(input=conc, axis=1)
  conc = tf.expand_dims(input=conc, axis=1)
   
  conc = layers.Conv2D(filters=filter_1,
                         kernel_size=1,
                         strides=(1, 1),
                         padding='same',
                         activation='tanh')(conc)
  conc = swish(conc)
  
  conc = layers.Conv2D(filters=filter_2,
                         kernel_size=1,
                         strides=(1, 1),
                         padding='same',
                         activation='tanh')(conc)
  conc = tf.nn.sigmoid(conc)
  conc = conc_1 * conc
  return conc
  

def mb_conv(conc,
            filter_1=16,
            filter_2=16,
            filter_3=16,
            kernel_size=3,
            strides=(2, 2)):
    conc_skip = conc
    conc = layers.Conv2D(filters=filter_1,
                         kernel_size=1,
                         strides=(1, 1),
                         padding='same',
                         activation='tanh')(conc)
    conc = layers.BatchNormalization(momentum=0.99)(conc)
    conc = swish(conc)

    conc = depthwise_conv(conc,
                          filters=filter_2,
                          kernel_size=kernel_size,
                          strides=strides)
    conc = layers.BatchNormalization(momentum=0.99)(conc)

    conc = se_block(conc)
    conc = swish(conc)

    conc = layers.Conv2D(filters=filter_3,
                         kernel_size=1,
                         strides=(1, 1),
                         padding='same',
                         activation='tanh')(conc)
    conc = layers.BatchNormalization(momentum=0.99)(conc)
    
    if strides == (1, 1) or strides == 1:
      conc = tf.keras.layers.Add()([conc, conc_skip])     
    return conc

  

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
                   kernel_size=1,
                   strides=(1, 1)):
  conc = layers.BatchNormalization(momentum=0.99)(conc)
  conc = layers.ReLU()(conc)
  conc = layers.Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       activation='tanh')(conc)
  conc = layers.BatchNormalization(momentum=0.99)(conc)
  conc = layers.ReLU()(conc)
  return conc

def mobile_conv(conc,
                filters=64,
                kernel_size=1,
                strides=(1, 1)):

  conc = layers.Conv2D(filters=filters,
                       kernel_size=kernel_size,
                       strides=strides,
                       padding='same',
                       activation='tanh')(conc)
  conc = layers.BatchNormalization(momentum=0.99)(conc)
  conc = layers.ReLU()(conc)
  return conc

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    x = tf.keras.layers.Activation('relu')(x)
    # Layer 2
    x = tf.keras.layers.Conv2D(filter, (3,3), padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = tf.keras.layers.Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = tf.keras.layers.Add()([x, x_skip])     
    x = tf.keras.layers.Activation('relu')(x)
    return x
  

class Model:
  def __init__(self, CLASSES_NUM):
    self.IMAGE_SIZE = 416
    self.IMG_SHAPE = (self.IMAGE_SIZE, self.IMAGE_SIZE, 3)
    
    input_layer = layers.Input(shape=self.IMG_SHAPE)

    conc = nikita_layer(input_layer,
                        filters_1=32,
                        filters_2=64)
    conc_skip = conc
    
    conc = inception_module(conc,
                     filters_1x1=16,
                     filters_3x3_reduce=32,
                     filters_3x3=64,
                     filters_5x5_reduce=64,
                     filters_5x5=128,
                     filters_pool_proj=32,
                     name='inception_3a')
    conc = tf.keras.layers.BatchNormalization(axis=3)(conc)

    conc = tf.keras.layers.Add()([conc, conc_skip])
    
    conc = depthwise_conv(conc,
                          filters=16,
                          kernel_size=3,
                          strides=(1, 1))
    conc = mobile_conv(conc,
                       filters=32,
                       kernel_size=3,
                       strides=(1, 1))
   
    conc = depthwise_conv(conc,
                          filters=64,
                          kernel_size=3,
                          strides=(2, 2))

    conc = depthwise_conv(conc,
                          filters=64,
                          kernel_size=3,
                          strides=(1, 1))
    conc = mobile_conv(conc,
                       filters=32,
                       kernel_size=1,
                       strides=(2, 2))

    conc = depthwise_conv(conc,
                          filters=64,
                          kernel_size=3,
                          strides=(1, 1))

    for i in range(3):
      conc = mobile_conv(conc,
                         filters=32,
                         kernel_size=3,
                         strides=(2, 2))
    conc = layers.ReLU()(conc)

    conc_skip = conc

    conc = inception_module(conc,
                     filters_1x1=16,
                     filters_3x3_reduce=16,
                     filters_3x3=32,
                     filters_5x5_reduce=16,
                     filters_5x5=32,
                     filters_pool_proj=16,
                     name='inception_3b')
    conc = tf.keras.layers.BatchNormalization(axis=3)(conc)
    
    conc = tf.keras.layers.Add()([conc, conc_skip])
                                 

    conc = layers.BatchNormalization(momentum=0.99)(conc)
    conc = layers.ReLU()(conc)

    conc = layers.Dropout(0.2)(conc)
    conc = layers.GlobalAveragePooling2D()(conc)
    conc = layers.Dense(CLASSES_NUM, activation='softmax')(conc)

    self.model = tf.keras.Model(input_layer, conc)












    
