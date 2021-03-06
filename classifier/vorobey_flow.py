
###################################
IMAGE_SIZE = 448
BATCH_SIZE = 32

DROPOUT_CONFIG = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

UNFREEZE_EPOCHS_CONFIG = [80, 85, 90, 95]

LR_CONFIG = [1e-5, 1e-6, 1e-7]

FILTERS_CONFIG = [8, 16, 32, 64]

K_PARTS = 5

###################################

CONFIG = [UNFREEZE_EPOCHS_CONFIG,
          DROPOUT_CONFIG,
          LR_CONFIG,
          FILTERS_CONFIG]

def roll_parameters(config, parameters=[]):
  config = config.copy()
  parameters = parameters.copy()
  if len(config) == 0:
    print('Return', parameters)
    return parameters
    
  next_config = config.copy()
  next_config.pop(0)
  for parameter in config[0]:
    yield roll_parameters(next_config, parameters+[parameter])


def roll(config, parameters=[]):
  for parameter in config[0]:
    parameters = parameters + [parameter]
    if len(parameters) == len(config):
      yield parameters

def roll2(config, parameters=[]):
  config = config.copy()
  parameters = parameters.copy()

  next_config = config.copy()
  next_config.pop(0)
  
  for parameter in config[0]:
    for parameter2 in roll(next_config, parameters + [parameter]):
      yield roll(next_config, parameters + [parameter])


for parameters in roll2(CONFIG):
  print('Here -> ', parameters)



