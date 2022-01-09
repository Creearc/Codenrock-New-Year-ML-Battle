import numpy as np
import os
import cv2

dataset = []
classes = set()

f = open('train.csv', 'r')
f.readline()

for s in f:
  img_name, class_index = s.split()
  dataset.append((img_name, class_index))
  classes.add(class_index)

f.close()

np.random.shuffle(dataset)
classes = list(classes)

path = '/home/alexandr/datasets/santas'
output_path = '/home/alexandr/datasets/santas_2'

path = 'train'
output_path = 'train_s'


try:
  for c in classes:
    os.makedirs('{}/{}'.format(output_path, c))
except:
  pass

for element in dataset:
  img = cv2.imread('{}/{}'.format(path, element[0]))
  cv2.imwrite('{}/{}/{}'.format(output_path, element[1], element[0]), img)

  

