import numpy as np
import os
import cv2

dataset = []

f = open('train.csv', 'r')
f.readline()

for s in f:
  img_name, class_index = s.split()
  dataset.append((img_name, class_index))

f.close()

np.random.shuffle(dataset)


path = '{}/train'.format(os.getcwd())

f = open('ded_train.txt', 'w')
for element in dataset:
  img = cv2.imread('train/{}'.format(element[0]))
  (width, height) = (img.shape[1] // 2, img.shape[0] // 2)
    
  f.write('{}/{} 0,{},0,{},{}\n'.format(path, element[0],
                                        width, height,
                                        element[1]))
  
f.close()
