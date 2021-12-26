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
  img = cv2.resize(img, (416, 416), interpolation = cv2.INTER_AREA)
  cv2.imwrite('train/{}'.format(element[0]), img)
    
  f.write('{}/{} 5,{},5,{},{}\n'.format(path, element[0],
                                        411, 411,
                                        element[1]))
  
f.close()
