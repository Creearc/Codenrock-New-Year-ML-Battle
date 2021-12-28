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


path = '/home/alexandr/datasets/santas'
output_path = '/home/alexandr/datasets/santas_2'

os.makedirs(output_path)


for element in dataset:
  img = cv2.imread('{}/{}'.format(path, element[0]))
  cv2.imwrite('{}/{}/{}'.format(output_path, element[0], element[0]), img)
    
  f.write('{}/{} 5,{},5,{},{}\n'.format(path, element[0],
                                        411, 411,
                                        element[1]))
  

