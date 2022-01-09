import os
import cv2
import numpy as np

######################################################################
'''                           PARAMETERS                           '''
######################################################################
path = 'train'

img_shape = (448, 448)
split_matrix = (4, 4)

######################################################################

imgs = os.listdir(path)

##img_name = imgs[2]
##img = cv2.imread('{}/{}'.format(path, img_name))
##
##img = cv2.resize(img, img_shape, cv2.INTER_AREA)
##out = img.copy()
##
##hist = []
##for i in range(3):
##  h = cv2.calcHist([img], [i], None, [256], [0,256])
##  hist.append(h)
##  print(np.median(h))
##
##path_shape = (img_shape[0] // split_matrix[0],
##              img_shape[1] // split_matrix[1])
##
##for x in range(split_matrix[0]):
##  for y in range(split_matrix[1]):
##    cv2.rectangle(img,
##                  (x * path_shape[0], y * path_shape[1]),
##                  ((x + 1) * path_shape[0], (y + 1) * path_shape[1]),
##                  (255, 0, 0), 1)
##    
##
##
##
##cv2.imshow('', img)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

dataset_path = 'train_s'

f = open('smth_wild.csv', 'w')

for folder in os.listdir(dataset_path):
  for file in os.listdir('{}/{}'.format(dataset_path, folder)):
    
    img = cv2.imread('{}/{}/{}'.format(dataset_path, folder, file),
                     cv2.IMREAD_COLOR)

    f.write('{}'.format(folder))

    for i in range(3):
      h = cv2.calcHist([img], [i], None, [256], [0,256])
      h = np.median(h)
      f.write(';{}'.format(h))
    f.write('\n')

f.close()
