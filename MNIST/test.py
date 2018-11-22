import cv2
import numpy as np
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

model = load_model('test.h5')

image = cv2.imread('D:/workspace-python/ml/test/9_9.png')
img = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
img = (img.reshape(1, 28, 28, 1)).astype('int32')/255
predict = model.predict_classes(img)
print (predict)
