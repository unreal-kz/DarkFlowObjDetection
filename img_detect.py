import cv2
from darkflow.net.build import TFNet
from random import *
import tensorflow as tf

options = {
    'model': 'cfg\yolo.cfg',
    'load': 'bin\yolo.weights',
    'threshold': 0.2 
}

tfnet = TFNet(options)

#img = cv2.imread('sample_img\sample_eagle.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('image3.jpg', cv2.IMREAD_COLOR)
img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r = tfnet.return_predict(img)

print(r[0]['label'])
r_size = len(r)
i = 0

while (i < r_size):
    tl = (r[i]['topleft']['x'],r[i]['topleft']['y'])
    br = (r[i]['bottomright']['x'],r[i]['bottomright']['y'])
    label = r[i]['label']
    img = cv2.rectangle(img,tl,br,(randint(0, 255),randint(0, 255),randint(0, 255)), 7)
    img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
    i=i+1
    
img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("result.png", img)