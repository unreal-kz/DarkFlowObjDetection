import cv2
from darkflow.net.build import TFNet
from random import *
import numpy as np
import time
import re


options = {
    'model': 'cfg\yolo.cfg',
    'load': 'bin\yolo.weights',
    'threshold': 0.3
}

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    stime = time.time()
    ret, frame = capture.read()
    results = tfnet.return_predict(frame)
    print (results)
    if ret:
        for c,r in zip(colors,results):
            tl = (r['topleft']['x'],r['topleft']['y'])
            br = (r['bottomright']['x'],r['bottomright']['y'])
            label = r['label']
            confe = r['confidence']
            text = '{}:{:.0f}%'.format(label,confe*100)
            frame = cv2.rectangle(frame,tl,br,c, 5)
            frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)
        cv2.imshow('frame',frame)
        print('FPS {:.1f}'.format(1/(time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()