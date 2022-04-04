import cv2
import numpy as np
# imread
def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),-1)
    return img 
# imwrite
def cv_imwrite(path,img):
    cv2.imencode('.jpg', img)[1].tofile(path)