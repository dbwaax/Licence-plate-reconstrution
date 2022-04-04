import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import shutil


# imread
def cv_imread(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8),-1)
    return img 
# imwrite
def cv_imwrite(path,img):
    cv2.imencode('.jpg', img)[1].tofile(path)
    
def checkfiles(root):
    if(not os.path.exists(root)):
        os.mkdir(root)
    else:
        try:
            shutil.rmtree(root)
            os.mkdir(root)
        except:
            os.mkdir(root)
        

