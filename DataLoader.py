from __future__ import division
import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
class DataLoader():

    def __init__(self, path):
        onlyPNG = [path + '/' + f for f in listdir(path) if (isfile(join(path, f)) and f.endswith("10.png"))]
        self.images = []
        self.test = []
        for file in onlyPNG:
            img = cv2.imread(file, -1)
            self.test.append(img)
            self.images.append(img/4096)



