from DataLoader import DataLoader
from util import *
import numpy as np
import cv2
data = DataLoader("slike")
counter = 0
for image in data.images:
    pixels = getPixels(image)

    pcaMatrix, mean = pca(pixels)

    transformedImage = transform(image, pcaMatrix, mean)

    counter += 1
    cv2.imwrite("./slike/rezultati/transformed" + str(counter) + ".png", transformedImage)
