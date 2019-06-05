from DataLoader import DataLoader
from util import *
import numpy as np
import cv2
data = DataLoader("slike")
counter = 0
for image in data.images:
    pixels = getPixels(image)

    pcaMatrix = pca(pixels)

    transformedImage = np.dot(image, pcaMatrix)
    counter += 1
    cv2.imwrite("./slike/rezultati/transformed" + str(counter) + ".png", transformedImage)
