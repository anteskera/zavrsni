from DataLoader import DataLoader
from util import *
import numpy as np
import cv2
data = DataLoader("slike")
counter = 0
for image in data.images:
    pixels, mean = getPixels(image)

    pcaMatrix = pca(pixels, mean)

    transformedImage = np.dot(image, pcaMatrix)
    counter += 1
    cv2.imwrite("./slike/transformed" + counter + ".png", transformedImage)
