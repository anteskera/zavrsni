from DataLoader import DataLoader
from util import *
import numpy as np
import cv2
data = DataLoader("slike")

for image in data.images:
    pixels, mean = getPixels(image)

    pcaMatrix = pca(pixels, mean)

    transformedImage = transformImageWithMatrix(image, pcaMatrix)
    cv2.imshow('transformedImage', transformedImage)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()