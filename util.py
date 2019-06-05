from __future__ import division
from sklearn.decomposition import PCA
import cv2
import numpy as np
import math
def getPixels(image):
    mean = np.mean(image, axis=(0, 1))
    h = image.shape[0]
    w = image.shape[1]
    distanceWithPixel = []
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            distance = np.matmul(image[y][x], mean)/(np.linalg.norm(mean))
            distanceWithPixel.append((distance, image[y][x]))

    sorted(distanceWithPixel, key=lambda x: x[0])

    n = 0.036 * len(distanceWithPixel)
    selectedPixels = distanceWithPixel[:int(n)]
    selectedPixels.append(distanceWithPixel[:-int(n)])

    pixelValues = np.array([pixel[1] for pixel in selectedPixels])

    return pixelValues, mean

'''def pca(X):
    # Principal Component Analysis
    # input: X, matrix with training data as flattened arrays in rows
    # return: projection matrix (with important dimensions first),
    # variance and mean


    X=X[0:-1]
    X = np.stack(X, axis=0)

    #get dimensions
    num_data, dim = X.shape

    #center data
    mean_X = X.mean(axis=0)
    for i in range(num_data):
        X[i] -= mean_X

    U, S, V = np.linalg.svd(X)
    V = V[:num_data]  # only makes sense to return the first num_data

    # return the projection matrix, the variance and the mean
    return V, S, mean_X'''

def transformImageWithMatrix(image, matrix):
    h = image.shape[0]
    w = image.shape[1]
    transformedImage = np.zeros((h, w))
    # loop over the image, pixel by pixel
    for y in range(0, h):
        for x in range(0, w):
            transformedImage[y][x] = np.matmul(image[y][x], matrix)
    return transformedImage



def pca(pixels, mean):
    # Principal Component Analysis
    # input: X, matrix with training data as flattened arrays in rows
    # return: projection matrix (with important dimensions first),
    # variance and mean

    # get dimensions

    X = pixels[0:-1]
    X = np.stack(X, axis=0)

    n, m = X.shape

    # center data
    for i in range(n):
        X[i] -= mean

    C = np.dot(X.T, X) / (n - 1)
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    pixels_pca = np.dot(X, eigen_vecs)

    return pixels_pca
