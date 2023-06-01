from cv2 import contourArea
import numpy as np
import cv2
import pytesseract
import skimage
import matplotlib.pyplot as plt


def mostrarImagen(path, img):
    cv2.imshow(path, img)
    cv2.waitKey(0)



path = 'imgs/001.png'

imagen = cv2.imread(path)
mostrarImagen(path, imagen)

gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
mostrarImagen(path, gris)

threshold = cv2.threshold(gris, 170, 255, cv2.THRESH_BINARY_INV)[1]
mostrarImagen(path, threshold)

contornos, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(threshold, contornos, -1, (0, 255, 0), 2)
mostrarImagen(path, threshold)
