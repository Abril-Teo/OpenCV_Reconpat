from cv2 import contourArea
import numpy as np
import cv2
import pytesseract
import skimage


def mostrarImagen(path, img):
    cv2.imshow(path, img)
    cv2.waitKey(2000)



path = 'imgs/000.png'

normal = cv2.imread(path, 1)
mostrarImagen(path, normal)

gris = cv2.imread(path, 0)
mostrarImagen(path, gris)

threshold = cv2.threshold(gris, 170, 255, cv2.THRESH_BINARY_INV)[1]
mostrarImagen(path, threshold)

contornos = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
canvas = np.zeros_like(path)
contornos2 = cv2.drawContours(canvas, contornos, -1, (0,255,0), 3)
mostrarImagen(path,contornos2)
"""mostrarImagen(path, contornos)"""