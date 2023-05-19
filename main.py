import numpy as np
import cv2
import pytesseract
import skimage

img = cv2.imread('imgs/000.png', 0)
cv2.imshow('imgs/000.png', img)

cv2.waitKey(0)
cv2.destroyAllWindows()