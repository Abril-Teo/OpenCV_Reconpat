import numpy as np
import cv2
import pytesseract
import skimage


def grayscale(img):
    gray = cv2.imread(img, 0)
    cv2.imshow(img, gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return gray

def threshold(img_gray, img):
    threshold = cv2.threshold(img_gray, 170, 255, cv2.THRESH_BINARY_INV)[1]
    cv2.imshow(img, threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return threshold

 
img = 'imgs/000.png'

img_gray = grayscale(img)
threshold(img_gray, img)

