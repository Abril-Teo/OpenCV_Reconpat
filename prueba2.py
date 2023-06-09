import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
from prueba import Reconpat


def mostrarImagen(img):
    cv2.imshow("imagen", img)
    cv2.waitKey(0)

recon = Reconpat()

img = cv2.imread('imgs/002.png')
mostrarImagen(img)

gray = recon.grayscale(img)
mostrarImagen(gray)

thresh = recon.apply_threshold(gray)
mostrarImagen(thresh)

contours = recon.find_contours(thresh)
canvas = np.zeros_like(img)
cv2.drawContours(canvas , contours, -1, (0, 255, 0), 2)
plt.axis('off')
mostrarImagen(canvas)

candidates = recon.filter_candidates(contours)
canvas = np.zeros_like(img)
cv2.drawContours(canvas , candidates, -1, (0, 255, 0), 2)
plt.axis('off')
mostrarImagen(canvas)

license = recon.get_lowest_candidate(candidates)
canvas = np.zeros_like(img)
cv2.drawContours(canvas , [license], -1, (0, 255, 0), 2)
plt.axis('off')
mostrarImagen(canvas)

cropped = recon.crop_license_plate(gray, license)
cropped2 = recon.crop_license_plate(img, license)
mostrarImagen(cropped2)

thresh_cropped = recon.apply_adaptive_threshold(cropped)
mostrarImagen(thresh_cropped)

clear_border = recon.clear_border(thresh_cropped)
final = recon.invert_image(clear_border)
mostrarImagen(final)

psm = 7
alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
options = "-c tessedit_char_whitelist={}".format(alphanumeric)
options += " --psm {}".format(psm)
txt = pytesseract.image_to_string(final, config=options)
print(txt[:2], txt[2:5], txt[5:])

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Path to the Tesseract executable