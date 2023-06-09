import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
from reconpat import Reconpat


def mostrarImagen(img):    # Funcion para mostrar la imagen y mostrar el programa paso a paso
    cv2.imshow("imagen", img)
    cv2.waitKey(0)

recon = Reconpat() 

img = cv2.imread('imgs/007.png')  # Mostrar la imagen sin ninguna alteracion
mostrarImagen(img)

gray = recon.grayscale(img)  # Pasar la imagen a escala de grises
mostrarImagen(gray)

thresh = recon.apply_threshold(gray) # Aplicar threshold a la imagen en grises 
mostrarImagen(thresh)

contours = recon.find_contours(thresh)  # Buscar contronos a la imagen con threshold
canvas = np.zeros_like(img)             # Dibujar los contronos en una imagen en negro 
cv2.drawContours(canvas , contours, -1, (0, 255, 0), 2)
plt.axis('off')
mostrarImagen(canvas)

candidates = recon.filter_candidates(contours)   # Toma solo los contornos que se asemejan a una patente
canvas = np.zeros_like(img)                      # Dibujar los contronos en una imagen en negro 
cv2.drawContours(canvas , candidates, -1, (0, 255, 0), 2)
plt.axis('off')
mostrarImagen(canvas)

license = recon.get_lowest_candidate(candidates)  # Funcion para solucionar problema de que en ocaciones a demas de la patente toma una luz del auto
canvas = np.zeros_like(img)                       # Dibujar los contronos en una imagen en negro 
cv2.drawContours(canvas , [license], -1, (0, 255, 0), 2)
plt.axis('off')
mostrarImagen(canvas)

cropped = recon.crop_license_plate(img, license)                # Toma la imagen original y recorta la parte de la patente
mostrarImagen(cropped)

gray_cropped = recon.grayscale(cropped)                         # Aplica escala de grises a la patente 
thresh_cropped = recon.apply_adaptive_threshold(gray_cropped)   # Aplica threshold a la patente
mostrarImagen(thresh_cropped)

clear_border = recon.clear_border(thresh_cropped)         # Limpia la imagen 
final = recon.invert_image(clear_border)                  # Invierte la imagen (Blancos a negros; negros a blancos)
mostrarImagen(final)                                      # La imagen queda lista para que se interpreten los digitos con pytesseract

psm = 7                                                  # Configuaraciones pytesseract
alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"            
options = "-c tessedit_char_whitelist={}".format(alphanumeric)
options += " --psm {}".format(psm)
txt = pytesseract.image_to_string(final, config=options)
print(txt[:2], txt[2:5], txt[5:])

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Path to the Tesseract executable