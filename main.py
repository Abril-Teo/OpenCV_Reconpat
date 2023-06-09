from cv2 import contourArea
import numpy as np
import cv2
import pytesseract
import skimage
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # Path to the Tesseract executable


def Reconpat(path):

    def mostrarImagen(path, img):
        if (path == 'imgs/001.png'):
            cv2.imshow(path, img)
            cv2.waitKey(0)



    imagen = cv2.imread(path)
    cv2.imshow(path, imagen)
    cv2.waitKey(0)

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    mostrarImagen(path, gris)

    threshold = cv2.threshold(gris, 170, 255, cv2.THRESH_BINARY_INV)[1]
    mostrarImagen(path, threshold)

    contornos, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros_like(imagen)
    cv2.drawContours(canvas, contornos, -1, (0, 255, 0), 2)
    plt.axis('off')
    mostrarImagen(path, canvas)

    tamaño_patente = 3.07692307692
    min_w=80
    max_w=110
    min_h=25
    max_h=52
    candidatos = []
    for cnt in contornos:
            x, y, w, h = cv2.boundingRect(cnt)
            aspecto_patente = float(w) / h
            if (np.isclose(aspecto_patente, tamaño_patente, atol=0.7) and
                (max_w > w > min_w) and
                (max_h > h > min_h)):
                candidatos.append(cnt)
    canvas = np.zeros_like(imagen)
    cv2.drawContours(canvas, candidatos, -1, (0, 255, 0), 2)
    plt.axis('off')
    mostrarImagen(path, canvas)

    ys = []
    for cnt in candidatos:
        x, y, w, h = cv2.boundingRect(cnt)
        ys.append(y)
    license = candidatos[np.argmax(ys)]
    canvas = np.zeros_like(imagen)
    cv2.drawContours(canvas, [license], -1, (0, 255, 0), 2)
    plt.axis('off')
    mostrarImagen(path, canvas)

    x,y,w,h = cv2.boundingRect(license)
    recortado = imagen[y:y+h,x:x+w]
    mostrarImagen(path, recortado)

    gris_recortado = cv2.cvtColor(recortado, cv2.COLOR_BGR2GRAY)
    threshold_recortado = cv2.adaptiveThreshold(gris_recortado , 255 ,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV , 7 , 13)
    mostrarImagen(path, threshold_recortado)

    bordes = skimage.segmentation.clear_border(threshold_recortado)
    mostrarImagen(path, bordes)

    final = cv2.bitwise_not(bordes)
    mostrarImagen(path, final)

    psm = 7
    alphanumeric = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    options = "-c tessedit_char_whitelist={}".format(alphanumeric)
    options += " --psm {}".format(psm)
    txt = pytesseract.image_to_string(final, config=options)
    print(txt[:2], txt[2:5], txt[5:-1])
    return txt



path = ['imgs/001.png', 'imgs/000.png', 'imgs/002.png', 'imgs/003.png', 'imgs/004.png', 'imgs/005.png', 'imgs/006.png', 'imgs/007.png',
 'imgs/008.png', 'imgs/009.png', 'imgs/010.png', 'imgs/011.png', 'imgs/012.png', 'imgs/013.png']

aux = 0

for i in path:
    aux += 1
    print(f'Patente Imagen {aux}:')
    Reconpat(i)

    print("---------------------")
