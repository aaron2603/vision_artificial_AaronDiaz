import cv2
import numpy as np

# ---------- FUNCION PARA OBTENER ESQUELETO ----------
def obtener_esqueleto(imagen_binaria):

    size = np.size(imagen_binaria)
    skel = np.zeros(imagen_binaria.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False

    while not done:

        eroded = cv2.erode(imagen_binaria, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(imagen_binaria, temp)
        skel = cv2.bitwise_or(skel, temp)
        imagen_binaria = eroded.copy()

        zeros = size - cv2.countNonZero(imagen_binaria)

        if zeros == size:
            done = True

    return skel


# ---------- PROCESAR IMAGEN BAILARINA ----------

img_bailarina = cv2.imread("bailarina.png", cv2.IMREAD_GRAYSCALE)

# mejorar contraste
img_bailarina = cv2.equalizeHist(img_bailarina)

# binarizar
_, bin_bailarina = cv2.threshold(img_bailarina,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# limpiar ruido
kernel = np.ones((3,3),np.uint8)
bin_bailarina = cv2.morphologyEx(bin_bailarina, cv2.MORPH_OPEN, kernel)

# obtener esqueleto
skel_bailarina = obtener_esqueleto(bin_bailarina)

# ---------- PROCESAR IMAGEN BAILARINA ----------

img_bailarina = cv2.imread("bailarina.png", cv2.IMREAD_GRAYSCALE)

# mejorar contraste
img_bailarina = cv2.equalizeHist(img_bailarina)

# suavizar para eliminar ruido
img_bailarina = cv2.GaussianBlur(img_bailarina, (5,5), 0)

# binarización automática
_, bin_bailarina = cv2.threshold(img_bailarina, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# kernel morfológico
kernel = np.ones((3,3), np.uint8)

# cerrar pequeñas separaciones en la silueta
bin_bailarina = cv2.morphologyEx(bin_bailarina, cv2.MORPH_CLOSE, kernel, iterations=2)

# eliminar ruido pequeño
bin_bailarina = cv2.morphologyEx(bin_bailarina, cv2.MORPH_OPEN, kernel, iterations=1)

# obtener esqueleto
skel_bailarina = obtener_esqueleto(bin_bailarina)


# ---------- MOSTRAR RESULTADOS ----------

cv2.imshow("Bailarina Original", img_bailarina)
cv2.imshow("Bailarina Binaria", bin_bailarina)
cv2.imshow("Esqueleto Bailarina", skel_bailarina)


cv2.waitKey(0)
cv2.destroyAllWindows()