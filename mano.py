import cv2
import numpy as np

# -------- FUNCION PARA OBTENER ESQUELETO --------
def obtener_esqueleto(imagen_binaria):

    size = np.size(imagen_binaria)
    skel = np.zeros(imagen_binaria.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
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


# -------- PROCESAR IMAGEN MANO --------

img_mano = cv2.imread("mano.jpeg", cv2.IMREAD_GRAYSCALE)

# mejorar contraste
img_mano = cv2.equalizeHist(img_mano)

# suavizar para reducir ruido
img_mano = cv2.GaussianBlur(img_mano, (5,5), 0)

# binarización automática
_, bin_mano = cv2.threshold(img_mano, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# kernel morfológico
kernel = np.ones((3,3), np.uint8)

# cerrar huecos entre dedos
bin_mano = cv2.morphologyEx(bin_mano, cv2.MORPH_CLOSE, kernel, iterations=2)

# eliminar ruido pequeño
bin_mano = cv2.morphologyEx(bin_mano, cv2.MORPH_OPEN, kernel, iterations=1)

# obtener esqueleto
skel_mano = obtener_esqueleto(bin_mano)


# -------- MOSTRAR RESULTADOS --------

cv2.imshow("Mano Original", img_mano)
cv2.imshow("Mano Binaria", bin_mano)
cv2.imshow("Esqueleto Mano", skel_mano)

cv2.waitKey(0)
cv2.destroyAllWindows()