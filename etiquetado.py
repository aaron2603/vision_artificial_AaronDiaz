import cv2
import numpy as np
imagen = cv2.imread("tornillos.jpg", cv2.IMREAD_GRAYSCALE)
_, imagen_bin = cv2.threshold(imagen, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Bin", imagen_bin)
cv2.waitKey(0)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_bin, 8, ltype=cv2.CV_32S)
print(f"Etiquetas_n: {num_labels} Etiquetas: {labels} Estadísticos: {stats} Centroides: {centroids}")

for i in range(1, num_labels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]
    if area > 100:
        cv2.rectangle(imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(imagen, (int(cX), int(cY)), 4, (0, 0, 255), -1)
cv2.imshow('ImagenEtiquetada', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()