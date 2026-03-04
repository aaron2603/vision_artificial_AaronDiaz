import cv2
import numpy as np
import matplotlib.pyplot as plt

ruta_imagen = "tras1.jpeg"
imagen = cv2.imread(ruta_imagen)

if imagen is None:
    print("No se pudo cargar la imagen")
    exit()

imagen_yuv = cv2.cvtColor(imagen, cv2.COLOR_BGR2YUV)

canal_y_original = imagen_yuv[:, :, 0]

canal_y_ecualizado = cv2.equalizeHist(canal_y_original)

imagen_yuv[:, :, 0] = canal_y_ecualizado

imagen_ecualizada = cv2.cvtColor(imagen_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow("Imagen Original", imagen)
cv2.imshow("Imagen Ecualizada", imagen_ecualizada)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Histograma Original (Canal Y)")
plt.hist(canal_y_original.flatten(), bins=256, range=(0,255))

plt.subplot(1,2,2)
plt.title("Histograma Ecualizado (Canal Y)")
plt.hist(canal_y_ecualizado.flatten(), bins=256, range=(0,255))

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()