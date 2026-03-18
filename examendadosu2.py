import cv2
import os

RUTA_IMGS = r'totaldados'
RUTA_TXT  = r'anotacionesparadados'

def leer_txt(ruta_txt, ancho, alto):
    datos = []

    with open(ruta_txt, 'r') as f:
        lineas = f.readlines()

    for linea in lineas:
        partes = linea.strip().split()

        if len(partes) != 5:
            continue

        clase = int(partes[0])
        x = float(partes[1])
        y = float(partes[2])
        w = float(partes[3])
        h = float(partes[4])

        x = int(x * ancho)
        y = int(y * alto)
        w = int(w * ancho)
        h = int(h * alto)

        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)

        valor = clase + 1

        datos.append((x1, y1, x2, y2, valor))

    return datos

for archivo in os.listdir(RUTA_IMGS):

    if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):

        nombre = os.path.splitext(archivo)[0]

        ruta_img = os.path.join(RUTA_IMGS, archivo)
        ruta_txt = os.path.join(RUTA_TXT, nombre + '.txt')

        if not os.path.exists(ruta_txt):
            print(f"No hay txt para: {archivo}")
            continue

        img = cv2.imread(ruta_img)

        if img is None:
            print(f"No se pudo cargar: {archivo}")
            continue

        h, w = img.shape[:2]

        datos = leer_txt(ruta_txt, w, h)

        total = 0

        for (x1, y1, x2, y2, valor) in datos:
            total += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, str(valor), (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,255,0), 3)

        print(f"{archivo} -> Dados: {total}")
        escala = 800 / w
        img_show = cv2.resize(img, (800, int(h * escala)))

        cv2.imshow("Resultado", img_show)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()