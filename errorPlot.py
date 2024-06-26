from descxgrad import *

with open('valores_buenos.json', 'r') as archivo:
    datos = json.load(archivo)

# Accede a los valores específicos (w_estrella y b_estrella) y guárdalos en arrays
w_estrella = datos.get('w_estrella', [])
b_estrella = datos.get('b_estrella', [])

w_estrella = np.array(w_estrella)
b_estrella = np.array(b_estrella)

images, d = cargar_datos("/Users/felip/OneDrive/Escritorio/chest_xray/test/ALL", escala=128)

errores = error_cuadratico_medio(images, w_estrella, b_estrella,d)

plot_error_curve(errores, 0.0001)