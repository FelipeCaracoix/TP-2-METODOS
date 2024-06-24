from descxgrad import *
# Define a range of alpha values to test
alpha_values = [0.001, 0.01, 0.05, 0.1, 0.5]

for i in range(alpha_values):
    # Cargar imágenes y datos
    images, d = cargar_datos("/Users/nicolasfranke/Desktop/DITELLA/Métodos Computacionales/TPs/chest_xray/test/ALL", escala=32)

    b = np.random.randn(1)
    w = np.random.randn(images[0].shape[0])
    #print(balancear_datos(images,d))


    errors = error_cuadratico_medio(images,w,b,d)
    print(errors)


    plt.plot(errors, label='Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.title('Decreasing Error over Iterations')
    plt.legend()
    plt.show()