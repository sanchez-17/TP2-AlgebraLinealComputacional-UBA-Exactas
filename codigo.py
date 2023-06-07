'''
TP2 ALC
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# leemos los .csv

train = pd.read_csv('mnist_train.csv',names=np.linspace(0,784,785))
test = pd.read_csv('mnist_test.csv',names=np.linspace(0,784,785))


#==============================================================================
# EJERCICIO 1
#==============================================================================

#-----------------------------------------------------------------------------
#(a) Realizar una funcion en python que dado los datos de las imagenes de entrenamiento 
#y una fila, grafique la imagen guardada en esa fila y en el tıtulo del grafico se 
#indique a que numero corresponde, es decir su clasificacion. Usar la funcion imshow() de pyplot.
#-----------------------------------------------------------------------------

def graficar(df,fila):
    plt.imshow(np.array(df.iloc[fila,1:]).reshape((28,28)),cmap='gray')
    numero = str(df.iloc[fila,0])
    plt.title(numero)
    plt.show()

#prueba:
fila = 1
df = train
#graficar(df,fila)

#-----------------------------------------------------------------------------
#(b) ¿Cuantas imagenes hay por cada dıgito en el conjunto de entrenamiento? ¿Y en el conjunto
#de testeo?
#-----------------------------------------------------------------------------

cantidad_de_imagenes_por_numero_train = train[0].value_counts().sort_index()

# las cantidades son:

#0    5923
#1    6742
#2    5958
#3    6131
#4    5842
#5    5420
#6    5918
#7    6265
#8    5851
#9    5949

cantidad_de_imagenes_por_numero_test = test[0].value_counts().sort_index()

# las cantidades son:

#0     980
#1    1135
#2    1032
#3    1010
#4     982
#5     892
#6     958
#7    1027
#8     974
#9    1009

#-----------------------------------------------------------------------------
#(c) Para las primeras 2.000 imagenes del conjunto de entrenamiento realizar una funcion en python
#que devuelva la imagen promedio de cada uno de los dıgitos.
#-----------------------------------------------------------------------------

imagenes = []   #guardaremos las imagenes en un array para luego graficarlas

# aclaracion: las imagenes promedio guardaran el numero que representan en la posicion [0],
# para graficarlas habra que omitir el primer elemento del array

for n in range(0,10):
    df = train[train[0] == n].iloc[:2000,:]    #creamos df unicamente con las imagenes del numero n
    imagenes_n = df.to_numpy()    #convertimos el df en un array bidimensional de numpy
    imagen_promedio = np.mean(imagenes_n,axis=0)  # .mean() calcula el promedio de todas las imagenes
    imagenes.append(imagen_promedio)
    globals()['imagen_'+str(n)] = imagen_promedio


#-----------------------------------------------------------------------------
#(d) Graficar cada una de las imagenes promedio obtenidas.
#-----------------------------------------------------------------------------

def graficar_imagenes():
    for imagen in imagenes:
        plt.imshow(imagen[1:].reshape((28,28)),cmap='gray')
        plt.show()


#==============================================================================
# EJERCICIO 2
#==============================================================================

#-----------------------------------------------------------------------------
#(a) Realizar una funcion en python que dadas las imagenes promedio del ejercicio 2(c), calcule la
#menor distancia Euclıdea entre todos los dıgitos y cada una de las primeras 200 imagenes de
#testeo. La funcion debe devolver un arreglo con las 200 predicciones.
#-----------------------------------------------------------------------------

# la funcion ditancia() toma dos imagenes (np.array de tamaño 784) y calcula distancia euclidea en R^784

def distancia(imagen1,imagen2):
    distancia=0
    for i in range(0,784):
        distancia+=np.sqrt((imagen1[i]-imagen2[i])**2)
    return distancia

# la funcion prediccion() toma la lista de promedios de las imagenes del 0 al 9, y una imagen a testear 
# la array a testear debe tener en la posicion [0] el numero de la imagen: tamaño de 785
# devolvera un float

def prediccion(imagenes,imagen_test):
    prediccion=imagen_0
    for imagen in imagenes:
        # se le saca el primer elemento al array (que indica el numero de la imagen)
        if distancia(imagen[1:],imagen_test[1:]) <= distancia(prediccion[1:],imagen_test[1:]):
            prediccion=imagen
    return prediccion[0]

# la funcion prediccion_200() toma un df (test) y la lista de promedios de las imagenes
# devolvera una lista de 200 predicciones, de las primeras 200 imagenes del dataframe

def prediccion_200(df,imagenes):
    df = df.iloc[:200,:]
    predicciones=[]
    for i in range(0,200):
        prediccion = imagen_0
        imagen_i = np.array(df.iloc[i,:])
        prediccion = prediccion(imagenes,imagen_i)
        predicciones.append(prediccion)
    return predicciones



#-----------------------------------------------------------------------------
#(b) Realizar una funcion en python que tome el arreglo de predicciones anteriores y evalue si es
#correcta o no la prediccion. Debe devolver la precision en la prediccion. Se define la precision
#como:
# precision = Σ(Casos acierto) / Σ(Casos totales)
#-----------------------------------------------------------------------------

def precision(df,imagenes):
    df = df.iloc[:200,:]
    predicciones = np.array(prediccion_200(df,imagenes))    # array de las 200 predicciones
    valores_posta = np.array(df.iloc[:,0])  # array de los valores reales de cada imagen
    # (predicciones == valores_posta) es un array de booleanos, contamos los valores 'True', 
    # es decir las coincidencias entre los array
    aciertos = pd.DataFrame(predicciones == valores_posta).value_counts()[True]     
    return aciertos/200

    
    


#-----------------------------------------------------------------------------
#(c) Graficar un par de casos de im ́agenes de testeo en los cuales no se haya acertado. ¿Considera
#buena la precisi ́on?
#-----------------------------------------------------------------------------



