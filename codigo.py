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
# (a) Realizar una funcion en python que dado los datos de las imagenes de entrenamiento 
# y una fila, grafique la imagen guardada en esa fila y en el tıtulo del grafico se 
# indique a que numero corresponde, es decir su clasificacion. Usar la funcion imshow() de pyplot.
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
# (b) ¿Cuantas imagenes hay por cada dıgito en el conjunto de entrenamiento? ¿Y en el conjunto
# de testeo?
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
# (c) Para las primeras 2.000 imagenes del conjunto de entrenamiento realizar una funcion en python
# que devuelva la imagen promedio de cada uno de los dıgitos.
#-----------------------------------------------------------------------------

imagenes = []   #guardaremos las imagenes en un array para luego graficarlas

# aclaracion: las imagenes promedio guardaran el numero que representan en la posicion [0],
# para graficarlas habra que omitir el primer elemento del array

for n in range(0,10):
    df = train[train[0] == n].iloc[:2000,:]    #creamos df unicamente con las imagenes del numero n
    imagenes_n = df.to_numpy()    #convertimos el df en un array bidimensional de numpy
    imagen_promedio = np.mean(imagenes_n,axis=0)  # .mean() calcula el promedio de todas las imagenes que se encuentran como filas de la matriz 'imagenes_n'
    imagenes.append(imagen_promedio)
    globals()['imagen_'+str(n)] = imagen_promedio   # asignamos la imagen promedio de cada numero 'n' a una variable llamada 'imagen_n' 


#-----------------------------------------------------------------------------
# (d) Graficar cada una de las imagenes promedio obtenidas.
#-----------------------------------------------------------------------------

def graficar_imagenes():
    for imagen in imagenes:
        plt.imshow(imagen[1:].reshape((28,28)),cmap='gray')
        plt.show()


#==============================================================================
# EJERCICIO 2
#==============================================================================

#-----------------------------------------------------------------------------
# (a) Realizar una funcion en python que dadas las imagenes promedio del ejercicio 2(c), calcule la
# menor distancia Euclıdea entre todos los dıgitos y cada una de las primeras 200 imagenes de
# testeo. La funcion debe devolver un arreglo con las 200 predicciones.
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
        imagen_i = np.array(df.iloc[i,:])   # se recorren las imagenes 
        prediccion_i = prediccion(imagenes,imagen_i)  # se realiza la prediccion de la imagen actual
        predicciones.append(prediccion_i)
    return predicciones



#-----------------------------------------------------------------------------
# (b) Realizar una funcion en python que tome el arreglo de predicciones anteriores y evalue si es
# correcta o no la prediccion. Debe devolver la precision en la prediccion. Se define la precision
# como:
#  precision = Σ(Casos acierto) / Σ(Casos totales)
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
# (c) Graficar un par de casos de imagenes de testeo en los cuales no se haya acertado. ¿Considera
# buena la precision?
#-----------------------------------------------------------------------------

def imagenes_no_acertadas(df,imagenes):
    df = df.iloc[:200,:]
    predicciones = np.array(prediccion_200(df,imagenes))    # array de las 200 predicciones
    valores_posta = np.array(df.iloc[:,0])  # array de los valores reales de cada imagen
    no_acertadas = pd.DataFrame(predicciones == valores_posta)     
    indices_imagenes_no_acertadas = no_acertadas[no_acertadas[0] == False].index
    return indices_imagenes_no_acertadas


def graficar_num_no_acertado():
    indices_imagenes_no_acertadas = imagenes_no_acertadas(test,imagenes)
    #generamos numero random para graficar alguna de las imagenes no acertadas
    r = np.random.randint(0,len(indices_imagenes_no_acertadas))
    graficar(test,indices_imagenes_no_acertadas[r]) 
    print('Numero: ',test.iloc[indices_imagenes_no_acertadas[r],0])
    print('Indice: ',indices_imagenes_no_acertadas[r])


#==============================================================================
# EJERCICIO 3
#==============================================================================

# Implemetar una funcion en Python que dada una matriz A halle la descomposicion SVD de A, por
# el metodo de la potencia.
# Llamamos descomposicion SVD en valores singulares a:
# A = U ΣV T Implemetar una funcion en Python que dada una matriz A halle la descomposicion SVD de A, por
# el metodo de la potencia.

def svd(A):



#==============================================================================
# EJERCICIO 4
#==============================================================================

# Se utilizara la descomposicion SVD para resolver la clasificacion de imagenes correspondiente a
# numeros manuscritos.

#-----------------------------------------------------------------------------
# (a) Tomar las primeras 2.000 imagenes del conjunto de imagenes de testeo y ordenarlas segun el
# dıgito al que corresponde de 0 a 9. Obtener 10 matrices correspondientes a cada dıgito. Estas
# matrices deben tener una dimension de 785 × cantidad imagenes, puede no haber la misma
# cantidad de imagenes para cada dıgito en las primeras 2.000 imagenes. Recordar que la primer
# columna es la clasificacion. Finalmente obtener Mi=0,...,9 matrices de 784 × cantidad imagenes
# quitando la primer columna. Se pueden guardar las matrices en un arreglo de tipo lista donde
# cada ıtem de la lista se corresponde con una matriz Mi y la posicion hace referencia al dıgito
# que representan.
#-----------------------------------------------------------------------------

test_2000 = test.iloc[:2000,:]

lista_matrices = []
for n in range(0,10):
    #obtengo matrices para cada numero
    matriz_n = test_2000[test_2000[0] == n].iloc[:,1:]  # se le saca la primer columna
    lista_matrices.append(matriz_n)


#-----------------------------------------------------------------------------
# (b) Realizar la descomposicion SVD de cada una de las matrices Mi utilizando la funcion creada
# en el ejercicio (3). Para ello realizar una funcion en Python que tome la lista de matrices Mi
# y devuelva en 3 listas la solucion de la descomposicion, es decir Ui, Σi y Vi.
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
# (c) Las columnas de Ui son combinacion lineal del espacio columna de Mi. Teniendo esto presente
# tomar la primer columna de cada Ui y graficarla como imagen, es decir convertir a una matriz
# de 28 × 28 y graficar. Explique que representa.
#-----------------------------------------------------------------------------



#-----------------------------------------------------------------------------
# (d) Repetir el ıtem anterior pero para las columnas 2 y 3 de cada una de las Ui. Comparar con lo
# obtenido en (c) y explicar las diferencias.
#-----------------------------------------------------------------------------
