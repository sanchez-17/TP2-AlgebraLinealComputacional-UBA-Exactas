'''
TP2 ALC
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# leemos los .csv

train = pd.read_csv('mnist_train.csv',names=np.linspace(0,784,785))
test = pd.read_csv('mnist_test.csv',names=np.linspace(0,784,785))

#%%
#==============================================================================
# EJERCICIO 1
#==============================================================================

#-----------------------------------------------------------------------------
# (a) Realizar una funcion en python que dado los datos de las imagenes de entrenamiento 
# y una fila, grafique la imagen guardada en esa fila y en el tıtulo del grafico se 
# indique a que numero corresponde, es decir su clasificacion. Usar la funcion imshow() de pyplot.
#-----------------------------------------------------------------------------

def graficar(df,fila):
    plt.imshow(np.array(df.iloc[fila,1:]).reshape(28,28),cmap='gray')
    numero = df.iloc[fila,0]
    plt.title(f'Numero: {numero}')
    plt.show()

#prueNumero: ',test.iloc[indices_imagenes_no_acertadas[r],0]ba:
    
fila = np.random.randint(0, len(train)) #Elegimos una imagen al azar
graficar(train,fila)

#%%
#-----------------------------------------------------------------------------
# (b) ¿Cuantas imagenes hay por cada dıgito en el conjunto de entrenamiento? ¿Y en el conjunto
# de testeo?
#-----------------------------------------------------------------------------

cantidad_de_imagenes_por_numero_train = train[0].value_counts().sort_index()
print("=========================\nConjunto de entrenamiento\n=========================")
print("Las cantidades por digito son: ")
print(cantidad_de_imagenes_por_numero_train)
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
print("=========================\nConjunto de entrenamiento\n=========================")
print("Las cantidades por digito son: ")
print(cantidad_de_imagenes_por_numero_test)
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
#%%
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
    return int(prediccion[0])

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

    # printeos extras
    numero = test.iloc[indices_imagenes_no_acertadas[r],0]
    indice = indices_imagenes_no_acertadas[r]
    predic = prediccion(imagenes,np.array(test.iloc[indices_imagenes_no_acertadas[r],:]))
    print('Numero: ',numero)
    print('Indice: ',indice)
    print('Prediccion: ',predic)
    print('Distancia al ',numero,' (valor real): ',distancia(imagenes[numero][1:],np.array(test.iloc[indice,1:])))
    print('Distancia al ',predic,' (prediccion): ',distancia(imagenes[predic][1:],np.array(test.iloc[indice,1:])))

    graficar(test,indices_imagenes_no_acertadas[r]) 


#==============================================================================
# EJERCICIO 3
#==============================================================================

# Implemetar una funcion en Python que dada una matriz A halle la descomposicion SVD de A, por
# el metodo de la potencia.
# Llamamos descomposicion SVD en valores singulares a:
# A = U ΣV T Implemetar una funcion en Python que dada una matriz A halle la descomposicion SVD de A, por
# el metodo de la potencia.

def metodo_potencia(A,x0,e):
    x_i = x0 / np.linalg.norm(x0)
    error = 0.000001
    #iteraciones=0
    while error < (1-e):
        x_j = x_i
        x_i = np.dot(A,x_i)
        x_i = x_i / np.linalg.norm(x_i)
        error = np.dot(x_i,x_j)
        #iteraciones+=1

    # printeos para testear que hace
    #print('x_i: ',x_i)
    #print('x_j: ',x_j)
    #print('Error: ',error)
    #print('Iteraciones: ',iteraciones)
    return x_i

def minima_dim(A):
    return min(len(A[0]),len(A))

def svd(A):
    
    # sea NxN la dimension de B, generamos vector x0 de dimension N
    x0 = np.random.random(len(A[0]))
    x0 = x0 / np.linalg.norm(x0)    #normalizamos x0

    # declaramos las matrices de la descomposicion:
    U = []
    S = []
    V = []

    # Sea MxN la dimension de A, nececito el minimo entre M y N para saber la cantidad
    # de iteraciones, ya que ese minimo me determina ya sea la cantidad de columnas de U o de V
    # que puedo hallar con esta funcion iterativa
    m = minima_dim(A)
    
    for i in range(1,m+1):
        # generamos B = At*A
        B = np.dot(np.transpose(A),A)

        # obtenemos cada u_i, v_i, y s_i(valor singular) aplicando la formula dada en el trabajo
        v_i = metodo_potencia(B,x0,0.1)
        s_i = np.linalg.norm(np.dot(A,v_i))
        u_i = np.dot(A,v_i) / s_i

        # armamos el vector de m ceros con el valor singular actual en posicion i-1
        s = np.zeros(m)
        s[i-1] = s_i
         
        U.append(u_i)   # agrego los u_i como filas, cuando termine de iterar transpongo y obtengo U con u_i como columnas
        S.append(s)
        V.append(v_i)   # agrego v_i como filas, osea que la V que devuelve la funcion ya es V transpuesta

        # actualizamos A restandole s * u_i * v_i
        # se hace un reshape de los vectores para poder hacer la multiplicacion de un vector Mx1 * 1xN, y que devuelva una matriz MxN
        A = A - np.dot((s_i*u_i).reshape(len(u_i),1),v_i.reshape(1,len(v_i)))
    
    U = np.transpose(U)

    return U,np.array(S),np.array(V)
        


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

def svd_Mi(lista_matrices):
    Ui = []
    Si = []
    Vi = []
    for matriz in lista_matrices:
        u_i,s_i,v_i = svd(np.array(matriz))     # cada matriz en la lista es un DF, lo pasamos a np.array
        Ui.append(u_i)
        Si.append(s_i)
        Vi.append(v_i)
    return Ui,Si,Vi



#-----------------------------------------------------------------------------
# (c) Las columnas de Ui son combinacion lineal del espacio columna de Mi. Teniendo esto presente
# tomar la primer columna de cada Ui y graficarla como imagen, es decir convertir a una matriz
# de 28 × 28 y graficar. Explique que representa.
#-----------------------------------------------------------------------------

# la funcion graficara la primer columna de cada Ui para la SVD de las 10 matrices

def graficar_u1(Ui):
    for ui in Ui:
        plt.imshow(np.array(df.iloc[fila,1:]).reshape((28,28)),cmap='gray')

#-----------------------------------------------------------------------------
# (d) Repetir el ıtem anterior pero para las columnas 2 y 3 de cada una de las Ui. Comparar con lo
# obtenido en (c) y explicar las diferencias.
#-----------------------------------------------------------------------------
