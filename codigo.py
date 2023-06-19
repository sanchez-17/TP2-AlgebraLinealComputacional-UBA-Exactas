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

def graficar(df,fila,pred = None):
    plt.imshow(np.array(df.iloc[fila,1:]).reshape(28,28),cmap='gray')
    numero = df.iloc[fila,0]
    plt.title(f'Numero: {numero}')
    if pred :
        plt.figtext(0.5, 0.01, f'prediccion: {pred}', ha='center', fontsize=10)
    plt.axis('off')
    plt.show()

#prueNumero: ',test.iloc[indices_imagenes_no_acertadas[r],0]ba:
    
#fila = np.random.randint(0, len(train)) #Elegimos una imagen al azar
#graficar(train,fila)

#%%
#-----------------------------------------------------------------------------
# (b) ¿Cuantas imagenes hay por cada dıgito en el conjunto de entrenamiento? ¿Y en el conjunto
# de testeo?
#-----------------------------------------------------------------------------

cantidad_de_imagenes_por_numero_train = train[0].value_counts().sort_index()

#print("=========================\nConjunto de entrenamiento\n=========================")
#print("Las cantidades por digito son: ")
#print(cantidad_de_imagenes_por_numero_train)
#print()

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

#print("=========================\nConjunto de testeo\n=========================")
#print("Las cantidades por digito son: ")
#print(cantidad_de_imagenes_por_numero_test)
#print()

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

# Muestro resultados en una sola tabla para una mejor comparacion
#df_train = pd.DataFrame({'Conjunto de entrenamiento': cantidad_de_imagenes_por_numero_train})
#df_test = pd.DataFrame({'Conjunto de testeo': cantidad_de_imagenes_por_numero_test})
#df_combined = pd.concat([df_train, df_test], axis=1)
#df_combined.index.name = 'Dígito'
#print("====================================\nConjunto de entrenamiento y testeo\n====================================")
#print("Las cantidades por dígito son: \n")
#print(df_combined)
#print()

#%%
#-----------------------------------------------------------------------------
# (c) Para las primeras 2.000 imagenes del conjunto de entrenamiento realizar una funcion en python
# que devuelva la imagen promedio de cada uno de los dıgitos.
#-----------------------------------------------------------------------------

imagenes_prom = []   #guardaremos las imagenes en un array para luego graficarlas

# aclaracion: las imagenes promedio guardaran el numero que representan en la posicion [0],
# para graficarlas habra que omitir el primer elemento del array

for n in range(0,10):
    imagen_promedio = np.array(train[train[0]==n].iloc[:2000,:].mean())
    imagenes_prom.append(imagen_promedio)
    globals()['imagen_'+str(n)] = imagen_promedio   # asignamos la imagen promedio de cada numero 'n' a una variable llamada 'imagen_n' 

#    df_n = train[train[0] == n].iloc[:2000,:]    #creamos df unicamente con las imagenes del numero n
#    imagenes_n = df_n.to_numpy()    #convertimos el df en un array bidimensional de numpy
#    imagen_promedio = np.mean(imagenes_n,axis=0)  # .mean() calcula el promedio de todas las imagenes que se encuentran como filas de la matriz 'imagenes_n'

#%%
#-----------------------------------------------------------------------------
# (d) Graficar cada una de las imagenes promedio obtenidas.
#-----------------------------------------------------------------------------

def graficar_imagenes():
    for imagen in imagenes_prom:
        plt.imshow(imagen[1:].reshape(28,28),cmap='gray')
        plt.show()

#graficar_imagenes()
#%%
#==============================================================================
# EJERCICIO 2
#==============================================================================

#-----------------------------------------------------------------------------
# (a) Realizar una funcion en python que dadas las imagenes promedio del ejercicio 2(c), calcule la
# menor distancia Euclıdea entre todos los dıgitos y cada una de las primeras 200 imagenes de
# testeo. La funcion debe devolver un arreglo con las 200 predicciones.
#-----------------------------------------------------------------------------

# la funcion distancia() toma dos imagenes (np.array de tamaño 784) y calcula distancia euclidea en R^784
"""
def distancia(imagen1,imagen2):
    distancia=0
    for i in range(0,784):
        distancia+=np.sqrt((imagen1[i]-imagen2[i])**2)
    return distancia
"""

def distancia(img1,img2):
    return np.linalg.norm(img1-img2)

# la funcion prediccion() toma la lista de promedios de las imagenes del 0 al 9, y una imagen a testear 
# la array a testear debe tener en la posicion [0] el numero de la imagen: tamaño de 785
# devolvera un float

def prediccion(imagenes,imagen_test):
    prediccion=imagenes[0] #imagen_0, a mi me tira error por ser variable global
    for imagen in imagenes_prom:
        # se le saca el primer elemento al array (que indica el numero de la imagen)
        if distancia(imagen[1:],imagen_test[1:]) <= distancia(prediccion[1:],imagen_test[1:]):
            prediccion=imagen
    return int(prediccion[0])

def prediccion_aux(imagenes,imagen_test):
    prediccion=imagenes[0]
    for imagen in imagenes_prom:
        # se le saca el primer elemento al array (que indica el numero de la imagen)
        if distancia(imagen[1:],imagen_test) <= distancia(prediccion[1:],imagen_test):
            prediccion=imagen
    return int(prediccion[0])
    

# la funcion prediccion_200() toma un df (test) y la lista de promedios de las imagenes
# devolvera una lista de 200 predicciones, de las primeras 200 imagenes del dataframe

def prediccion_200_Aux(df_test,imagenes_p):
    df = df_test.iloc[:200,test.columns[1:]].values
    predicciones = []
    for i in range(200):
        imgTest_i = df[i]
        pred_i = prediccion_aux(imagenes_p,imgTest_i)
        predicciones.append(pred_i)
    return predicciones

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
   

def precision_aux(df_test, imagenes_p):
    predicciones = prediccion_200_Aux(df_test,imagenes_p)
    y_test = df_test.iloc[:200,0].values #.values para pasar a array numpy
    aciertos = sum(predicciones == y_test)
    return aciertos/200

#print("Precision: ", precision_aux(test,imagenes_prom))
#%%
#-----------------------------------------------------------------------------
# (c) Graficar un par de casos de imagenes de testeo en los cuales no se haya acertado. ¿Considera
# buena la precision?
#-----------------------------------------------------------------------------


#=============================================================
#Gaston
def imgs_no_acertadas(df_test,imagenes_p):
    df_test = df_test.iloc[:200,:]
    predicciones = prediccion_200_Aux(df_test,imagenes_p)
    y_test = df_test.iloc[:200,0].values
    indices = predicciones != y_test
    res = df_test.iloc[indices,:] 
    res = res.reset_index(drop=True) #reseteo los indices, pues ya no son continuos
    return res

def graficar_alguna_img_sin_acertar():
    imgs = imgs_no_acertadas(test,imagenes_prom)
    cant_imgs = len(imgs)
    i = np.random.randint(0, cant_imgs)
    
    img_no_acert =imgs.iloc[i,1:]
    pred = prediccion(imagenes_prom,img_no_acert)
    
    graficar(imgs,i)
    print("prediccion: ",pred)

#=============================================================

def imagenes_no_acertadas(df,imagenes):
    df = df.iloc[:200,:]
    predicciones = np.array(prediccion_200(df,imagenes))    # array de las 200 predicciones
    valores_posta = np.array(df.iloc[:,0])  # array de los valores reales de cada imagen
    no_acertadas = pd.DataFrame(predicciones == valores_posta)     
    indices_imagenes_no_acertadas = no_acertadas[no_acertadas[0] == False].index
    return indices_imagenes_no_acertadas


def graficar_num_no_acertado():
    indices_imagenes_no_acertadas = imagenes_no_acertadas(test,imagenes_prom)
    #generamos numero random para graficar alguna de las imagenes no acertadas
    r = np.random.randint(0,len(indices_imagenes_no_acertadas))

    # printeos extras
    numero = test.iloc[indices_imagenes_no_acertadas[r],0]
    indice = indices_imagenes_no_acertadas[r]
    predic = prediccion(imagenes_prom,np.array(test.iloc[indices_imagenes_no_acertadas[r],:]))
    print('Numero: ',numero)
    print('Indice: ',indice)
    print('Prediccion: ',predic)
    print('Distancia al ',numero,' (valor real): ',distancia(imagenes_prom[numero][1:],np.array(test.iloc[indice,1:])))
    print('Distancia al ',predic,' (prediccion): ',distancia(imagenes_prom[predic][1:],np.array(test.iloc[indice,1:])))

    graficar(test,indices_imagenes_no_acertadas[r]) 

#graficar_alguna_img_sin_acertar()
#%%

#==============================================================================
# EJERCICIO 3
#==============================================================================

# Implemetar una funcion en Python que dada una matriz A halle la descomposicion SVD de A, por
# el metodo de la potencia.
# Llamamos descomposicion SVD en valores singulares a:
# A = U ΣV T 


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
        v_i = metodo_potencia(B,x0,0.00000000000000001)
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
    S = np.array(S)
    V = np.array(V)

    return U,S,V
        

#%%
#==============================================================================
# EJERCICIO 4
#==============================================================================

# Se utilizara la descomposicion SVD para resolver la clasificacion de imagenes correspondiente a
# numeros manuscritos.

#-----------------------------------------------------------------------------
# (a) Tomar las primeras 2.000 imagenes del conjunto de imagenes de entrenamiento y ordenarlas segun el
# dıgito al que corresponde de 0 a 9. Obtener 10 matrices correspondientes a cada dıgito. Estas
# matrices deben tener una dimension de 785 × cantidad imagenes, puede no haber la misma
# cantidad de imagenes para cada dıgito en las primeras 2.000 imagenes. Recordar que la primer
# columna es la clasificacion. Finalmente obtener Mi=0,...,9 matrices de 784 × cantidad imagenes
# quitando la primer columna. Se pueden guardar las matrices en un arreglo de tipo lista donde
# cada ıtem de la lista se corresponde con una matriz Mi y la posicion hace referencia al dıgito
# que representan.
#-----------------------------------------------------------------------------

train_2000 = train.iloc[:2000,:]

lista_matrices = []

for n in range(0,10):
    #obtengo matrices para cada numero
    matriz_n = train_2000[train_2000[0] == n].iloc[:,1:]  # se le saca la primer columna
    lista_matrices.append(np.transpose(np.array(matriz_n)))

#%%
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
        u_i,s_i,v_i = svd(matriz)     # cada matriz en la lista es un DF, lo pasamos a np.array
        Ui.append(u_i)
        Si.append(s_i)
        Vi.append(v_i)
    return Ui,Si,Vi

#%%
#-----------------------------------------------------------------------------
# (c) Las columnas de Ui son combinacion lineal del espacio columna de Mi. Teniendo esto presente
# tomar la primer columna de cada Ui y graficarla como imagen, es decir convertir a una matriz
# de 28 × 28 y graficar. Explique que representa.
#-----------------------------------------------------------------------------

# la funcion graficara la primer columna de cada Ui para la SVD de las 10 matrices

def graficar_u1(lista_matrices):
    Ui,Si,Vi = svd_Mi(lista_matrices)
    for digito,ui in enumerate(Ui):
        plt.imshow(np.transpose(ui)[0].reshape((28,28)),cmap='gray')
        plt.title(f'Columna u1 del {digito}')
        plt.axis('off')
        plt.subplots_adjust(top=0.90)
        plt.suptitle(f"Digito: {digito}", fontsize=15)
        plt.show()
        
graficar_u1(lista_matrices)
#%%
#-----------------------------------------------------------------------------
# (d) Repetir el ıtem anterior pero para las columnas 2 y 3 de cada una de las Ui. Comparar con lo
# obtenido en (c) y explicar las diferencias.
#-----------------------------------------------------------------------------

def graficar_u2_u3(lista_matrices):
    Ui,Si,Vi = svd_Mi(lista_matrices)
    #i=0
    for digito,ui in enumerate(Ui):
        plt.subplot(1,2,1)
        plt.imshow(np.transpose(ui)[1].reshape((28,28)),cmap='gray')
        plt.title('Columna u2')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(np.transpose(ui)[2].reshape((28,28)),cmap='gray')
        plt.title('Columna u3')
        plt.axis('off')
        plt.subplots_adjust(top=0.99,hspace=0.4)
        plt.suptitle(f"Digito: {digito}", fontsize=15)
        plt.show()
        #i+=1

graficar_u2_u3(lista_matrices)
#%%-----------------------------------------------------------------------------

def comparar_promedio_svd(lista_matrices,imagenes_prom):
    Ui,Si,Vi = svd_Mi(lista_matrices)
    for i in range(0,10):
        imagen_promedio = imagenes_prom[i][1:].reshape((28,28))
        imagen_u1 = np.transpose(Ui[i])[0].reshape((28,28))
        plt.subplot(1,2,1)
        plt.imshow(imagen_promedio,cmap='gray')
        plt.title('imagen promedio')
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(imagen_u1,cmap='gray')
        plt.title('imagen u1')
        plt.axis('off')
        plt.show()

comparar_promedio_svd(lista_matrices,imagenes_prom)
#%%
#-----------------------------------------------------------------------------
# (e)
#• Sea ˆUi,k ∈ R784×k la matriz que resulta de tomar las primeras k columnas de Ui (hacerlo
#para k de 1 a 5).
#• Obtener la matriz ˆUi,k ˆU t
#i,k, es decir, la matriz que proyecta ortogonalmente sobre la imagen
#de ˆUi,k.
#• Obtener el residuo como: ri,k(x) = x − ˆUi,k ˆU t
#i,kx, donde x es la imagen (vectorizada) de la
#cual se quiere conocer la m´ınima distancia al subespacio generado por la imagen de ˆUi,k.
#• Guardar el ˆi cuyo residuo es el menor, es decir, ˆi = min{i : ∥ri,k(x)∥}, para i = 0, . . . , 9.
#´Esta es la predicci´on para la imagen x, en la aproximaci´on de rango k.
#• Comparar con el d´ıgito esperado.
#%%-----------------------------------------------------------------------------

# toma la lista de Ui, luego calcula el i cuyo residuo es menor
# k es la cantidad de columnas a utilizar de cada Ui
# x es la imagen a predecir su digito
def predecir(Ui,k,x):
    menor_residuo = 0
    digito_menor_residuo = 0 
    for i in range(9):
        Uik = Ui[i][:,:k-1]
        residuo_i = np.linalg.norm(x-(np.dot(Uik,np.dot(np.transpose(Uik),x))))
        if i == 0:
            menor_residuo = residuo_i
        if residuo_i < menor_residuo:
            menor_residuo = residuo_i
            digito_menor_residuo = i
    return digito_menor_residuo

def prediccion_SVD(Ui,k,x):
    residuos = []
    for i in range(10):
        Uik = Ui[i][:,:k]
        residuo_i = np.linalg.norm(x-(np.dot(Uik,np.dot(np.transpose(Uik),x))))
        residuos.append(residuo_i)
    digito_menor_residuo = np.argmin(residuos)
    return digito_menor_residuo

#%% Prueba unitaria
Ui,Si,Vi = svd_Mi(lista_matrices)
idx_random = np.random.randint(0, len(test))
x = np.array(test.iloc[idx_random,1:])
x_label = test.iloc[idx_random,0]

predicciones = []
for k in range(1,6):
    pred = prediccion_SVD(Ui,k,x)
    predicciones.append(pred)

#Elegimos la prediccion con mas frecuencia a lo largo del valor k
valores, conteos = np.unique(predicciones, return_counts=True)
idx_max_frec = np.argmax(conteos)
prediccion_img = valores[idx_max_frec]

graficar(test,idx_random,str(prediccion_img))
#%%

def prediccion_n_imgs(df_test,Ui,n,k):
    df = df_test.iloc[:n,test.columns[1:]].values
    predicciones = []
    for i in range(n):
        x = df[i]
        pred = prediccion_SVD(Ui,k,x)
        predicciones.append(pred)
    return predicciones

def precision_SVD(df_test,Ui,n,k):
    """
    Devuelve una lista con las precisiones para cada valor de 1 a k
    """
    preds_k = []
    for i in range(1,k+1):
        predicciones = prediccion_n_imgs(df_test,Ui,n,i)
        y_test = df_test.iloc[:n,0].values #.values para pasar a array numpy
        aciertos = sum(predicciones == y_test)
        preds_k.append(aciertos/n)
        print(f"precision para k={i}:{aciertos/n:.2f} para {n} imagenes de test")
    return preds_k
#%%

def graf_precisiones(df_test,Ui,n,k):
    precisiones = precision_SVD(df_test,Ui,n,k)
    x = range(1,k+1)
    plt.plot(x, precisiones,marker="o",drawstyle="steps-post")
    min_precision = np.min(precisiones)
    plt.ylim(min_precision-0.02, 1) #Limites para el eje y
    plt.xticks(x, [int(val) for val in x]) #valores enteros en eje x
    plt.title('Precision según k')
    plt.xlabel('Valores de k')
    plt.ylabel('Precision')
    plt.show()
    

k = 5
n = 200 #n imagenes de test
df_test = test
graf_precisiones(df_test,Ui,n,k)

