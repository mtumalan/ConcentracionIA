import numpy as np

def initCentroides(k, data):
    '''
    Inicializa los centroides de manera aleatoria

    Parámetros
    k : int
    data : np.array

    Salida
    centroides : np.array
    '''
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroides = data[indices]
    return centroides

def asignarClusters(data, centroides):
    '''
    Asigna un cluster a cada punto

    Parámetros
    data : np.array
    centroides : np.array

    Salida
    clusters : np.array
    '''
    # Calcula la distancia de cada punto a todos los centroides
    distancias = np.sqrt(((data[:, np.newaxis, :] - centroides[np.newaxis, :, :])**2).sum(axis=2))
    # Asigna el cluster más cercano a cada punto
    clusters = np.argmin(distancias, axis=1)
    return clusters

def actualizarCentroides(data, clusters, k):
    '''
    Actualiza los centroides

    Parámetros
    data : np.array
    clusters : np.array
    k : int

    Salida
    centroides : np.array
    '''
    # Calcula el nuevo centroide de cada cluster
    centroides = np.zeros((k, data.shape[1]))
    for i in range(k):
        # Calcula la media de los puntos asignados a cada cluster
        centroides[i] = np.mean(data[clusters == i], axis=0)
    return centroides

def kmeans(data, k, max_iter=100):
    '''
    Ejecuta el algoritmo k-means

    Parámetros
    data : np.array
    k : int
    max_iter : int

    Salida
    clusters : np.array
    centroides : np.array
    '''

    # Inicializar centroides
    centroides = initCentroides(k, data)
    for i in range(max_iter):
        # Asignar clusters a cada punto
        clusters = asignarClusters(data, centroides)

        # Actualizar centroides
        centroides_new = actualizarCentroides(data, clusters, k)

        # Verificar si los centroides han cambiado
        if np.all(centroides == centroides_new):
            break

        # Actualizar centroides
        centroides = centroides_new

    return clusters, centroides

def funcionObjetivo(data, clusters, centroides):
    '''
    Calcula la función objetivo

    Parámetros
    data : np.array
    clusters : np.array
    centroides : np.array

    Salida
    suma : float
    '''
    suma = 0
    for i in range(len(data)):
        # Calcula la distancia al cuadrado de cada punto a su centroide
        suma += np.sum((data[i] - centroides[clusters[i]])**2)
    return suma / len(data)

if __name__ == '__main__':
    # Cargar datos
    x = np.loadtxt('/home/mtumalan/Desktop/Repos/ConcentracionIA/Vic(Mate)/kmeans/ex7data2.txt') 

    # Ejecutar k-means
    clusters, centroides = kmeans(x, 3)
    print(funcionObjetivo(x, clusters, centroides))

    # Graficar
    import matplotlib.pyplot as plt
    plt.scatter(x[:, 0], x[:, 1], c=clusters)
    plt.scatter(centroides[:, 0], centroides[:, 1], c='red', marker='x')
    plt.show()