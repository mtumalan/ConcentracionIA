import numpy as np

def kMeansInitCentroids(X, K):
    '''
    Inicializa los centroides de manera aleatoria (no es necesario aquí, ya que se proporcionan los centroides)
    '''
    indices = np.random.choice(X.shape[0], K, replace=False)
    centroides = X[indices]
    return centroides

def asignarClusters(X, initial_centroids):
    '''
    Asigna un cluster a cada punto
    '''
    distancias = np.sqrt(((X[:, np.newaxis, :] - initial_centroids[np.newaxis, :, :])**2).sum(axis=2))
    clusters = np.argmin(distancias, axis=1)
    return clusters

def actualizarCentroides(X, idx, K):
    '''
    Actualiza los centroides (aunque aquí no es necesario ya que no realizamos iteraciones)
    '''
    centroides = np.array([X[idx == k].mean(axis=0) if np.any(idx == k) else X[np.random.randint(0, X.shape[0])] for k in range(K)])
    return centroides

def runkMeans(X, initial_centroids):
    '''
    Ejecuta una versión simple del algoritmo k-means, donde se asignan los puntos a los clusters dados los centroides iniciales.
    '''
    # Asignar clusters
    clusters = asignarClusters(X, initial_centroids)
    
    return clusters

# Puntos a clasificar
X = np.array([[2, 3, 5],
              [1, 3, 2],
              [6, 2, 4],
              [-1, 1, 3]])

# Centroides iniciales dados
initial_centroids = np.array([[0, 1, 1], [4, 1, 2]])

# Ejecutar K-means para asignar los puntos a clusters
clusters = runkMeans(X, initial_centroids)

# Imprimir los resultados
for i, point in enumerate(X):
    print(f"Punto {point} se asigna al cluster {clusters[i]}")