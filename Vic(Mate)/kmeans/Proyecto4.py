import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def kMeansInitCentroids(X, K):
    '''
    Inicializa los centroides de manera aleatoria
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
    Actualiza los centroides
    '''
    centroides = np.array([X[idx == k].mean(axis=0) if np.any(idx == k) else X[np.random.randint(0, X.shape[0])] for k in range(K)])
    return centroides

def runkMeans(X, initial_centroids, max_iters=500, true=True):
    '''
    Ejecuta el algoritmo k-means

    Parámetros:
    X : np.array - Conjunto de datos
    initial_centroids : np.array - Centroides iniciales
    max_iters : int - Máximo número de iteraciones
    true : bool - Si es True, se dibuja el progreso; si es False, se devuelven clusters y centroides

    Retorna:
    clusters, centroides
    '''
    centroides = initial_centroids
    K = len(centroides)
    centroidesPass = [centroides]

    for i in range(max_iters):
        # Asignar clusters
        clusters = asignarClusters(X, centroides)
        
        # Actualizar centroides
        centroides_new = actualizarCentroides(X, clusters, K)

        # Verificar si los centroides no cambiaron
        if np.allclose(centroides, centroides_new):
            break

        centroides = centroides_new
        centroidesPass.append(centroides)

        # Si true, dibujar el progreso de la agrupación
        if true:
            plt.scatter(X[:, 0], X[:, 1], c=clusters)
            centroidesPass_np = np.array(centroidesPass)
            for j in range(K):
                plt.plot(centroidesPass_np[:, j, 0], centroidesPass_np[:, j, 1], marker='x', c='red')
            plt.show()

    return clusters, centroides

def compress_image_kmeans(image_path, K, max_iters):
    '''
    Comprime una imagen png usando k-means
    '''
    # Load image
    image = imread(image_path)
    
    # Handle RGBA images (discard alpha channel)
    if image.shape[-1] == 4:
        image = image[..., :3]
    
    # Normalize the image to [0, 1]
    image = image / 255.0
    
    # Reshape image into (m*n, 3) for k-means clustering
    X = image.reshape(-1, 3)
    
    # Initialize centroids and run K-means
    initial_centroids = kMeansInitCentroids(X, K)
    clusters, centroides = runkMeans(X, initial_centroids, max_iters, False)
    
    # Assign each pixel to its nearest centroid
    X_compressed = centroides[clusters].reshape(image.shape)
    
    # Ensure the compressed image is in the range [0, 1]
    X_compressed = np.clip(X_compressed, 0, 1)

    # Plot the original and compressed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title('Imagen Original')
    axes[0].axis('off')
    
    axes[1].imshow(X_compressed)
    axes[1].set_title('Imagen Comprimida')
    axes[1].axis('off')

    plt.show()

if __name__ == '__main__':
    compress_image_kmeans('/home/mtumalan/Desktop/Repos/Clases/ConcentracionIA/Vic(Mate)/kmeans/bird_small.png', 16, 10)