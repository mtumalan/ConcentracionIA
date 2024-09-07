'''
Red neuronal de Hopfield que almacena y recupera patrones de entrada basandose en imagenes

Mauricio Tumalan Castillo - A01369288
'''

import numpy as np
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize

class HopfieldNetwork(object):
    def train_weights(self, training_data):
        num_data = len(training_data)
        self.numneurons = training_data[0].shape[0]

        # Iniciar matriz de pesos
        W = np.zeros((self.numneurons, self.numneurons))
        rho = np.sum([np.sum(t) for t in training_data]) / (num_data*self.numneurons)

        # Hebbi learning
        for i in range(num_data):
            t = training_data[i] - rho
            W += np.outer(t, t)

        # Elementos diagonales iguales a 0
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data

        self.W = W
    
    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        self.num_iter = num_iter
        self.threshold = threshold
        self.asyn = asyn

        # Copiar para evitar modificar los datos originales
        datacopy = np.copy(data) 

        # Definir lista predicciones
        predicted = []
        for i in range(len(datacopy)):
            predicted.append(self._run(datacopy[i]))
        return predicted
    
    def _run(self, data):
        if self.asyn == False:
            for _ in range(self.num_iter):
                data = np.sign(self.W @ data - self.threshold)
        else:
            for _ in range(self.num_iter):
                i = np.random.randint(0, self.numneurons)
                data[i] = np.sign(self.W[i] @ data - self.threshold)
        return data
    
    def _energy(self, data):
        return -0.5 * data @ self.W @ data + np.sum(data) * self.threshold
    
    def plot_weights(self):
        plt.figure()
        plt.imshow(self.W)
        plt.colorbar()
        plt.title('W')
        plt.show()

def reshape(data):
    '''Redimensiona los datos a una matriz cuadrada'''
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def noiseInput(input, noise):
    '''Agrega ruido a la imagen de entrada'''
    noisyInput = np.copy(input)
    inv = np.random.binomial(n=1,p=noise, size=len(input))
    for i, v in enumerate(inv):
        if v == 1:
            noisyInput[i] = -noisyInput[i]
    return noisyInput

def plot(data,test,predicted,figsize=(5,6)):
    '''Grafica los datos originales, de prueba y predichos'''

    data = [reshape(d) for d in data]
    test = [reshape(t) for t in test]
    predicted = [reshape(p) for p in predicted]

    fig, ax = plt.subplots(9, 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            ax[i, 0].set_title('Train data')
            ax[i, 1].set_title("Input data")
            ax[i, 2].set_title('Output data')

        ax[i, 0].imshow(data[i])
        ax[i, 0].axis('off')
        ax[i, 1].imshow(test[i])
        ax[i, 1].axis('off')
        ax[i, 2].imshow(predicted[i])
        ax[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig('hopfield60.png')
    #plt.show()

def preprocessing(img, w, h):
    img = resize(img, (w, h), mode='reflect')

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1

    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    # Carga de datos
    img1 = rgb2gray(skimage.data.astronaut())  # RGB, necesita conversion
    img2 = skimage.data.camera()
    img3 = skimage.data.checkerboard()
    img4 = rgb2gray(skimage.data.chelsea())  # RGB, necesita conversion
    img5 = skimage.data.clock()
    img6 = skimage.data.coins()
    img7 = rgb2gray(skimage.data.hubble_deep_field())  # RGB, necesita conversion
    img8 = skimage.data.moon()
    img9 = skimage.data.horse()

    data = [img1, img2, img3, img4, img5, img6, img7, img8, img9]
    #show original images
    fig, ax = plt.subplots(1, 9, figsize=(20, 10))
    for i in range(len(data)):
        ax[i].imshow(data[i], cmap='gray')
        ax[i].axis('off')
    plt.savefig('original.png')

    # Preprocesamiento
    w, h = 128, 128
    data = [preprocessing(d, w, h) for d in data]
    
    # Crear red
    model = HopfieldNetwork()
    model.train_weights(data)

    test = [noiseInput(d, 0.2) for d in data]

    predicted = model.predict(test, threshold = 0, asyn = False)
    plot(data, test, predicted)

if __name__ == '__main__':
    main()