# Creacion de una red neuronal con backpropagation y feedforward
# Autor: Mauricio Tumalan Castillo
# Matricula: A01369288

import numpy as np

def sigmoidal(z):
    return 1 / (1 + np.exp(-z))

def sigmoidalGradiente(z):
    g = sigmoidal(z)
    return g * (1 - g)

def randInicializacionPesos(L_in, L_out, epsilon_init=0.12):
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

def entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_=0.1, alpha=0.1, num_iter=2000):
    W1 = randInicializacionPesos(input_layer_size, hidden_layer_size)
    W2 = randInicializacionPesos(hidden_layer_size, num_labels)
    b1 = np.zeros((hidden_layer_size, 1))
    b2 = np.zeros((num_labels, 1))

    m = X.shape[0]
    y = np.where(y == 10, 0, y)  # Reasigna el 10 al dígito 0
    y_matrix = np.eye(num_labels)[y.reshape(-1)]

    X = np.concatenate([np.ones((m, 1)), X], axis=1)  # Agregar bias unit

    for i in range(num_iter):
        # Forward propagation
        z2 = X.dot(W1.T) + b1.T
        a2 = sigmoidal(z2)
        a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)
        z3 = a2.dot(W2.T) + b2.T
        a3 = sigmoidal(z3)

        # Cálculo del costo
        J = (-1 / m) * np.sum(y_matrix * np.log(a3) + (1 - y_matrix) * np.log(1 - a3))
        J += (lambda_ / (2 * m)) * (np.sum(np.square(W1[:, 1:])) + np.sum(np.square(W2[:, 1:])))

        if i % 100 == 0:
            print(f'Iteración {i}: Costo {J:.6f}')

        # Backpropagation
        delta3 = a3 - y_matrix
        delta2 = delta3.dot(W2[:, 1:]) * sigmoidalGradiente(z2)

        Delta1 = delta2.T.dot(X)
        Delta2 = delta3.T.dot(a2)

        # Actualización de pesos
        W1[:, 1:] -= (alpha / m) * (Delta1[:, 1:] + lambda_ * W1[:, 1:])
        W1[:, 0] -= (alpha / m) * Delta1[:, 0]
        b1 -= (alpha / m) * np.sum(delta2, axis=0).reshape(b1.shape)

        W2[:, 1:] -= (alpha / m) * (Delta2[:, 1:] + lambda_ * W2[:, 1:])
        W2[:, 0] -= (alpha / m) * Delta2[:, 0]
        b2 -= (alpha / m) * np.sum(delta3, axis=0).reshape(b2.shape)

    return W1, b1, W2, b2

def prediceRNYaEntrenada(X, W1, b1, W2, b2):
    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    z2 = X.dot(W1.T) + b1.T
    a2 = sigmoidal(z2)
    a2 = np.concatenate([np.ones((a2.shape[0], 1)), a2], axis=1)

    z3 = a2.dot(W2.T) + b2.T
    a3 = sigmoidal(z3)

    return np.argmax(a3, axis=1)

def main():
    data = np.loadtxt('Vic(Mate)/redNeuronal/digitos.txt')
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    sigma[sigma == 0] = 1
    X = (X - mu) / sigma

    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10
    lambda_ = 0.05
    alpha = 0.01
    num_iter = 20000

    W1, b1, W2, b2 = entrenaRN(input_layer_size, hidden_layer_size, num_labels, X, y, lambda_, alpha, num_iter)

    predicciones = prediceRNYaEntrenada(X, W1, b1, W2, b2)
    precision = np.mean(predicciones == y) * 100
    print(f'Precisión: {precision:.2f}%')

if __name__ == '__main__':
    main()