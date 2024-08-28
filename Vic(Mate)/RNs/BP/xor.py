import numpy as np

# Funciones de activación y sus derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialización de datos y parámetros
# XOR: entrada y salida esperada
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

# Parámetros de la red
input_neurons = 2
hidden_neurons = 2
output_neurons = 1
learning_rate = 0.1
epochs = 10000

# Inicialización aleatoria de los pesos
input_hidden_weights = np.random.uniform(size=(input_neurons, hidden_neurons))
hidden_output_weights = np.random.uniform(size=(hidden_neurons, output_neurons))

# Bias
hidden_bias = np.random.uniform(size=(1, hidden_neurons))
output_bias = np.random.uniform(size=(1, output_neurons))

# Entrenamiento con retropropagación
for epoch in range(epochs):
    # Forward Propagation
    hidden_layer_input = np.dot(inputs, input_hidden_weights) + hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, hidden_output_weights) + output_bias
    predicted_output = sigmoid(output_layer_input)

    # Cálculo del error
    error = expected_output - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(hidden_output_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Actualización de los pesos y bias
    hidden_output_weights += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    input_hidden_weights += inputs.T.dot(d_hidden_layer) * learning_rate
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# Resultados finales
print("Predicción después de entrenamiento: \n", predicted_output)
