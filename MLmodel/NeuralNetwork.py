import numpy as np
import pandas as pd
from sklearn import preprocessing


def _init_layers(nn_architecture, seed):
    np.random.seed(seed)
    number_of_layers = len(nn_architecture)
    params_values = {}

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        params_values['W' + str(layer_idx)] = np.random.randn(
            layer_output_size, layer_input_size) * 0.1
        params_values['b' + str(layer_idx)] = np.random.randn(
            layer_output_size, 1) * 0.1
    return params_values


# define some activation functions that we will use in the future
def _sigmoid(Z):
    return 1/(1+np.exp(-Z))


def _relu(Z):
    return np.maximum(0, Z)


def _tanh(Z):
    return (np.exp(Z)-np.exp(-Z)) / (np.exp(Z)+np.exp(-Z))


def _sigmoid_backward(dA, Z):
    sig = _sigmoid(Z)
    return dA * sig * (1 - sig)


def _relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def _tanh_backward(dA, Z):
    mid = (1 - ((np.exp(Z)-np.exp(-Z)) / (np.exp(Z)+np.exp(-Z))) ** 2)
    return dA * mid


''' --------------------End of Activation Function ------------------------------'''
# suppose we are facing with single layer. Compute WA+b to get next layer Z.


def _single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="tanh"):
    Z_curr = np.dot(W_curr, A_prev) + b_curr

    if activation is "relu":
        activation_func = _relu
    elif activation is "tanh":
        activation_func = _tanh
    elif activation is "sigmoid":
        activation_func = _sigmoid
    else:
        raise Exception('Non-supported activation function')

    return activation_func(Z_curr), Z_curr

# compute next layer recursively using forward propagation


def _full_forward_propagation(X, params_values, nn_architecture):
    memory = {}
    A_curr = X

    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_prev = A_curr

        activ_function_curr = layer["activation"]
        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]
        A_curr, Z_curr = _single_layer_forward_propagation(
            A_prev, W_curr, b_curr, activ_function_curr)
#        print(A_curr)
        memory["A" + str(idx)] = A_prev
        memory["Z" + str(layer_idx)] = Z_curr

    return A_curr, memory

#[Y_hat,memory] = full_forward_propagation(X, params_values, nn_architecture)
# compute dA,dZ,dW,db using chain rule for previous layer


def _single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="tanh"):
    m = A_prev.shape[1]

    if activation is "relu":
        backward_activation_func = _relu_backward
    elif activation is "tanh":
        backward_activation_func = _tanh_backward
    elif activation is "sigmoid":
        backward_activation_func = _sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr
# compute derivative recursively to get descent of parameters


def _full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    Y = Y.reshape(Y_hat.shape)

    dA_prev = - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activ_function_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]
        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = _single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activ_function_curr)

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values
#grads_values = full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture)
# print(params_values)


def _update(params_values, grads_values, nn_architecture, learning_rate):
    for layer_idx in range(1, 4):
        params_values["W" + str(layer_idx)] -= learning_rate * \
            grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * \
            grads_values["db" + str(layer_idx)]

    return params_values
#update(params_values, grads_values, nn_architecture, learning_rate)
# print(params_values)


def _get_cost_value(Y_hat, Y):

    L = sum(sum((Y-Y_hat)**2))

    return L


def _train(X, Y, nn_architecture, epochs, learning_rate):
    params_values = _init_layers(nn_architecture, 2)
    cost_history = []
    accuracy_history = []

    for i in range(epochs):
        Y_hat, cashe = _full_forward_propagation(
            X, params_values, nn_architecture)
        cost = _get_cost_value(Y_hat, Y)
        cost_history.append(cost)
        print('i = ', i, 'cost = ', cost, 'avg_unit_cost', cost/(947*252))
#        accuracy = get_accuracy_value(Y_hat, Y)
#        accuracy_history.append(accuracy)

        grads_values = _full_backward_propagation(
            Y_hat, Y, cashe, params_values, nn_architecture)
        params_values = _update(
            params_values, grads_values, nn_architecture, learning_rate)
#        cost_old = cost
#        if (np.abs(cost_old-cost/cost_old)) < 1e-7:
#            break
    return params_values, cost_history, accuracy_history


''' Main Function --- Training '''


def NNPredict(n, m, t, nn_architecture, epochs, learning_rate, stockdata):
    seed = 4500
    total_data = stockdata.values
    X = total_data[0:n, 0:t]
    Y = total_data[0:n, 10:t+10]
    X = np.reshape(X, [n, t])
    params_values = _init_layers(nn_architecture, seed)
    _train(X, Y, nn_architecture, epochs, learning_rate)
    # print(params_values['W1'],params_values['W2'],params_values['W3'],params_values['b1'],params_values['b2'],params_values['b3'])
    '''Testing'''

    X = total_data[0:n, 253:494]
    Y = total_data[0:n, 253:494]
    [Y_hat, memory] = _full_forward_propagation(
        X, params_values, nn_architecture)
    return_val = _get_cost_value(Y_hat, Y)
    print('Y_hat=', Y_hat, 'Loss = ', return_val)
    return Y_hat, return_val