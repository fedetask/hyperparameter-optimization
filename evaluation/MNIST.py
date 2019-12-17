import os
import sys
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
sys.path.append(os.path.abspath('../gaussian_process/'))
from gaussian_process import GaussianProcess
import itertools as it
from pathlib import Path

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

kernel_sizes = [3, 5, 7, 9]
pool_sizes = [2, 3, 4]
dense_units = [32, 64, 128, 256]
drop_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
# Total combinations: 5x3x4x5 = 300
space = [kernel_sizes, pool_sizes, dense_units, drop_rates]
epochs = 1

def get_model(hyperparams):
    # Return a neural network for the given parameters
    # hyperparams: [kernel_size, pool_size, dense_units, drop_rate]
    ker_size, pool_size, dense_units, drop_rate = hyperparams
    model = Sequential()
    model.add(Conv2D(28, kernel_size=(ker_size, ker_size), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
    model.add(Dense(dense_units, activation=tf.nn.relu))
    model.add(Dropout(drop_rate))
    model.add(Dense(10,activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def evaluate(hyperparams, nepochs):
    # Train the model for given hyperparameters and epochs and return accuracy
    # hyperparams: [kernel_size, pool_size, dense_units, drop_rate]
    model = get_model(hyperparams)
    model.fit(x=x_train,y=y_train, epochs=1)
    loss, accuracy = model.evaluate(x_test, y_test)
    return accuracy

def find_max_gp(n_evaluations):
    # Iterate for n_evaluations a gaussian process over the parameter space
    # Return array of evaluated points and array with corrisponding values
    gp = GaussianProcess(space_dim=len(space),
                                    length_scale=1,
                                    noise=0,
                                    standardize=True)
    eval_point = [kernel_sizes[0],
                 pool_sizes[0],
                 dense_units[0],
                 drop_rates[0]]

    for i in range(n_evaluations):
        val = evaluate(eval_point, epochs)
        print('Evaluated '+str(eval_point)+' with value '+str(val))
        gp.add_points([eval_point], [val])
        eval_point = gp.most_likely_max(space)
        argmax, max = gp.get_max()
        print('Current maximum at '+str(argmax)+' with value '+str(max))
    return gp.known_points, gp.known_values


def find_max_grid_search():
    # Iterate grid search for the whole parameter space
    # Return array of evaluated points and array with corresponding values
    combinations = list(it.product(*space))
    eval_points = np.empty((len(combinations), len(space)))
    values = np.empty(len(combinations))
    for i, hyperparams in enumerate(combinations):
        val = evaluate(hyperparams, epochs)
        eval_points[i] = hyperparams
        values[i] = val
    return eval_points, values


if __name__ == "__main__":
    # The goal is to compare the evolution of the gp evaluations against
    # the grid search starting from the same point
    n_evaluations = 20
    eval_points_gp, values_gp = find_max_gp(n_evaluations)
    eval_points_gs, values_gs = find_max_grid_search()
    
    Path('eval_points_gp.npy').touch()
    Path('values_gp.npy').touch()
    Path('eval_points_gs.npy').touch()
    Path('values_gs.npy').touch()

    np.save('eval_points_gp.npy', eval_points_gp)
    np.save('eval_points_gs.npy', eval_points_gs)
    np.save('values_gp.npy', values_gp)
    np.save('values_gs.npy', values_gs) 
