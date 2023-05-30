import tensorflow as tf
import numpy as np

def radial_basis_activation(input, centers, radius):
    diff = tf.subtract(tf.expand_dims(input, 1), centers)
    squared_diff = tf.square(diff)
    distances = tf.reduce_sum(squared_diff, axis=2)
    return tf.exp(-distances / (2 * tf.square(radius)))


def add_neuron(model, input_data, output_data, hidden_size):
    input_size = input_data.shape[1]
    prev_output_size = model.layers[-1].output_shape[1]

    # Create a new model with an additional neuron
    new_model = tf.keras.Sequential()
    new_model.add(model)
    new_model.add(tf.keras.layers.Dense(hidden_size, activation='linear'))

    # Initialize the weights and biases of the new neuron randomly
    new_model.layers[-1].set_weights([
        np.random.randn(prev_output_size, hidden_size),
        np.random.randn(hidden_size)
    ])

    # Train the new model with the additional neuron
    new_model.compile(optimizer='adam', loss='mean_squared_error')
    new_model.fit(input_data, output_data, epochs=100, verbose=0)

    return new_model


def rbnn_model(X_train, y_train, hidden_size):
    input_size = X_train.shape[1]

    # Initialize the RBNN model with an initial number of neurons
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden_size, activation='linear', input_shape=(input_size,)))

    # Train the initial model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=100, verbose=0)

    # Add neurons dynamically
    for _ in range(5):  # Add 5 neurons
        model = add_neuron(model, X_train, y_train, hidden_size)

    return model



# Input data
turbin1_ML = np.genfromtxt('turbin1-machinelearning.csv', delimiter=',')
turbinParam = turbin1_ML[:, 0:7]
effi = turbin1_ML[:, 7]  # Update the column index for the output data

# Normalize the input data
turbinParam = (turbinParam - np.mean(turbinParam, axis=0)) / np.std(turbinParam, axis=0)

# Reshape the output data to match the input data shape
effi = effi.reshape((-1, 1))

# RBNN model
hidden_size = 10

model = rbnn_model(turbinParam, effi, hidden_size)
model.save('model.h5')





