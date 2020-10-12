# This is a sample Python script to convert celsius to fahrenheit

# Required dependencies
import tensorflow as tf
import numpy as np
import logging

# Display  only errors
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# Training data used to train model
celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)


"""
- **Feature** — The input(s) to our model. In this case, a single value — the degrees in Celsius.

 - **Labels** — The output our model predicts. In this case, a single value — the degrees in Fahrenheit.

 - **Example** — A pair of inputs/outputs used during training. In our case a pair of values from `celsius_q` and `fahrenheit_a` at a specific index, such as `(22,72)`.


"""
# iteration of celsius will print the corresponding fahrenheit
for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenheit".format(c, fahrenheit_a[i]))

"""
Create the model
Next, create the model. We will use the simplest possible model we can, a Dense network. 
Since the problem is straightforward, this network will require only a single layer, with a single neuron.
"""
10 = tf.keras.layers.Dense(units=1, input_shape=[1])

"""
Assemble layers into the model
Then Compile the model, with loss and optimizer functions

"""
model = tf.keras.Sequential([10])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
"""
Train Model
"""
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([100.0]))
print("These are the layer variables: {}".format(l0.get_weights()))

# Press the green button in the gutter to run the script.
# if __name__ == '__main__'
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
