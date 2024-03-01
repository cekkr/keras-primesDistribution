# Temporary snippets

## How to correlate to the model hidden information
This is a ChatGPT snippet:
Yes, it is possible to pass additional hidden information, represented as floats, to a Q-learning model alongside the game state represented by an integer matrix. One common approach to handle this situation is to use multiple inputs in a Keras model.

In Keras, you can create a model with multiple input layers, each handling a different type of input data. You can then combine these input layers using different types of layers (e.g., Dense, LSTM, etc.) to build a more complex model that can handle both the game state and the hidden information.

Here's a general outline of how to create a Keras model with multiple inputs:

```python
from keras.models import Model
from keras.layers import Input, Dense, concatenate

# Assuming game state is represented as an integer matrix of shape (game_rows, game_cols)
game_input = Input(shape=(game_rows, game_cols), name='game_input')

# Assuming hidden information is represented as a float vector of shape (hidden_dim,)
hidden_info_input = Input(shape=(hidden_dim,), name='hidden_info_input')

# Add layers to process the game state and hidden information
# Example: flatten the game input and add some Dense layers for game processing
game_processed = Flatten()(game_input)
game_processed = Dense(64, activation='relu')(game_processed)

# Example: add some Dense layers for hidden information processing
hidden_info_processed = Dense(32, activation='relu')(hidden_info_input)

# Combine the processed game state and hidden information using concatenate
combined = concatenate([game_processed, hidden_info_processed])

# Add more layers as needed for the Q-learning model
# Example: Dense output layer for Q-values
q_values = Dense(num_actions, activation='linear')(combined)

# Create the model with the specified inputs and outputs
model = Model(inputs=[game_input, hidden_info_input], outputs=q_values)

# Compile the model and define the loss and optimizer as needed
model.compile(optimizer='adam', loss='mse')

# Train the model using appropriate data for both inputs
# Example: game_data and hidden_info_data are the training data for game state and hidden information, respectively
model.fit([game_data, hidden_info_data], q_values_data, epochs=...)
```

In this example, we create two input layers, one for the game state and the other for the hidden information. We process each input using different layers (Dense in this case), and then concatenate the processed outputs. Finally, we continue with more layers and an output layer to produce the Q-values.

When training the model, you will need to provide appropriate training data for both the game state and hidden information inputs, along with the corresponding target Q-values.

Remember to preprocess your data appropriately before feeding it into the model. The specific architecture and hyperparameters will depend on the complexity of your game and the nature of the hidden information.


### About flatten layer
The structure of the input array does matter for a Keras model. The shape of the input data must match the input shape expected by the first layer of the model. Keras models are not indifferent to the input shape; it needs to be compatible with the network architecture.

For example, if your first layer is a convolutional layer, the input data should have a shape representing `(batch_size, height, width, channels)` where `batch_size` is the number of samples in your batch, `height` and `width` are the dimensions of your input images, and `channels` is the number of color channels (e.g., 3 for RGB images).

Flattening is a specific operation that converts multi-dimensional data into a one-dimensional array. It's commonly used when you transition from convolutional layers to dense layers in a neural network. For instance, if you have a convolutional layer producing output of shape `(batch_size, height, width, channels)`, you typically flatten this into a shape of `(batch_size, height * width * channels)` before feeding it into a dense layer.

When you design your model, make sure the input shape specified in the first layer matches the shape of your input data. If you're dealing with images, convolutional layers are often used early in the model, and the input data shape would typically be `(height, width, channels)`. The need for flattening generally arises when transitioning from convolutional layers to dense layers.

So, to answer your question: No, it's not necessary to always flatten the input array. The structure of the input array should align with the expectations of your model's architecture, and flattening is a specific operation that's applied when needed, typically when transitioning from convolutional layers to dense layers.


## Inspirations

### BigFloat
Exists a library in Python able to create floating point variable with dynamic precision: https://pypi.org/project/bigfloat/