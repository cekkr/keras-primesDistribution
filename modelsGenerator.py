import keras.layers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Flatten, Input, Lambda, BatchNormalization, Reshape, GRU, Conv2D, MaxPooling2D, LSTM
from keras.layers import Activation, Concatenate, AveragePooling2D, GlobalAveragePooling2D, TimeDistributed
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
import tensorflow as tf

import sqlite3
import os
import json
import time

import config
from db import DB


class modelsGenerator:

    def __init__(self, inputShape, outputShape):
        self.inputShape = inputShape
        self.outputShape = outputShape

        self.con = sqlite3.connect(config.currentDir + "dataset.db")
        self.cur = self.con.cursor()

        self.db = DB(config.currentDir + "dataset.db")

        self.initDB()

    def initDB(self):
        if not self.tableExists('dataset'):
            self.cur.execute("CREATE TABLE dataset(id INTEGER NOT NULL PRIMARY KEY, input TEXT, target TEXT, time INTEGER)")

        if not self.tableExists('combination'):
            self.cur.execute("CREATE TABLE combination(id INTEGER NOT NULL PRIMARY KEY, descriptor TEXT, value TEXT, score REAL, parent INTEGER)")

        if not self.db.exists('dataset'):
            dataset = self.db.get('dataset')
            dataset.input = 'TEXT'
            dataset.target = 'TEXT'
            dataset.time = 'INTEGER'

            dataset.create()

        if not self.db.exists('combination'):
            dataset = self.db.get('combination')
            dataset.descriptor = 'TEXT'
            dataset.value = 'TEXT'
            dataset.score = 'REAL'
            dataset.parent = 'INTEGER'

            dataset.create()

    def tableExists(self, name):
        sql = "SELECT name FROM sqlite_master WHERE type='table' AND name='"+name+"'"
        res = self.cur.execute(sql).fetchone()
        return res is not None

    def loadModel(self):
        model = load_model(config.currentDir + 'model.h5')

        self.fileTraining = config.currentDir + 'lastTraining.json'

        self.fileWeights = config.currentDir + 'weights.dat'
        if (os.path.exists(self.fileWeights)):
            model.load_weights(self.fileWeights)

        self.dirOutputs = config.currentDir + 'outputs'
        if not os.path.isdir(self.dirOutputs):
            os.makedirs(self.dirOutputs)

        self.model = model

    def trainInput(self, input, target):
        input = json.dumps(input)
        target = json.dumps(target)

        self.cur.execute("INSERT INTO dataset (input, target, time) VALUES (?,?,?)", (input, target, time.time()))
        self.cur.commit()

    ###
    ### Models generator
    ###

    def initGenerator(self):
        self.defCombinations()
        self.resetCombinations()

    def resetCombinations(self):
        #self.unitsRange = [5, 10]  # 32 - 1024
        self.combinationsManager = CombinationManager(self.cur, self.combinations, 'layer')

    def generateModel(self):
        layer = self.getLayer()

    def getLayer(self):
        for layer in self.layers:
            if layer.surface:
                return layer

    ###
    ### Define combinations
    ###
    def defCombinations(self):
        self.combinations = {}

        self.defActivations()
        self.defLayers()

    def defActivations(self):
        self.activations = []
        self.combinations['activation'] = self.activations

        sigmoid = self.defActivation('sigmoid')
        sigmoid.default = 'sigmoid'

        sigmoid = self.defActivation('relu')
        sigmoid.default = 'relu'

        sigmoid = self.defActivation('gelu')
        sigmoid.default = 'gelu'

        sigmoid = self.defActivation('linear')
        sigmoid.default = 'linear'

        sigmoid = self.defActivation('softmax')
        sigmoid.default = 'softmax'

    def defActivation(self, name):
        act = Combination(name, 'activation')
        self.activations.append(act)

    def defLayers(self):
        self.layers = []
        self.combinations['layer'] = self.layers

        # Dense
        dense = self.defLayer('dense')
        dense.instance = Dense

        # LSTM
        lstm = self.defLayer('lstm')
        lstm.instance = LSTM

        # Concatenate
        concatenate = self.defLayer('concatenate')
        concatenate.instance = keras.layers.Concatenate
        concatenate.surface = False

    def defLayer(self, name):
        layer = Combination(name, 'layer')
        self.layers.append(layer)


#todo: Create "Combination" recursive class
class Combination:
    def __init__(self, name, type):
        self.name = name

        self.type = type
        self.defType()

        self.surface = True

    def defType(self):
        match self.type:
            case 'layer':
                self.defTypeLayer()

    def defTypeLayer(self):
        self.options = [
            'units',
            'activation'
        ]

class CombinationManager:
    def __init__(self, cursor, options, base):
        self.cur = cursor
        self.options = options
        self.base = base
        self.models = []

    def generateModels(self):
        if len(self.models) == 0:
            print("generate base models")

###
### General model functions
###

def weighted_binary_crossentropy(y_true, y_pred):
    # Calculate the binary cross-entropy loss
    bce_loss = binary_crossentropy(y_true, y_pred)

    # Assign higher weights to errors where y_true is close to 1
    weighted_loss = tf.where(y_true < 0.5, bce_loss, 10 * bce_loss)

    return weighted_loss

"""## Models

### DenseNet
"""

def getModelDenseNet(input_shape):

  activation = 'gelu'

  def dense_block(x, growth_rate, num_layers):
      x_in = x  # Input to the dense block
      for _ in range(num_layers):
          # Bottleneck layer
          x = BatchNormalization()(x)
          x = Activation(activation)(x)
          x = Conv2D(growth_rate, kernel_size=(1, 1), padding='same')(x)

          # Composite function
          x = BatchNormalization()(x)
          x = Activation(activation)(x)
          x = Conv2D(growth_rate, kernel_size=(3, 3), padding='same')(x)

          # Concatenate with previous block
          x = Concatenate()([x, x_in])

      return x

  def transition_block(x, compression_factor):
      num_filters = int(x.shape[-1] * compression_factor)

      x = BatchNormalization()(x)
      x = Activation(activation)(x)
      x = Conv2D(num_filters, kernel_size=(1, 1), padding='same')(x)
      x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)

      return x

  def DenseNet(input_shape, num_blocks=4, num_layers_per_block=4, growth_rate=32, compression_factor=0.5, num_classes=10):
      inputs = Input(shape=input_shape)

      # Initial Convolutional layer
      x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
      x = BatchNormalization()(x)
      x = Activation(activation)(x)
      x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

      # Dense blocks and transition blocks
      x_in = x
      for i in range(num_blocks):
          x = dense_block(x, growth_rate, num_layers_per_block)
          x_in = transition_block(x, compression_factor)

      # Final blocks
      x = dense_block(x, growth_rate, num_layers_per_block)
      x = BatchNormalization()(x)
      x = Activation(activation)(x)
      x = GlobalAveragePooling2D()(x)

      # Dense layers
      x = Dense(256, activation=activation)(x)
      x = Dense(128, activation=activation)(x)

      # Output layer
      x = Dense(num_classes, activation='linear')(x) # activation='softmax' => sigmoid

      model = Model(inputs, x)
      return model

  model = DenseNet(input_shape=input_shape, num_blocks=3, num_layers_per_block=3, growth_rate=256, compression_factor=0.5, num_classes=actions)
  model.compile(optimizer="adam", loss="mean_absolute_error")
  return model

"""### LSTM"""

def getModelLSTM(input_shape):
  activation = 'gelu'
  lstm_units = 256
  num_lstm_layers = 4

  inputs = Input(shape=input_shape)
  timeSeries = TimeDistributed(LSTM(units=lstm_units, activation=activation), input_shape=input_shape)(inputs)

  # Build a dense block with LSTM layers
  lstm_layers = []
  prev_layer = timeSeries

  for _ in range(num_lstm_layers):
      lstm_layer = LSTM(units=lstm_units, return_sequences=True, activation=activation)(prev_layer)
      lstm_layers.append(lstm_layer)
      prev_layer = Concatenate()([prev_layer, lstm_layer])

  # Dense layers
  prev_layer = Dense(128, activation=activation)(prev_layer)
  prev_layer = Dense(64, activation=activation)(prev_layer)

  # Flatten
  prev_layer = Flatten()(prev_layer)

  # Final output layer
  output = Dense(actions, activation='linear', bias_initializer=tf.keras.initializers.Constant(0.0))(prev_layer)

  # Create the model
  model = Model(inputs=inputs, outputs=output)

  # Compile the model with appropriate loss, optimizer, and metrics
  model.compile(loss=weighted_binary_crossentropy, optimizer='adam') #, metrics=['accuracy']
  return model

#model = getModelLSTM()