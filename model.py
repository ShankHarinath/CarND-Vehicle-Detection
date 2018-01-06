import numpy as np
import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Lambda, \
    MaxPooling2D, Dense, Dropout, BatchNormalization, LeakyReLU, \
    concatenate
from tensorflow.python.keras.optimizers import Adam, SGD
from tensorflow.python.keras.metrics import categorical_accuracy, \
    mean_squared_error
from tensorflow.python.keras.callbacks import BaseLogger, \
    ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.python.keras import backend as K

from dataset import Dataset

output_dir = os.getcwd() + '/out/'


class Model:

    def __init__(
        self,
        input_shape=(64, 64, 3),
        batch_size=128,
        epochs=20,
        learning_rate=0.001,
        ):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dataset = Dataset(self.batch_size)

    def generate_model(self):
        gpu_list = ['/gpu:%d' % i for i in range(8)]
        gpus = iter(gpu_list)

        model = Sequential()
        model.add(Lambda(lambda x: x / 255.,
                  input_shape=self.input_shape, name='Lambda'))

        with tf.device(next(gpus)):
            model.add(Conv2D(filters=16, kernel_size=(3, 3),
                      activation='relu', padding='same', name='Conv2'))
            model.add(Dropout(0.5, name='Dropout1'))

        with tf.device(next(gpus)):
            model.add(Conv2D(filters=32, kernel_size=(3, 3),
                      activation='relu', padding='same', name='Conv3'))
            model.add(Dropout(0.5, name='Dropout2'))

        with tf.device(next(gpus)):
            model.add(Conv2D(filters=64, kernel_size=(3, 3),
                      activation='relu', padding='same', name='Conv4'))
            model.add(MaxPooling2D(pool_size=(8, 8), name='MaxPool'))
            model.add(Dropout(0.5, name='Dropout3'))

        with tf.device(next(gpus)):
            model.add(Conv2D(filters=1, kernel_size=(8, 8),
                      activation='sigmoid', name='Conv5'))
        return model

    def _compile_model(self, model):

        # Optimizer
        # optimizer = Adam(lr=self.learning_rate, decay=1e-6)

        optimizer = 'rmsprop'

    # Training loss

        loss = 'mse'

        # Training metrics

        metrics = ['acc']

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, model_weights_out=output_dir
                    + 'model_weights.h5', use_test_set=True):

        # Create the model

        model = self.generate_model()

        # Flatten to train the model

        model.add(Flatten())
        model.reset_states()

        # Compile the model

        self._compile_model(model)

        # Read data

        (train_data, validation_data, test_data) = \
            self.dataset.read_data()
        (aug_train_data, aug_validation_data, aug_test_data) = \
            self.dataset.read_augmented_data()

        train_data += aug_train_data
        validation_data += aug_validation_data
        test_data += aug_test_data

        # Get the data generators
        if use_test_set:
            (train_generator, validation_generator, _) = \
                self.dataset.get_generators(train_data,
                    validation_data, test_data, use_test_set)
            train_data = train_data + test_data
        else:
            (train_generator, validation_generator, test_generator) = \
                self.dataset.get_generators(train_data,
                    validation_data, test_data, use_test_set)

        # Define training metrics & callbacks
        tensorboard = TensorBoard(log_dir='./logs',
                                  batch_size=self.batch_size,
                                  write_graph=True)
        filepath = output_dir + 'model-weights.h5'
        checkpoints = ModelCheckpoint(
            filepath,
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            period=1,
            )
        callbacks = [tensorboard, checkpoints]

        # Fit the model

        history = model.fit_generator(
            train_generator,
            steps_per_epoch=len(train_data) / self.batch_size,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=len(validation_data) / self.batch_size,
            callbacks=callbacks,
            epochs=self.epochs,
            )

        # Get test accuracy

        if not use_test_set:
            accuracy = \
                model.evaluate_generator(generator=test_generator,
                    steps=len(test_data) / self.batch_size)
            print ('Test Accuracy: ', accuracy)

