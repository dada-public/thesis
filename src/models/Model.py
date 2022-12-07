import os
import sys
import librosa
import numpy as np
import yaml

from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling2D, MaxPooling1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
from dotenv import find_dotenv, load_dotenv
from os.path import join

load_dotenv(find_dotenv())

sys.path.append(join(os.getenv("path.root"), 'src', 'utils'))
sys.path.append(join(os.getenv("path.root"), 'src', 'features'))

from utils import clean_dir  # noqa
from Features import Feature  # noqa

DROP = 0.3  # Dropout


class Model:

    def __init__(self):
        ''' It defines and compiles the model'''

        # The numbers in use for our optimizer come from the original
        # paper

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-8),
            metrics=['accuracy']
        )

    def plot(self):
        ''' It plots a summary of the algorithm in use '''
        print(self.model.summary())
        return self

    def train(self):
        ''' It defines and launchs training process'''

        # We'll stop after 20 epochs without a validation accuracy
        # of (at least) 0.001

        halt = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=15,
            verbose=0,
            mode='auto',
            restore_best_weights=True
        )

        self.model.fit(
            self.X_train, self.Y_train,
            batch_size=25,  # Deliberately low, to avoid alloc exceptions
            epochs=400,  # We trust in halt
            validation_data=(self.X_val, self.Y_val),
            callbacks=[halt]
        )

        return self

    def save(self, dest):
        ''' It saves the model'''
        self.model.save(
            join(dest, "{0}_{1}D.h5".format(self.use, str(self.dim)))
        )
        return self

    def load(self, name):
        ''' Loads a previously trained model'''
        self.model = load_model(join(os.getenv("path.models"), name + '.h5'))
        return self

    def eval(self, song):
        ''' It evals the algorithm against the TEST dataset. It needs
            to reproduce first the process of data augmentation
        '''

        feat = Feature(use=self.use, dim=self.dim)
        excerpts, labels = feat.process(song, 11)
        classes = list(self.model.predict_classes(np.array(excerpts)))
        return max(set(classes), key=classes.count)


class Model1D(Model):

    def __init__(self, rs=42, use='MEL'):
        ''' It defines the geometry for Conv1D neural network '''
        self.use = use
        self.rs = rs
        self.dim = 1

        SOURCE = os.getenv("path.data.processed")

        X = np.load(join(SOURCE, self.use, 'train',  'x.npy'))
        Y = np.load(join(SOURCE, self.use, 'train',  'y.npy'))

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=self.rs
        )

        self.model = Sequential()

        self.model.add(Conv1D(
                filters=16,
                kernel_size=3,
                strides=1,
                padding='same',
                input_shape=(
                    self.X_train[0].shape[0],
                    self.X_train[0].shape[1]
                ),
                activation='relu'
            )
        )
        self.model.add(MaxPooling1D(pool_size=2, strides=2))
        self.model.add(Dropout(DROP))

        self.model.add(Conv1D(
                filters=32,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu'
            )
        )
        self.model.add(MaxPooling1D(pool_size=2, strides=2))
        self.model.add(Dropout(DROP))

        self.model.add(Conv1D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu'
            )
        )
        self.model.add(MaxPooling1D(pool_size=2, strides=2))
        self.model.add(Dropout(DROP))

        self.model.add(LSTM(100, dropout=DROP, kernel_regularizer=l2(0.001)))
        self.model.add(Dense(10, activation="softmax"))

        super().__init__()


class Model2D(Model):

    def __init__(self, rs=42, use='MEL'):
        ''' It defines neural network's geometry for Conv2D algorithm '''

        self.use = use
        self.rs = rs
        self.dim = 2

        SOURCE = os.getenv("path.data.processed")

        X = np.load(join(SOURCE, self.use, 'train',  'x.npy'))
        Y = np.load(join(SOURCE, self.use, 'train',  'y.npy'))

        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=self.rs
        )

        self.model = Sequential()

        self.model.add(Conv2D(
                8,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same',
                input_shape=self.X_train[0].shape
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(DROP))

        self.model.add(Conv2D(
                16,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same',
                input_shape=self.X_train[0].shape
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(DROP))

        self.model.add(Conv2D(
                32,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same',
                input_shape=self.X_train[0].shape
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(DROP))

        self.model.add(Flatten())

        self.model.add(Dense(
            100,
            activation='relu',
            kernel_regularizer=l2(0.01)
            )
        )
        self.model.add(Dropout(DROP))

        self.model.add(Dense(
                100,
                activation='relu',
                kernel_regularizer=l2(0.001)
            )
        )
        self.model.add(Dropout(DROP))

        self.model.add(Dense(10, activation='softmax'))

        super().__init__()
