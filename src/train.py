'''
it predicts genres ¯\_(ツ)_/¯
'''

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import keras
import h5py
from keras.utils import to_categorical
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Dropout
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint

MODEL_SAVE_PATH = '../models/genrepredict_CHECKPOINT.h5'

def splitsongs(X, y, window=0.1, overlap=0.5):
    """
    @description: Method to split a song into multiple songs using overlapping windows
    """
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape * window)
    offset = int(chunk * (1. - overlap))

    # Split the song and create new ones on windows
    spsong = [X[i:i + chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)


def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    """
    @description: Method to convert a list of songs to a np array of melspectrograms
    """
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
        hop_length = hop_length)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))


def read_data(src_dir, genres, song_samples, spec_format, debug=True):
    # Empty array of dicts with the processed features from all files
    arr_specs = []
    arr_genres = []

    # Read files from the folders
    for x, _ in genres.items():
        folder = src_dir + x

        for root, subdirs, files in os.walk(folder):
            for file in files:
                # Read the audio file
                file_name = folder + "/" + file
                signal, sr = librosa.load(file_name)
                signal = signal[:song_samples]

                # Debug process
                if debug:
                    print("Reading file: {}".format(file_name))

                # Convert to dataset of spectograms/melspectograms
                signals, y = splitsongs(signal, genres[x])

                # Convert to "spec" representation
                specs = spec_format(signals)

                # Save files
                arr_genres.extend(y)
                arr_specs.extend(specs)

    return np.array(arr_specs), np.array(arr_genres)


# Parameters
gtzan_dir = '../data/genres/'
print('data dir: ', gtzan_dir)
song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}

# Read the data
print('reading the data')
X, y = read_data(gtzan_dir, genres, song_samples, to_melspectrogram, debug=False)
np.save('x_gtzan_npy.npy', X)
np.save('y_gtzan_npy.npy', y)

X = np.load('x_gtzan_npy.npy')
y = np.load('y_gtzan_npy.npy')

# One hot encoding of the labels
y = to_categorical(y)

print('stacking data like pancakes')
X_stack = np.squeeze(np.stack((X,) * 3, -1))
# X_stack.shape

print('splitting train & test sets like moses parting the red sea')
X_train, X_test, y_train, y_test = train_test_split(X_stack, y, test_size=0.3, random_state=42, stratify = y)

# Model Definition
input_shape = X_train[0].shape
num_genres = 10

def cnn_vgg16(input_shape, num_genres, freezed_layers):
    input_tensor = Input(shape=input_shape)
    vgg16 = VGG16(include_top=False, weights='imagenet',
                  input_tensor=input_tensor)

    top = Sequential()
    top.add(Flatten(input_shape=vgg16.output_shape[1:]))
    top.add(Dense(256, activation='relu'))
    top.add(Dropout(0.5))
    top.add(Dense(num_genres, activation='softmax'))

    model = Model(inputs=vgg16.input, outputs=top(vgg16.output))
    for layer in model.layers[:freezed_layers]:
        layer.trainable = False

    return model

model = cnn_vgg16(input_shape, num_genres, 5)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
callbacks_list = [checkpointer]

print('get ready for the training fiesta')
hist = model.fit(X_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          callbacks=callbacks_list,
          validation_data=(X_test, y_test))

# score = model.evaluate(X_test, y_test, verbose=0)
# print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

#model.save('../models/nbs_vgg16.h5')