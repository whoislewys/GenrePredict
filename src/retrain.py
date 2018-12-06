# retrain.py
from sklearn.model_selection import train_test_split
import numpy as np
import keras
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Sequential, Dense, Activation, Flatten, Input, Dropout
from keras.callbacks import ModelCheckpoint

X = np.load('x_gtzan_npy.npy')
y = np.load('y_gtzan_npy.npy')

y = to_categorical(y)

print('stacking data like pancakes')
X_stack = np.squeeze(np.stack((X,) * 3, -1))

print('splitting train & test sets like moses parting the red sea')
X_train, X_test, y_train, y_test = train_test_split(X_stack, y, test_size=0.3, random_state=42, stratify = y)

MODEL_SAVE_PATH = '../models/genrepredict.h5'
checkpointer = ModelCheckpoint(filepath=MODEL_SAVE_PATH, verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
callbacks_list = [checkpointer]
model = load_model(MODEL_SAVE_PATH)

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['categorical_accuracy'])

hist = model.fit(X_train, y_train,
				 batch_size=128,
				 epochs=5,
				 verbose=1,
				 callbacks=callbacks_list,
				 validation_data=(X_test, y_test))