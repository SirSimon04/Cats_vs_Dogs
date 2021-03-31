import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

NAME = "Cats_vs_Dogs-cnn-64x2-{}".format(int(time.time()))
X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

X = X/255.0

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = Sequential()

model.add(Conv2D(64, (3,3), activation="relu",input_shape = X.shape[1:]))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64, activation="relu"))

model.add(Dense(1,activation="sigmoid"))

model.compile(loss="binary_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model.fit(X, y,
          batch_size=500,
          validation_split=0.1,
          epochs=3,
          callbacks=[tensorboard])