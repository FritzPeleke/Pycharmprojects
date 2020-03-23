import tensorflow as tf
import numpy as np
from tensorflow.keras import backend
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

np.random.seed(42)
tf.compat.v1.set_random_seed(42)

def reset_graph():
    tf.compat.v1.reset_default_graph()
    backend.clear_session()

reset_graph()

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train, X_test = X_train.astype(np.float32)/255.0, X_test.astype(np.float32)/255.0
y_train, y_test = y_train.astype(np.int32), y_test.astype(np.int32)
X_valid, X_train = X_train[:300], X_train[300:]
y_valid, y_train = y_train[:300], y_train[300:]


model = tf.keras.Sequential()
Lrelu = tf.keras.layers.LeakyReLU()
#Block 1
model.add(Conv2D(64, kernel_size=3, padding='SAME',
                 input_shape=[32, 32, 3]))
model.add(Lrelu)
model.add(Conv2D(64, kernel_size=3, padding='SAME'))
model.add(Lrelu)
model.add(Conv2D(64, kernel_size=3, padding='SAME'))
model.add(Lrelu)
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Conv2D(128, kernel_size=3, padding='SAME'))
model.add(Lrelu)
model.add(Conv2D(128, kernel_size=3, padding='SAME'))
model.add(Lrelu)
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Dropout(rate=0.5))
model.add(Conv2D(256, kernel_size=3, padding='SAME'))
model.add(Lrelu)
model.add(Conv2D(256, kernel_size=3, padding='SAME'))
model.add(Lrelu)
model.add(MaxPool2D(pool_size=2, strides=2))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

ES = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
RLR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
opt = tf.keras.optimizers.Nadam()
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=X_train, y=y_train, batch_size=128, epochs=15,
                    validation_split=0.1, callbacks=[ES, RLR])
score = model.evaluate(X_valid, y_valid)

print(score)
