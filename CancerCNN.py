import os
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(42)
tf.compat.v1.set_random_seed(42)
random.seed(42)
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from time import time


pd.options.display.width = 0
path = os.path.join(os.getcwd(), 'histopathologic-cancer-detection')
path_totrain = os.path.join(path, 'train')
path_tolabels = os.path.join(path, 'train_labels.csv')
files = np.asarray(os.listdir(path_totrain))


def rest_graph():
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()


def read_pic(X):
    img = cv2.imread(X)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return rgb_img


rest_graph()


df_labels = pd.read_csv(path_tolabels)
labels = df_labels.values
positive_indices = list(df_labels[df_labels.label == 1].index)
negative_indices = list(df_labels[df_labels.label == 0].index)

df_labels['Path'] = path_totrain + '/' + (df_labels['id'] + '.' + 'tif')
size = len(files)

#Distribution of data
per_pos, per_neg = len(positive_indices)/size, len(negative_indices)/size
df_labels.hist()
plt.show()

#visualisation of pictures
ran_neg_ind = random.sample(negative_indices, 4)
ran_pos_ind = random.sample(positive_indices, 4)
ran_positive_pics = files[ran_pos_ind]
ran_negative_pics = files[ran_neg_ind]
PathRanPosPics = [os.path.join(path_totrain, pic) for pic in ran_positive_pics]
PathRanNegPics = [os.path.join(path_totrain, pic) for pic in ran_negative_pics]

fig, axs = plt.subplots(2, 4, figsize=[15, 4])
fig.suptitle('Histopathologic scans of lymph node', fontweight='bold')
for i in range(4):
    axs[0, i].imshow(read_pic(PathRanNegPics[i]))
    axs[0, i].set_title('Positive Example', fontweight='bold')

    axs[1, i].imshow(read_pic(PathRanNegPics[i]))
    axs[1, i].set_title('Negative Example', fontweight='bold')
plt.show()

df_labels['label'] = df_labels['label'].astype(str)
train_df = df_labels
print(train_df.head())


#train_df = train_df.iloc[:160000]
#Splitting data into train, val_set and test_set
train_set, valid_set = train_test_split(train_df, test_size=0.2)
train_set, test_set = train_test_split(train_set, test_size=0.2)


Datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, vertical_flip=True)
train_gen = Datagen.flow_from_dataframe(train_set, directory=None, x_col='Path',
                                        y_col='label',
                                        target_size=(96, 96),
                                        batch_size=128, class_mode='binary',
                                        shuffle=True)
valDatgen = ImageDataGenerator(rescale=1./255)
val_gen = valDatgen.flow_from_dataframe(valid_set, x_col='Path', y_col='label',
                                        target_size=(96, 96), batch_size=128,
                                        class_mode='binary', shuffle=False)
testDatgen = ImageDataGenerator(rescale=1./255)
test_gen = testDatgen.flow_from_dataframe(test_set, x_col='Path', y_col='label',
                                          shuffle=False, target_size=(96, 96),
                                          class_mode='binary')

steps = int(np.ceil(train_set.shape[0]/128))
print(steps)
#Creating a CNN
model = Sequential()
model.add(Conv2D(64, kernel_size=3, strides=2, padding='SAME', activation='relu',
                 input_shape=[96, 96, 3]))
model.add(Conv2D(64, kernel_size=3, strides=1, padding='SAME', activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=1, padding='SAME', activation='relu'))
model.add(Conv2D(64, kernel_size=3, strides=1, padding='SAME', activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))
model.add(Dropout(rate=0.3))

model.add(Conv2D(128, kernel_size=3, strides=2, padding='SAME', activation='relu'))
model.add(Conv2D(128, kernel_size=3, strides=1, padding='SAME', activation='relu'))
model.add(Conv2D(128, kernel_size=3, strides=1, padding='SAME', activation='relu'))
model.add(Conv2D(128, kernel_size=3, strides=1, padding='SAME', activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))

model.add(Conv2D(256, kernel_size=3, strides=2, padding='SAME', activation='relu'))
model.add(Conv2D(256, kernel_size=3, strides=1, padding='SAME', activation='relu'))
model.add(Conv2D(256, kernel_size=3, strides=1, padding='SAME', activation='relu'))
model.add(MaxPool2D(pool_size=3, strides=2))
model.add(Dropout(rate=0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, 'sigmoid'))
print(model.summary())
opt = tf.keras.optimizers.Adam()
ES = EarlyStopping(monitor='val_loss', patience=3, verbose=1,
                                      restore_best_weights=True)
LRR = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1)
model.compile(optimizer=opt, loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(train_gen, steps_per_epoch=steps, epochs=15,
                              callbacks=[ES, LRR], validation_data=val_gen)

score = model.evaluate_generator(test_gen)
print(score)
