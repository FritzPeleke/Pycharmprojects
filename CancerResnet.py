import os
import random
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Layer
from tensorflow.keras.layers import GlobalAvgPool2D, Flatten, AveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from functools import partial
from tensorflow.keras.callbacks import TensorBoard
from time import time


pd.options.display.width = 0
path = os.path.join(os.getcwd(), 'histopathologic-cancer-detection')
path_totrain = os.path.join(path, 'train')
path_tolabels = os.path.join(path, 'train_labels.csv')
files = np.asarray(os.listdir(path_totrain))


def rest_graph(seed=42):
    tf.compat.v1.reset_default_graph()
    tf.keras.backend.clear_session()
    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed)


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


DefaultConv2D = partial(Conv2D, kernel_size=3, strides=1, padding='SAME', use_bias=False)

class Residual_unit(Layer):
    def __init__(self, filters, strides, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters=filters, strides=strides),
            BatchNormalization(),
            self.activation,
            DefaultConv2D(filters=filters),
            BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters=filters, strides=strides, kernel_size=1),
                BatchNormalization()
            ]

    def call(self, inputs, **kwargs):
        z = inputs
        for layer in self.main_layers:
            z = layer(z)
        skip_z = inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)
        return self.activation(z + skip_z)

#fit parameters
steps_per_epoch = int(np.ceil(train_set.shape[0]/128))
adam = tf.keras.optimizers.Adam()
LRreduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5,
                               verbose=1, restore_best_weights=True)


#creating Resnet-18
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=3, activation='relu', strides=2, padding='SAME',
                 input_shape=[96, 96, 3]))
model.add(MaxPool2D(pool_size=3))
model.add(Dropout(rate=0.5))
prev_filters = 64
for filters in [64] * 4 + [128] * 4 + [256] * 4 + [512] * 4:
    strides = 1 if filters == prev_filters else 2
    model.add(Residual_unit(filters=filters, strides=strides))
    prev_filters = filters

model.add(GlobalAvgPool2D())
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

print(model.summary())
model.compile(optimizer=adam, loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit_generator(generator=train_gen, epochs=15,
                              validation_data=val_gen,
                              steps_per_epoch=steps_per_epoch,
                              callbacks=[LRreduce, early_stopping])

score = model.evaluate_generator(test_gen)

print(score)


