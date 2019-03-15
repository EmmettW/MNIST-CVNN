from __future__ import absolute_import, division, print_function
import matplotlib as plt
import tensorflow as tf
import keras as ks

input_shape = ''
num_classes = 0
batch_size=128
epochs=6
num_classes=10
history=0
# need to prepare input data

dim_x, dim_y = 28, 28
(x_train, y_train), (x_test, y_test) = ks.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], dim_x, dim_y, 1)
x_test = x_test.reshape(x_test.shape[0], dim_x, dim_y, 1)
input_shape = (dim_x, dim_y, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


#build model
model= ks.Sequential()

model.add(ks.layers.Conv2D(32, kernel_size=(5,5), strides=(1,1), activation='relu', input_shape=input_shape))
model.add(ks.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(ks.layers.Conv2D(64, (5,5), activation='relu'))
model.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(ks.layers.Flatten())
model.add(ks.layers.Dense(1000, activation='relu'))
model.add(ks.layers.Dense(num_classes, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=ks.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test)) 

score=model.evaluate(x_test, y_test, verbose=0)
print('Test loss: ', score[0])
print('Test accuracy: ', score[1])

