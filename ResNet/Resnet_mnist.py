import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images, test_images = train_images / 255.0, test_images / 255.0
inputs = keras.Input(shape=(28, 28, 1))
x = inputs
#x = layers.MaxPooling2D(2)(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(_x)
x = _x
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(_x)
x = x + _x
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(_x)
x = x + _x
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(x)
_x = layers.Conv2D(128, 3, activation='relu', padding="same")(_x)
x = x + _x
x = layers.MaxPooling2D(2)(x)

x = layers.Flatten()(x)
x = layers.Dense(128)(x)
x = layers.Dense(128)(x)
x = layers.Dense(10, activation='softmax')(x)
outputs = x

model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images,  test_labels, verbose=2)
