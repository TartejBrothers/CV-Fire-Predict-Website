import zipfile
zip_ref = zipfile.ZipFile('/content/urecamain.zip', 'r')
zip_ref.extractall('/content')
zip_ref.close()

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout,AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GaussianNoise, GlobalMaxPooling2D
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import matplotlib.pyplot as plt
#Loading the images
x_train= keras.utils.image_dataset_from_directory(
    directory = '/content/Train',
    labels='inferred',
    label_mode = 'int',
    batch_size=100,
    image_size=(256,256)
)

x_test= keras.utils.image_dataset_from_directory(
    directory = '/content/Test',
    labels='inferred',
    label_mode = 'int',
    batch_size=100,
    image_size=(256,256)
)
class_names = x_train.class_names
class_names
# Display a few sample images
plt.figure(figsize=(10, 10))
for images, labels in x_train.take(1):
    for i in range(6):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
# Normalizing
def process(image,label):
    image = tf.cast(image/256. ,tf.float32)
    return image,label

x_train = x_train.map(process)
x_test = x_test.map(process)
input_shape = (256, 256, 3)
#Model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(GaussianNoise(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(GaussianNoise(0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(GlobalMaxPooling2D())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(x_train,epochs=5, batch_size=200, validation_data=x_test)
model.evaluate(x_test)
#Plotting  Graphs - Accuracy
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
#Loss
plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# Loading and displaying an image
image_path = "/content/Fire Image.jpg"
image = mpimg.imread(image_path)
resized_image = cv2.resize(image, (256, 256))
normalized_image = resized_image / 256.0
input_image = np.expand_dims(normalized_image, axis=0)


plt.imshow(image)
plt.axis('off')
plt.show()
#Making prediction
prediction = model.predict(input_image)
Fire = prediction >= 0.5
if Fire:
    print("Fire")
else:
    print("No Fire")