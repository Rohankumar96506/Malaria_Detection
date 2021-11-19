import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,BatchNormalization,Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np


ds_train=tf.keras.preprocessing.image_dataset_from_directory(r"G:\malaria\cell_images\cell_images",labels="inferred",label_mode="categorical",batch_size=64,image_size=(128,128),shuffle=True,seed=187,validation_split=0.1,subset="training")
ds_validation=tf.keras.preprocessing.image_dataset_from_directory(r"G:\malaria\cell_images\cell_images",labels="inferred",label_mode="categorical",batch_size=64,image_size=(128,128),shuffle=True,seed=154,validation_split=0.1,subset="validation")

width = 128
height = 128
datagen = ImageDataGenerator(rescale=1/255.0, validation_split=0.2)
trainDatagen = datagen.flow_from_directory(directory=r"G:\malaria\cell_images\cell_images",target_size=(width,height),class_mode = 'categorical',batch_size = 8,subset='training')
valDatagen = datagen.flow_from_directory(directory=r'G:\malaria\cell_images\cell_images',
                                           target_size=(width,height),
                                           class_mode = 'categorical',
                                           batch_size = 16,
                                           subset='validation')

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(128,128,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=(134,131,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(filters=128, kernel_size=(3,3),input_shape=(134,131,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Conv2D(filters=256, kernel_size=(3,3),input_shape=(134,131,3),activation='relu',padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=2))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=["accuracy"])
model.summary()

history = model.fit(ds_train,epochs=6,validation_data=ds_validation,verbose=1)

model = tf.keras.models.load_model(r'malaria.h5')

#model.save(r'malaria.h5')


pic2 = r"G:\malaria\cell_images\normal2.png"
image_array = cv2.imread(pic2)
image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
new_array = cv2.resize(image_array,(128,128))
new_array = new_array.reshape(-1,128,128,3)
new_aray = [new_array]

print( model.predict(new_array))
#1 uninfited
#0infected