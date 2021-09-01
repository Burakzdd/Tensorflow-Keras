import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

veri_adresi = '/home/burakzdd/Desktop/cats_and_dogs_filtered'

train_veri = os.path.join(veri_adresi, 'train')
validation_veri = os.path.join(veri_adresi, 'validation')
""""
train_cat = os.path.join(train_veri, 'cats')
train_dog = os.path.join(train_veri, 'dogs')
valid_cat = os.path.join(validation_veri, 'cats')
valid_dog = os.path.join(validation_veri, 'dogs')
"""
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation='relu', input_shape = (150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])
model.summary()

model.compile(optimizer=RMSprop(lr=0.001),loss= 'binary_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0/255. )
validation_datagen = ImageDataGenerator(rescale=1.0/255. )

train_generator = train_datagen.flow_from_directory(train_veri,batch_size=20, class_mode='binary',target_size=(150,150))
validation_generator = validation_datagen.flow_from_directory(validation_veri, batch_size=20, class_mode='binary', target_size=(150,150))

model.fit(train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100, 
        epochs=15, 
        validation_steps=50, 
        verbose=2)
model.save('cat_dog_model.h5')