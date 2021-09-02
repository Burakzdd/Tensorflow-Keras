import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import cv2
model = tf.keras.models.load_model('/home/burakzdd/cat_dog_model.h5')
img = cv2.imread("/home/burakzdd/Desktop/cats_and_dogs_filtered/validation/cats/cat.2010.jpg")
img2 = tf.keras.preprocessing.image.load_img(
    "/home/burakzdd/Desktop/cats_and_dogs_filtered/validation/cats/cat.2010.jpg", target_size=(150,150)
)
img_array = tf.keras.preprocessing.image.img_to_array(img2)
img_array = tf.expand_dims(img_array, 0)
classes = model.predict(img_array)
if classes[0]>0:
    print("Bu bir köpektir")
    cv2.putText(img, "kopek",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
else:
    print("Bu bir kedidir")
    cv2.putText(img, "Kedi",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
cv2.imshow("sonuc",img)
cv2.waitKey(0)
cv2.destroyAllWindows
