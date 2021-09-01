import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.python.keras.layers.core import Activation

#tensorflow veri setlerinden olan içinde gri renkte 10 farklı çeşitteki kıyafet görselleri olan fashion veri setini alıyoruz
mnist = tf.keras.datasets.fashion_mnist

#verileri eğitim ve test için ikiye ayırıyoruz. (Bu veri setinde toplamda 70bin görsel vardır. Bunun 10bin'i test, 60bin'i eğitim olarak ayrılmaktadır.)
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

#bu veriler 0-1 aralığında normalize edilmektedir.
training_images  = training_images / 255.0
test_images = test_images / 255.0

#model oluşturulur.
model = tf.keras.models.Sequential([
    #görsel düzleştirilir
    tf.keras.layers.Flatten(),
    #katmanlar arasına nöronlar eklenir
    tf.keras.layers.Dense(128, activation="relu"),
    #son katmanda ise softmax sınıflandırıcısı kullanılarak 10 farklı veri için sınıflandırma işlemi eklenir
    tf.keras.layers.Dense(10, activation = "softmax")
])
#model derlenir

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)


#şimdi ise eğitilen modeli değerlendirelim
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)
#burda alınan değerlerden en büyük olan index tahmin sınıf değerimiz.
#İlk görsele göre işlem yapalım:
print(classifications[0])
#bu değerlerden maksimum değere sahip olan indesi alalım ve karşılaştıralım.
print("Olması gerekn sııf :"+ str(test_labels[0]))
index_max = np.argmax(classifications[0])
print("tahmin edilen sınıf :"+ str(index_max))

