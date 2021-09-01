
#Bu modelde y=3x +1 i eğitelim

#ilk olarak kütüphaneler aktifleştirilir
import tensorflow as tf
import numpy as np
from tensorflow import keras

#veriler ve etiketleri hazırlanır
x_train = np.array([1,2,3,4,5,6,7], dtype = float)
y_train = np.array([4,7,10,13,16,19,22], dtype = float)

#model oluşturulup eğitilir
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss = "mean_squared_error")
model.fit(x_train,y_train, epochs=50)

#Kullanıcıdan alınan bir değer ile test edilir
veri = float(input("Bir deger giriniz:"))

#Beklenen ve modelden elde edilen sonuç karşılaştırılır
print("Beklenen sonuç:" + str(veri*3 + 1))

sonuc = model.predict([veri])
print("Elde edilen sonuç" + str(sonuc))