import tensorflow as tf

#her bir epoch'ta callback sınıfı çağırılarak isteenen doğruluk ya da kayıp değerine ulaşılmışsa eğitim durdurulur.
# Burda eğer eğitimin doğruluğu %80 olmuşsa eğitimi bitir diyoruz. 
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9):
      print("\n%80 doğruluğa ulaşıldı, bu nedenle eğitim duruduruluyor!")
      self.model.stop_training = True
callbacks = myCallback()

#Alışılan işlemler yapılır veri seti alınır model oluşturulu, eğitime başlanır..
mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])