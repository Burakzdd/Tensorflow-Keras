#Bu çalışma da ise çalışma 2 ve3'ten alışık olunan modele yeni katmanlar eklenmektedir.
import tensorflow as tf

#Veri seti alınarak eğitim için uygun hale getirilir.
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

#Model oluşturulur burada yeni olarak Conv2D ve MaxPooling2D katmanları eklenir.
model = tf.keras.models.Sequential([
  #Conv2D katmanı verilen boyuta göre görseli filtrelemektedir.
  # Yani (3,3) lük bir matrisi piksellerde dolaştırarak filtreleme işlemi yapmaktadır.
  # Her bir kenardan birer birim kısaltarak örneğin (28,28) boyutlarındaki görseli (26,26) boyutlarıa kısaltır 
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  #MaxPooling katmanında ise havuzlama işlemi diyede geçen görseli özelliklerini bozmadan yarıya indirme işlemi yapılmaktadır.
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#summary komutu ile modelin özeti gözlemlenmektedir
model.summary()

model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)