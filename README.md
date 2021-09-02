# Tensorflow-Keras
Tensorflow Architecture and Classification Examples (Tensorflow Mimarisi ve Sınıflandırma Örnekleri)

Tensorflow, yapay zekanın derin öğrenme çalışmaları yapan geliştiriciler için üretilmiş açık kaynaklı bir kütüphanedir. Sistem özelinde verilen verilerin kodlanması ve ayırt edilmeleri sağlanmaktadır. Her ne kadar Deep Learning uygulamaları bünyesinde kullanımı görülse de TensorFlow başlıca çok daha geniş bir alanı kapsamına almaktadır.
Keras ise Tensorflow 2 nin üst düzey API’sidir (Uygulama Programlama Arayüzü). Keras, python’da yazılmış açık kaynaklı bir sinir ağı kütüphanesidir. Keras, tensorflow için bir arabirim görevi görür. CNN ve ANN gibi ağlar ile hızlı deneyler sağlamak için tasarlanmıştır.
Some basic Git commands are:

# Tensorflow ile bir kedi köpek sınıflandırıcısı
!!Bu repoda ki kedi-köpek-sınıflandırma dosyası içersinde kodlar mevcuttur.
İlk olarak kedi ve köpek görsellerinden oluşan veri setine sahip olmamız gerekmektedir. Bu çalışma da kaggle üzerindeki kedi köpek veri seti ile çalışılmıştır. Bu veri setine bu adresten https://www.kaggle.com/c/dogs-vs-cats ulaşabilirsiniz ya da aşağıdaki kod ile terminal üzerinden indirebilirsiniz.
```
wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
```
Veri setini incelediğimiz de kullanıma hazır şekilde train ve validation olarak ayrılmış şekilde olup aynı zamanda bu klasörlerde kendi için cats ve dogs diye ayrılmaktadır. Yani elde bulunan veriler tamamen kullanıma hazırdır.

## Verileri Hazırlama
Veriler kullanıma hazır bir şekilde önceden ayrıldığı için bu kısımda sadece verileri okuyoruz.
```
veri_adresi = '/home/burakzdd/Desktop/cats_and_dogs_filtered'
train_veri = os.path.join(veri_adresi, 'train')
validation_veri = os.path.join(veri_adresi, 'validation')
```
## **Model Oluşturma** 
Verilerin özelliklerini çıkarabilmemize yarayacak olan katmanlarımızı inşaa ediyoruz.
```
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
```
Burada katmanları tek tek inceleyecek olursak;
### Convulition2D Katmanı
Bu katman belirlenen boyuttaki bir filtreyi görsel üzerinde gezdirerek görüntüyü kenarlarından filtrelemektedir. Mesela bu çalışmada (3,3)lük bir filtre kullandık. Katman görsel üzreinden 3*3lük bir filtre ile geçtiğinde her bir kenardan birer birim görselin bpyutunu düşürmektedir. 150,150 olan görsel 148,148 boyutlarına düşürülmektedir.

### MaxPooling2D Katmanı
Havuzlama işlemi yapan bu katman görselin özelliklerini koruyarak tamamen boyutunu yarıya düşürmektedir.

### Flatten Katmanı
Bu katman modeli düzleştirmektedir. Yani tek boyutlu bir dizi haline çevirmektedir.

### Dense
Bu katman ise ağımıza nöronlar ekleyerek ağı daha karmaşık hale getirmektedir. Bu da ağımızın daha güçlü olmasını sağlamaktadır. Örneğin burada katmanlar arasına 512 tane nöron eklemiş olduk.

### Relu Aktivasyonu
Görüldüğü üzere çoğu katmanda activation='relu' ifadesi mevcuttur. Bu aktivasyon eğer gelen değer sıfırdan küçükse sıfıra eşitlemekte, sıfırdan büyükse ise olduğu gibi bırakmaktadır.

### Sigmoid aktivasyonu
İkili sınıflandırmalarda kullanılmaktadır. Eğer ilk sınıfa aitse döndürdüğü değer pozitif ikinci sınıfa aitse döndürdüğü değer negatif olarak düşünülmektedir. Bu basit yapısı sayesinde ikili sınıflandırmalarda gereçkten çok başarılı bir şekilde çalışmaktadır.

Bu oluşturulan model aşağıdaki komut ile gözlemlenebilrmektedir.
```
model.summary()
```
Modeli derleme işlemi ise aşağıdaki kod ile gerçekleştrilmektedir. Burada ikili sınıflandırma yapılacağı için kayıp fonskiyonu binary_crossentropy olarak belirlenmiştir.
```
model.compile(optimizer=RMSprop(lr=0.001),loss= 'binary_crossentropy',metrics=['accuracy'])
```
![alt text](https://github.com/Burakzdd/Tensorflow-Keras/blob/main/g%C3%B6rseller/modelSummary.png)
## Eğitim
Eğitime geçmeden önce tüm görselleri normalleştirmemiz gerekmektedir bu işlem ImageDataGenerator fonksiyonu ile kolaylıkla gereçkeleştirebilmektedir. Aşağıda görüldüğü  gibi fonksiyon içinde görseli 1/255 boyutlarında yeniden şekillendirilmektedir.
```
train_datagen = ImageDataGenerator(rescale=1.0/255. )
validation_datagen = ImageDataGenerator(rescale=1.0/255. )
```
Daha sonra veriler tamamiyle eğitime hazır hale getirilmektedir. Burada batch_size tek seferde kaç görselin eğtime gireceğini ifade etmektedir. class_mode='binary' olarak işaretlendiği kısımda ise sınıflandırmanın ikili sınıflandırma olarak yapılacağını, target_size ise görselin giriş görüntüsünü ifade etmektedir.
```
train_generator = train_datagen.flow_from_directory(train_veri,batch_size=20, class_mode='binary',target_size=(150,150))
validation_generator = validation_datagen.flow_from_directory(validation_veri, batch_size=20, class_mode='binary', target_size=(150,150))
```
Bu işlem de tamamalndıktan sorna eğitime başlanabilir. Burada ilk iki paremetreye hazırlanan verilerimizi, daha sonra ise adım sayıları girilmektedir. Son paramterede yer alan verbose'u yazmak zorunda değilsiniz. Eğer yazmazsanız bu 1 olarak belirlenmiştir. 2 olarak belirlendiğinde sadece eğitim sırasında uzun uzun epochs'un % si görülmemektedir
```
model.fit(train_generator,
        validation_data=validation_generator,
        steps_per_epoch=100, 
        epochs=15, 
        validation_steps=50, 
        verbose=2)
```
## Modeli Kaydetme
Modelin eğitildikten sonra kaybolmaması için kaydedilmesi gerekmektedir. Bu işlem .save komutu ile kolaylıkla yapılabilmektedir. İşlem sonunda .h5 uzantılı model bilgisayarınıza belirlediğiniz isim ile kaydolmaktadır.
```
model.save('cat_dog_model.h5')
```
![alt text](https://github.com/Burakzdd/Tensorflow-Keras/blob/main/g%C3%B6rseller/modelsave.png)
## Modeli Kullanma
Kayıtlı olan bir model kod içerisinde yüklenerek tekrardan kullanılmaktadır. Bu işlem aşağıdaki load_model komutu ile yapılmaktadır. Parantez içirisine modelin kayıtlı olduğu dizinin adresi girilmelidir.
```
model = tf.keras.models.load_model('/home/burakzdd/cat_dog_model.h5')
```
## Test Etme
Yapılan modelin test aşamasına geçmeden önce sınıflandırılacak görüntü hazır hale getirilmelidir. Aşağıdaki koda bakacak olursak ilk olarak daha sonra bastırmak için görsel okundu. Görselin tekrardan okunduğu esas kısımda ise görsel ön işlemden geçirilmektedir. Burda görüntü 150,150 boyutlarına düşürülmektedir. Daha sonra dizi formuna çevrilmektedir.
```
img = cv2.imread("/home/burakzdd/Desktop/cats_and_dogs_filtered/validation/cats/cat.2010.jpg")
img2 = tf.keras.preprocessing.image.load_img(
    "/home/burakzdd/Desktop/cats_and_dogs_filtered/validation/cats/cat.2010.jpg", target_size=(150,150)
)
img_array = tf.keras.preprocessing.image.img_to_array(img2)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis
```
Ön işlemler bittikten sonra model tahmin işlemine gönderilerek görselin sınıf tahmini yapılmaktadır.
```
classes = model.predict(img_array)
```
Yapılan sınıf tahmininin çıktısına aşağıdaki gibi ulaşılabilmektedir. İkili sigmoid sınıflandırması yapıldığı için eğer tahmin değeri sıfırdan büyükse görsel ilk sınafa, sıfırdan küçükse ikinci sınıfa aittir. Burada ayrıca ilk başta okunan görselin üzerine sınıf etiketi de bastırılarak sınıflandırma güzel bir şekilde gözlemlenmektedir.
```
if classes[0]>0:
    print("Bu bir köpektir")
    cv2.putText(img, "kopek",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
else:
    print("Bu bir kedidir")
    cv2.putText(img, "Kedi",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
```
![alt text](https://github.com/Burakzdd/Tensorflow-Keras/blob/main/g%C3%B6rseller/kopek.png)
![alt text](https://github.com/Burakzdd/Tensorflow-Keras/blob/main/g%C3%B6rseller/kedi.png)
