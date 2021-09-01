# Tensorflow-Keras
Tensorflow Architecture and Classification Examples (Tensorflow Mimarisi ve Sınıflandırma Örnekleri)

Tensorflow, yapay zekanın derin öğrenme çalışmaları yapan geliştiriciler için üretilmiş açık kaynaklı bir kütüphanedir. Sistem özelinde verilen verilerin kodlanması ve ayırt edilmeleri sağlanmaktadır. Her ne kadar Deep Learning uygulamaları bünyesinde kullanımı görülse de TensorFlow başlıca çok daha geniş bir alanı kapsamına almaktadır.
Keras ise Tensorflow 2 nin üst düzey API’sidir (Uygulama Programlama Arayüzü). Keras, python’da yazılmış açık kaynaklı bir sinir ağı kütüphanesidir. Keras, tensorflow için bir arabirim görevi görür. CNN ve ANN gibi ağlar ile hızlı deneyler sağlamak için tasarlanmıştır.

# Tensorflow ile bir kedi köpek sınıflandırıcısı
!!Bu repoda ki kedi-köpek-sınıflandırma dosyası içersinde kodlar mevcuttur.
İlk olarak kedi ve köpek görsellerinden oluşan veri setine sahip olmamız gerekmektedir. Bu çalışma da kaggle üzerindeki kedi köpek veri seti ile çalışılmıştır. Bu verisetine bu adresten ulaşabilirsiniz ya da aşağıdaki kod ile terminal üzerinden indirebilirsiniz.
wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

Veri setini incelediğimiz de kullanıma hazır şekilde train ve validation olarak ayrılmış şekilde olup aynı zamanda bu klasörlerde kendi için cats ve dogs diye ayrılmaktadır. Yani elde bulunan veriler tamamen kullanıma hazırdır.

## Verileri Hazırlama
Veriler kullanıma hazır bir şekilde önceden ayrıldığı için bu kısımda sadece verileri okuyoruz.

veri_adresi = '/home/burakzdd/Desktop/cats_and_dogs_filtered'

train_veri = os.path.join(veri_adresi, 'train')
validation_veri = os.path.join(veri_adresi, 'validation')

## Model Oluşturma 
Verilerin özelliklerini çıkarabilmemize yarayacak olan katmanlarımızı inşaa ediyoruz.
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
Burada katmanları tek tek inceleyecek olursak;
### Convulition2D Katmanı
Bu katman belirlenen boyuttaki bir filtreyi görsel üzerinde gezdirerek görüntüyü kenarlarından filtrelemektedir. Mesela bu çalışmada (3,3)lük bir filtre kullandık. Katman görsel üzreinden 3*3lük bir filtre ile geçtiğinde her bir kenardan birer birim görselin bpyutunu düşürmektedir. 150,150 olan görsel 148,148 boyutlarına düşürülmektedir.

### MaxPooling2D Katmanı
Havuzlama işlemi yapan bu katman görselin özelliklerini koruyarak tamamen boyutunu yarıya düşürmektedir.

### Flatten Katmanı
Bu katman görseli düzleştirmektedir. Yani tek boyutlu bir dizi haline çevirmektedir.

### Dense
Bu katman ise ağımıza nöronlar ekleyerek ağı daha karmaşık hale getirmektedir. Bu da ağımızın daha güçlü olmasını sağlamaktadır. Örneğin burada katmanlar arasına 512 tane nöron eklemiş olduk.

### Relu Aktivasyonu
Görüldüğü üzere çoğu katmanda activation='relu' ifadesi mevcuttur. Bu aktivasyon eğer gelen değer sıfırdan küçükse sıfıra eşitlemekte, sıfırdan büyükse ise olduğu gibi bırakmaktadır.

### Sigmoid aktivasyonu
İkili sınıflandırmalarda kullanılmaktadır. Eğer ilk sınıfa aitse döndürdüğü değer pozitif ikinci sınıfa aitse döndürdüğü değer negatif olarak düşünülmektedir. Bu basit yapısı sayesinde ikili sınıflandırmalarda gereçkten çok başarılı bir şekilde çalışmaktadır.

Bu oluşturulan model aşağıdaki komut ile gözlemlenebilrmektedir.
model.summary()

Modeli derleme işlemimodel.compile(optimizer=RMSprop(lr=0.001),loss= 'binary_crossentropy',metrics=['accuracy']) ise aşağıdaki kod ile gerçekleştrilmektedir. Burada ikili sınıflandırma yapılacağı için kayıp fonskiyonu binary_crossentropy olarak belirlenmiştir.
