# DeepFace Model & Detector Comparison

Bu proje, farklı yüz tanıma modelleri ve yüz algılama yöntemlerini kullanarak yüz doğrulama işlemi yapar. DeepFace kütüphanesini kullanarak belirli bir veri kümesi üzerinde modellerin doğruluk oranlarını hesaplar ve karşılaştırır. Ardından, elde edilen sonuçları bir tablo olarak görselleştirir.

## İçindekiler
- [Gereksinimler](#gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Çalışma Mantığı](#çalışma-mantığı)
- [Çıktılar](#çıktılar)

---
## Gereksinimler
Aşağıdaki kütüphanelerin yüklenmiş olması gerekmektedir:

- `pandas`
- `os`
- `deepface`
- `tqdm`
- `torch`
- `tensorflow`
- `matplotlib`

Eğer bu kütüphaneler sisteminizde yüklü değilse aşağıdaki komut ile yükleyebilirsiniz:

```sh
pip install pandas deepface tqdm torch tensorflow matplotlib
```

---

## Kurulum
1. Proje dizininde bir `dataset` klasörü oluşturun ve karşılaştırmak istediğiniz yüz görüntülerini içine yerleştirin.
2. Yüz karşılaştırmalarını içeren `master.csv` dosyasını hazırlayın. Bu dosya aşağıdaki formatta olmalıdır:

```
file_x,file_y,Decision
image1.jpg,image2.jpg,Yes
image3.jpg,image4.jpg,No
```

- **file_x**: İlk görüntünün dosya adı.
- **file_y**: İkinci görüntünün dosya adı.
- **Decision**: Görsellerin aynı kişiye ait olup olmadığını belirten değer (`Yes` veya `No`).

---

## Kullanım
Bu proje iki ana betikten oluşmaktadır:

### 1. Model Performansını Ölçme (`deepface_comparison.py`)
Aşağıdaki komutu çalıştırarak yüz tanıma modellerinin doğruluk oranlarını hesaplayabilirsiniz:

```sh
python deepface_comparison.py
```

Bu betik çalıştırıldığında:
- `master.csv` dosyasındaki her iki görüntüyü farklı model ve yüz algılama yöntemleriyle karşılaştırır.
- Sonuçları `deepface_comparison_results.csv` dosyasına kaydeder.

### 2. Sonuçları Görselleştirme (`table.py`)
Hesaplanan doğruluk oranlarını tablo halinde görmek için şu komutu çalıştırabilirsiniz:

```sh
python table.py
```

Bu betik çalıştırıldığında:
- `deepface_comparison_results.csv` dosyasını okur.
- Model ve yüz algılama yöntemlerini içeren bir pivot tablosu oluşturur.
- Tabloyu belirlenen sıraya göre düzenler ve matplotlib kullanarak görselleştirir.

---

## Çalışma Mantığı
### `deepface_comparison.py`
1. `models` ve `detectors` listeleri içindeki yüz tanıma modelleri ve yüz algılama yöntemleri belirlenir.
2. `master.csv` dosyası okunur ve her satır için:
   - İlgili model ve yüz algılama yöntemi kullanılarak `DeepFace.verify()` fonksiyonu çağrılır.
   - Gerçek sonuç (CSV'deki `Decision` sütunu) ile modelin tahmini karşılaştırılır.
   - Doğru tahmin sayısı artırılır.
3. Tüm modeller ve algılama yöntemleri için doğruluk oranı hesaplanır ve CSV dosyasına kaydedilir.

### `table.py`
1. `deepface_comparison_results.csv` dosyası okunur ve `pandas` DataFrame olarak yüklenir.
2. `pivot()` fonksiyonu kullanılarak `Detector` değerleri satır indeksleri, `Model` değerleri sütun olarak ayarlanır.
3. Algılayıcılar (`Detector`) ve modeller (`Model`) önceden belirlenen sıraya göre düzenlenir:
   
   **Algılayıcı Sırası:**
   - `retinaface`
   - `mtcnn`
   - `dlib`
   - `ssd`
   
   **Model Sırası:**
   - `ArcFace`
   - `VGG-Face`
   - `Dlib`
   - `Facenet`
   - `GhostFaceNet`

4. Matplotlib kullanılarak tablo çizilir ve ekrana görsel olarak gösterilir.

---

## Çıktılar


### `deepface_comparison.py` Çıktısı
Betik tamamlandığında `deepface_comparison_results.csv` dosyasında aşağıdaki gibi bir çıktı oluşur:

```
Model,Detector,Accuracy
VGG-Face,retinaface,85.23
Facenet,mtcnn,88.45
ArcFace,dlib,79.12
... (devam eder)
```

Bu dosya, her model ve yüz algılama yöntemi kombinasyonunun doğruluk oranını gösterir.

### `table.py` Çıktısı
Bu betik çalıştırıldığında ekrana aşağıdaki gibi bir tablo çizilir:

```
| Detector  | ArcFace | VGG-Face | Dlib | Facenet | GhostFaceNet |
|-----------|---------|----------|------|---------|--------------|
| retinaface| 85.23   | 87.34    | 78.12| 88.45   | 80.50        |
| mtcnn     | 86.12   | 89.45    | 79.56| 87.98   | 81.23        |
| dlib      | 82.34   | 85.12    | 77.89| 86.76   | 79.45        |
| ssd       | 80.56   | 84.23    | 75.45| 85.67   | 78.12        |
```

Bu tablo, farklı model ve yüz algılama yöntemlerinin doğruluk oranlarını göstermektedir.


