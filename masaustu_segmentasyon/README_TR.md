# Masaüstü Segmentasyon Uygulaması

Bu proje, ayak röntgeni (PNG) görüntülerinde "Kalkaneus" kemiğini segmente etmek için geliştirilmiş bir masaüstü uygulamasıdır. PyQt6 tabanlı bir arayüz ve FastAPI tabanlı yerel bir API sunar.

## Özellikler

- **Görüntü Görüntüleme:** Klasördeki PNG dosyalarını listeleme ve görüntüleme.
- **Segmentasyon:** Fırça ve silgi araçları ile manuel maske oluşturma.
- **Maske Overlay:** Yarı saydam kırmızı maske katmanı.
- **Zoom/Pan:** Görüntü üzerinde yakınlaştırma ve gezinme.
- **Lokal API:** Görüntü ve maskelere erişim sağlayan REST API.

## Kurulum

1. Python 3.9 veya daha yeni bir sürümün yüklü olduğundan emin olun.
2. Bir sanal ortam oluşturun (önerilir):
   ```bash
   python -m venv venv
   # Windows için aktivasyon:
   venv\Scripts\activate
   ```
3. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

## Çalıştırma

Uygulamayı başlatmak için proje ana dizininde şu komutu çalıştırın:

```bash
python main.py
```

## Kullanım

1. Uygulama açıldığında **"Görüntü Klasörü Seç"** butonuna tıklayarak PNG dosyalarınızın bulunduğu klasörü seçin.
2. Sol taraftaki listeden bir görüntü seçin.
3. **Fırça** aracı ile kemik bölgesini boyayın. **Silgi** ile hataları düzeltebilirsiniz.
4. **Fırça Boyutu** ve **Maske Saydamlığı** ayarlarını sağ panelden değiştirebilirsiniz.
5. **"Maskeyi Kaydet"** butonuna basarak çalışmanızı kaydedin.
   - Maskeler, seçtiğiniz klasörün içinde `maskeler` adlı bir alt klasöre kaydedilir.
   - Dosya adı formatı: `ornek_mask.png`

## API Kullanımı

Uygulama çalışırken `http://127.0.0.1:8000` adresinde bir API sunucusu çalışır.

- **Dokümantasyon:** `http://127.0.0.1:8000/docs`
- **Endpointler:**
  - `GET /api/goruntuler`: Görüntü listesini döner.
  - `GET /api/goruntu/{dosya_adi}`: Görüntü dosyasını döner.
  - `GET /api/maske/{dosya_adi}`: Maske dosyasını döner.
  - `POST /api/maske/{dosya_adi}`: Maske verisini (base64) kaydeder.

## Proje Yapısı

- `masaustu_segmentasyon/`: Ana proje klasörü.
  - `arayuz/`: GUI kodları.
  - `sunucu/`: API kodları.
  - `mantik/`: İş mantığı ve dosya işlemleri.
  - `veri/`: Örnek veri klasörleri.
