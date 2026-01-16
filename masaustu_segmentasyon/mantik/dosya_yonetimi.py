import os
import glob

def goruntu_listesini_getir(klasor_yolu: str) -> list[str]:
    """
    Verilen klasördeki tüm .png dosyalarını listeler.
    Sadece dosya adlarını döndürür.
    """
    if not os.path.exists(klasor_yolu):
        return []
    
    # Sadece .png dosyalarını al (Büyük/küçük harf duyarsız)
    # Genişletilmiş format desteği (kullanıcı hatası veya farklı formatlar için)
    uzantilar = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    
    resim_dosyalari = []
    print(f"DEBUG: '{klasor_yolu}' taranıyor (Recursive)...")
    
    for kok, dizinler, dosyalar in os.walk(klasor_yolu):
        for dosya in dosyalar:
            if dosya.lower().endswith(uzantilar):
                # Tam yol
                tam_yol = os.path.join(kok, dosya)
                # Klasör yoluna göre göreceli yol
                goreceli_yol = os.path.relpath(tam_yol, klasor_yolu)
                resim_dosyalari.append(goreceli_yol)
    
    print(f"DEBUG: '{klasor_yolu}' konumunda ve alt klasörlerinde toplam {len(resim_dosyalari)} uygun dosya bulundu.")
    
    # Sadece dosya adlarını (göreceli yolları) al ve sırala
    dosya_adlari = sorted(resim_dosyalari)
    return dosya_adlari

def maske_yolunu_olustur(goruntu_klasoru: str, dosya_adi: str) -> str:
    """
    Görüntü klasörüne göre maske dosyasının tam yolunu oluşturur.
    Maskeler, görüntü klasörünün yanındaki veya altındaki 'maskeler' klasöründe tutulur.
    Burada proje yapısına uygun olarak:
    Eğer görüntü klasörü .../veri/goruntuler ise, maske klasörü .../veri/maskeler olsun.
    Ancak kullanıcı rastgele bir klasör seçerse, o klasörün içinde 'maskeler' diye bir klasör oluşturup oraya kaydedelim.
    """
    # Basitlik için: Seçilen klasörün içinde (yanında) olsun.
    # Kullanıcı isteği: "Her iki dosya da: Orijinal görüntünün bulunduğu klasöre"
    maske_klasoru = goruntu_klasoru
    
    # ornek.png -> ornek_mask.png
    isim, uzanti = os.path.splitext(dosya_adi)
    maske_adi = f"{isim}_mask{uzanti}"
    
    return os.path.join(maske_klasoru, maske_adi)

def maske_var_mi(maske_yolu: str) -> bool:
    """Maske dosyasının var olup olmadığını kontrol eder."""
    return os.path.exists(maske_yolu)
