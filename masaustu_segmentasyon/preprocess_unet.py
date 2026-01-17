"""
U-Net Dataset Preprocessing Script
===================================

Medical Image Processing için hazırlanmış veri seti ön işleme scripti.

Giriş:
    input_images/  : 0001.png, 0002.png, ... 
    input_masks/   : 0001_mask.png, 0002_mask.png, ...

Çıkış:
    output_images/ : 0001.png, 0002.png, ... (512x512)
    output_masks/  : 0001_mask.png, 0002_mask.png, ... (512x512)

Kullanım:
    python preprocess_unet.py
"""

import cv2
import numpy as np
import os
from pathlib import Path


def resize_and_pad(image, target_size=512, interpolation=cv2.INTER_LINEAR):
    """
    Görüntüyü 512x512'ye aspect ratio koruyarak dönüştürür.
    Eksik kısımlara siyah padding ekler.
    
    Args:
        image: Giriş görüntüsü (numpy array)
        target_size: Hedef boyut (varsayılan: 512)
        interpolation: OpenCV interpolasyon metodu
    
    Returns:
        numpy.ndarray: 512x512 işlenmiş görüntü
    """
    h, w = image.shape[:2]
    
    # En uzun kenarı 512'ye sığdır (aspect ratio koru)
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    # Resize işlemi
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Siyah canvas oluştur
    if len(image.shape) == 2:  # Grayscale
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    else:  # RGB/BGR
        canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=np.uint8)
    
    # Merkezde konumlandır (padding ekle)
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    
    # Resize edilmiş görüntüyü canvas'a yerleştir
    if len(image.shape) == 2:
        canvas[top:top + new_h, left:left + new_w] = resized
    else:
        canvas[top:top + new_h, left:left + new_w, :] = resized
    
    return canvas


def process_image_mask_pair(image_path, mask_path, output_image_path, output_mask_path, target_size=512):
    """
    Görüntü-maske çiftini işler ve kaydeder.
    
    Args:
        image_path: Kaynak görüntü yolu
        mask_path: Kaynak maske yolu
        output_image_path: Hedef görüntü yolu
        output_mask_path: Hedef maske yolu
        target_size: Hedef boyut
    
    Returns:
        bool: İşlem başarılı ise True
    """
    try:
        # Görüntüyü oku
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"    ✗ Görüntü okunamadı: {os.path.basename(image_path)}")
            return False
        
        # Maskeyi oku (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"    ✗ Maske okunamadı: {os.path.basename(mask_path)}")
            return False
        
        # Boyut kontrolü
        if image.shape[:2] != mask.shape[:2]:
            print(f"    ⚠ Boyut uyumsuzluğu! Görüntü: {image.shape[:2]}, Maske: {mask.shape[:2]}")
        
        # Görüntüyü işle (INTER_LINEAR - detayları koru)
        processed_image = resize_and_pad(image, target_size, cv2.INTER_LINEAR)
        
        # Maskeyi işle (INTER_NEAREST - binary kal, bulanıklaşma yok)
        processed_mask = resize_and_pad(mask, target_size, cv2.INTER_NEAREST)
        
        # Maskeyi binary tut (threshold uygula)
        _, processed_mask = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Kaydet
        cv2.imwrite(output_image_path, processed_image)
        cv2.imwrite(output_mask_path, processed_mask)
        
        return True
        
    except Exception as e:
        print(f"    ✗ Hata: {e}")
        return False


def main():
    """
    Ana preprocessing fonksiyonu.
    """
    print("=" * 70)
    print("U-Net Dataset Preprocessing (512x512)")
    print("=" * 70)
    print()
    
    # Klasör yolları
    input_images_dir = "input_images"
    input_masks_dir = "input_masks"
    output_images_dir = "output_images"
    output_masks_dir = "output_masks"
    
    # Giriş klasörlerini kontrol et
    if not os.path.exists(input_images_dir):
        print(f"✗ HATA: '{input_images_dir}' klasörü bulunamadı!")
        input("\nKapatmak için Enter'a basın...")
        return
    
    if not os.path.exists(input_masks_dir):
        print(f"✗ HATA: '{input_masks_dir}' klasörü bulunamadı!")
        input("\nKapatmak için Enter'a basın...")
        return
    
    # Çıkış klasörlerini oluştur
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)
    
    # input_images klasöründeki dosyaları listele
    image_files = sorted([f for f in os.listdir(input_images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    
    if not image_files:
        print(f"✗ UYARI: '{input_images_dir}' klasöründe görüntü dosyası bulunamadı.")
        input("\nKapatmak için Enter'a basın...")
        return
    
    print(f"Toplam {len(image_files)} adet görüntü bulundu.")
    print(f"Hedef boyut: 512x512")
    print("-" * 70)
    
    success_count = 0
    missing_mask_count = 0
    error_count = 0
    
    # Her görüntü için eşleşen maskeyi bul ve işle
    for idx, image_filename in enumerate(image_files, 1):
        # Dosya adından numarayı çıkar (örn: 0001.png -> 0001)
        base_name = os.path.splitext(image_filename)[0]
        extension = os.path.splitext(image_filename)[1]
        
        # Eşleşen maske dosya adını oluştur (örn: 0001_mask.png)
        mask_filename = f"{base_name}_mask.png"
        
        print(f"\n[{idx}/{len(image_files)}] İşleniyor: {image_filename}")
        
        # Dosya yolları
        image_path = os.path.join(input_images_dir, image_filename)
        mask_path = os.path.join(input_masks_dir, mask_filename)
        
        # Maske var mı kontrol et
        if not os.path.exists(mask_path):
            print(f"    ⚠ Maske bulunamadı: {mask_filename}")
            missing_mask_count += 1
            continue
        
        print(f"    ✓ Eşleşen maske: {mask_filename}")
        
        # Çıkış dosya yolları (orijinal isimleri koru)
        output_image_path = os.path.join(output_images_dir, image_filename)
        output_mask_path = os.path.join(output_masks_dir, mask_filename)
        
        # İşle
        if process_image_mask_pair(image_path, mask_path, output_image_path, output_mask_path):
            print(f"    ✓ Başarılı")
            success_count += 1
        else:
            error_count += 1
    
    # Özet
    print("\n" + "=" * 70)
    print("İşlem Tamamlandı!")
    print("-" * 70)
    print(f"  ✓ Başarılı         : {success_count}")
    print(f"  ⚠ Maske Bulunamadı : {missing_mask_count}")
    print(f"  ✗ Hata             : {error_count}")
    print("=" * 70)
    
    print(f"\nİşlenmiş dosyalar:")
    print(f"  • Görüntüler : {output_images_dir}/")
    print(f"  • Maskeler   : {output_masks_dir}/")
    print()


if __name__ == "__main__":
    main()
    input("Kapatmak için Enter'a basın...")
