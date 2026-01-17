"""
Binary Mask Preprocessing Script for U-Net Training
====================================================

Bu script, farklı boyutlardaki binary segmentasyon maskelerini 512x512 piksel 
boyutuna dönüştürür. Aspect ratio korunur ve eksik kısımlar siyah (0) ile doldurulur.

Kullanım:
    python preprocess_masks.py

Gereksinimler:
    - opencv-python (cv2)
    - numpy

Klasörler:
    - input_masks/   : Kaynak mask dosyaları (.png)
    - output_masks_512/ : İşlenmiş 512x512 maskeler
"""

import cv2
import numpy as np
import os
from pathlib import Path


def preprocess_binary_mask(input_path, output_path, target_size=512):
    """
    Binary maskeyi 512x512'ye dönüştürür (aspect ratio korunur, padding eklenir).
    
    Args:
        input_path: Kaynak mask dosyası yolu
        output_path: Hedef mask dosyası yolu
        target_size: Hedef boyut (varsayılan: 512)
    
    Returns:
        bool: İşlem başarılı ise True
    """
    try:
        # Maskeyi oku (grayscale)
        mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            print(f"HATA: Dosya okunamadı: {input_path}")
            return False
        
        # Mevcut boyutları al
        h, w = mask.shape
        
        # Aspect ratio'yu koru - en uzun kenarı 512'ye sığacak şekilde ölçeklendir
        if h > w:
            # Yükseklik daha uzun
            new_h = target_size
            new_w = int(w * (target_size / h))
        else:
            # Genişlik daha uzun veya eşit
            new_w = target_size
            new_h = int(h * (target_size / w))
        
        # INTER_NEAREST kullan (binary maskelerde blur olmasın)
        resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # 512x512 siyah (0) canvas oluştur
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
        
        # Resmi ortala - padding hesapla
        top = (target_size - new_h) // 2
        left = (target_size - new_w) // 2
        
        # Resize edilmiş maskeyi canvas'a yerleştir
        canvas[top:top+new_h, left:left+new_w] = resized_mask
        
        # Binary değerleri kontrol et (sadece 0 ve 255 olmalı)
        # Eğer ara değerler varsa (interpolasyon hatası), threshold uygula
        _, canvas = cv2.threshold(canvas, 127, 255, cv2.THRESH_BINARY)
        
        # Hedef klasörü oluştur (yoksa)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Kaydet
        cv2.imwrite(output_path, canvas)
        
        return True
        
    except Exception as e:
        print(f"HATA ({os.path.basename(input_path)}): {e}")
        return False


def batch_process(input_folder="input_masks", output_folder="output_masks_512", target_size=512):
    """
    Klasördeki tüm PNG dosyalarını işler.
    
    Args:
        input_folder: Kaynak klasör
        output_folder: Hedef klasör
        target_size: Hedef boyut
    """
    # Kaynak klasörü kontrol et
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"HATA: Kaynak klasör bulunamadı: {input_folder}")
        print(f"Lütfen '{input_folder}' klasörünü oluşturun ve mask dosyalarını içine koyun.")
        return
    
    # PNG dosyalarını bul
    png_files = list(input_path.glob("*.png"))
    
    if not png_files:
        print(f"UYARI: '{input_folder}' klasöründe PNG dosyası bulunamadı.")
        return
    
    print(f"Toplam {len(png_files)} adet PNG dosyası bulundu.")
    print(f"İşleniyor... (Hedef boyut: {target_size}x{target_size})")
    print("-" * 60)
    
    success_count = 0
    error_count = 0
    
    for idx, png_file in enumerate(png_files, 1):
        input_file = str(png_file)
        output_file = os.path.join(output_folder, png_file.name)
        
        print(f"[{idx}/{len(png_files)}] İşleniyor: {png_file.name}...", end=" ")
        
        if preprocess_binary_mask(input_file, output_file, target_size):
            print("✓ Başarılı")
            success_count += 1
        else:
            print("✗ HATA")
            error_count += 1
    
    print("-" * 60)
    print(f"İşlem Tamamlandı!")
    print(f"  ✓ Başarılı: {success_count}")
    print(f"  ✗ Hata: {error_count}")
    print(f"\nİşlenmiş maskeler: {output_folder}/")


if __name__ == "__main__":
    print("=" * 60)
    print("Binary Mask Preprocessing (512x512)")
    print("=" * 60)
    print()
    
    # Script'in bulunduğu dizindeki klasörleri kullan
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "input_masks")
    output_folder = os.path.join(script_dir, "output_masks_512")
    
    # Batch işlemi başlat
    batch_process(input_folder, output_folder, target_size=512)
    
    print()
    input("Kapatmak için Enter'a basın...")
