"""
Paired Image-Mask Dataset Preprocessing for Semantic Segmentation
==================================================================

Bu script, X-ray görüntüleri ve karşılık gelen binary maskelerini birlikte 
512x512 boyutuna hizalayarak dönüştürür. Aspect ratio korunur ve aynı padding 
her ikisine de uygulanır.

Kullanım:
    python preprocess_dataset.py

Gereksinimler:
    - opencv-python (cv2)
    - numpy

Klasör Yapısı:
    images_folder/      : Orijinal X-ray görüntüleri
    masks_folder/       : Karşılık gelen binary maskeler
    
    output_images_512/  : İşlenmiş görüntüler
    output_masks_512/   : İşlenmiş maskeler
    debug_overlays/     : (Opsiyonel) Hizalama kontrol görselleri
"""

import cv2
import numpy as np
import os
from pathlib import Path


def find_matching_mask(image_name, masks_folder):
    """
    Görüntü dosyası için eşleşen maske dosyasını bulur.
    
    Eşleştirme stratejileri:
    1. Tam isim eşleşmesi (case1.png -> case1.png)
    2. _mask eklenmişse (case1.png -> case1_mask.png)
    3. _seg eklenmişse (case1.png -> case1_seg.png)
    
    Args:
        image_name: Görüntü dosyası adı (örn: case1.png)
        masks_folder: Maske klasörü yolu
    
    Returns:
        str: Eşleşen maske dosyasının tam yolu veya None
    """
    masks_path = Path(masks_folder)
    base_name = Path(image_name).stem  # Uzantısız dosya adı
    extension = Path(image_name).suffix  # Uzantı (.png, .jpg, vb.)
    
    # Olası maske dosya adları
    possible_names = [
        image_name,  # Aynı isim
        f"{base_name}_mask{extension}",
        f"{base_name}_mask.png",
        f"{base_name}_seg{extension}",
        f"{base_name}_seg.png",
        f"{base_name}.png",  # Her zaman PNG dene
    ]
    
    for mask_name in possible_names:
        mask_path = masks_path / mask_name
        if mask_path.exists():
            return str(mask_path)
    
    return None


def resize_and_pad(image, target_size=512, interpolation=cv2.INTER_LINEAR):
    """
    Görüntüyü target_size x target_size boyutuna dönüştürür.
    Aspect ratio korunur ve eksik kısımlar siyah (0) ile doldurulur.
    
    Args:
        image: Giriş görüntüsü (grayscale veya RGB)
        target_size: Hedef boyut (varsayılan: 512)
        interpolation: OpenCV interpolasyon metodu
    
    Returns:
        tuple: (işlenmiş_görüntü, scale_factor, top_pad, left_pad)
    """
    h, w = image.shape[:2]
    
    # Aspect ratio'yu koru - en uzun kenarı target_size'a sığdır
    if h > w:
        new_h = target_size
        new_w = int(w * (target_size / h))
    else:
        new_w = target_size
        new_h = int(h * (target_size / w))
    
    # Scale factor (debug için)
    scale_factor = new_h / h if h > w else new_w / w
    
    # Resize işlemi
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Canvas oluştur (görüntü grayscale veya RGB olabilir)
    if len(image.shape) == 2:  # Grayscale
        canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    else:  # RGB
        canvas = np.zeros((target_size, target_size, image.shape[2]), dtype=np.uint8)
    
    # Padding hesapla ve ortala
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    
    # Resize edilmiş görüntüyü canvas'a yerleştir
    if len(image.shape) == 2:
        canvas[top:top+new_h, left:left+new_w] = resized
    else:
        canvas[top:top+new_h, left:left+new_w, :] = resized
    
    return canvas, scale_factor, top, left


def create_overlay(image, mask, alpha=0.4, mask_color=(0, 255, 0)):
    """
    Debug için: Maske ile görüntüyü üst üste koyar.
    
    Args:
        image: Orijinal görüntü (grayscale veya RGB)
        mask: Binary maske
        alpha: Maske opaklığı (0.0 - 1.0)
        mask_color: Maske rengi (B, G, R)
    
    Returns:
        np.ndarray: Overlay görüntü
    """
    # Görüntüyü RGB'ye çevir (grayscale ise)
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = image.copy()
    
    # Maskeyi renklendir
    mask_colored = np.zeros_like(img_rgb)
    mask_colored[mask > 127] = mask_color
    
    # Blend
    overlay = cv2.addWeighted(img_rgb, 1.0, mask_colored, alpha, 0)
    
    return overlay


def process_pair(image_path, mask_path, output_image_path, output_mask_path, 
                 target_size=512, save_overlay=False, overlay_path=None):
    """
    Görüntü-maske çiftini işler.
    
    Args:
        image_path: Kaynak görüntü yolu
        mask_path: Kaynak maske yolu
        output_image_path: Hedef görüntü yolu
        output_mask_path: Hedef maske yolu
        target_size: Hedef boyut
        save_overlay: Overlay kaydedilsin mi?
        overlay_path: Overlay dosya yolu (opsiyonel)
    
    Returns:
        bool: Başarılı ise True
    """
    try:
        # Görüntüyü oku (X-ray genelde grayscale, ama RGB de olabilir)
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"  ✗ Görüntü okunamadı: {os.path.basename(image_path)}")
            return False
        
        # Maskeyi oku (grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  ✗ Maske okunamadı: {os.path.basename(mask_path)}")
            return False
        
        # Boyutları kontrol et (eşleşmeli)
        if image.shape[:2] != mask.shape[:2]:
            print(f"  ⚠ Uyarı: Görüntü ve maske boyutları farklı!")
            print(f"    Görüntü: {image.shape[:2]}, Maske: {mask.shape[:2]}")
        
        # Görüntüyü işle (INTER_LINEAR - X-ray detaylarını koru)
        processed_image, scale, top, left = resize_and_pad(
            image, 
            target_size=target_size, 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Maskeyi işle (INTER_NEAREST - binary kal)
        processed_mask, _, _, _ = resize_and_pad(
            mask, 
            target_size=target_size, 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Maskeyi binary tut (threshold uygula)
        _, processed_mask = cv2.threshold(processed_mask, 127, 255, cv2.THRESH_BINARY)
        
        # Hedef klasörleri oluştur
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        
        # Kaydet
        cv2.imwrite(output_image_path, processed_image)
        cv2.imwrite(output_mask_path, processed_mask)
        
        # Overlay kaydet (debug)
        if save_overlay and overlay_path:
            overlay = create_overlay(processed_image, processed_mask)
            os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
            cv2.imwrite(overlay_path, overlay)
        
        return True
        
    except Exception as e:
        print(f"  ✗ Hata: {e}")
        return False


def batch_process_dataset(images_folder="images_folder", 
                          masks_folder="masks_folder",
                          output_images="output_images_512",
                          output_masks="output_masks_512",
                          debug_overlays="debug_overlays",
                          target_size=512,
                          save_debug_overlays=True,
                          max_debug_samples=10):
    """
    Tüm dataset'i toplu işler.
    
    Args:
        images_folder: Kaynak görüntü klasörü
        masks_folder: Kaynak maske klasörü
        output_images: Hedef görüntü klasörü
        output_masks: Hedef maske klasörü
        debug_overlays: Debug overlay klasörü
        target_size: Hedef boyut
        save_debug_overlays: Debug overlay kaydet?
        max_debug_samples: Maksimum overlay sayısı
    """
    images_path = Path(images_folder)
    
    if not images_path.exists():
        print(f"✗ HATA: Görüntü klasörü bulunamadı: {images_folder}")
        return
    
    # Desteklenen formatlar
    image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_path.glob(f"*{ext}")))
        image_files.extend(list(images_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"✗ UYARI: '{images_folder}' klasöründe görüntü dosyası bulunamadı.")
        return
    
    print(f"Toplam {len(image_files)} adet görüntü bulundu.")
    print(f"Hedef boyut: {target_size}x{target_size}")
    print("=" * 70)
    
    success_count = 0
    missing_mask_count = 0
    error_count = 0
    debug_count = 0
    
    for idx, image_file in enumerate(image_files, 1):
        image_name = image_file.name
        print(f"\n[{idx}/{len(image_files)}] İşleniyor: {image_name}")
        
        # Eşleşen maskeyi bul
        mask_path = find_matching_mask(image_name, masks_folder)
        
        if mask_path is None:
            print(f"  ⚠ Maske bulunamadı, atlanıyor...")
            missing_mask_count += 1
            continue
        
        print(f"  ✓ Eşleşen maske: {os.path.basename(mask_path)}")
        
        # Output dosya yolları
        output_image_path = os.path.join(output_images, image_name)
        
        # Maske her zaman PNG olarak kaydet
        mask_output_name = Path(image_name).stem + ".png"
        output_mask_path = os.path.join(output_masks, mask_output_name)
        
        # Overlay yolu (ilk N örnek için)
        overlay_path = None
        if save_debug_overlays and debug_count < max_debug_samples:
            overlay_name = Path(image_name).stem + "_overlay.png"
            overlay_path = os.path.join(debug_overlays, overlay_name)
            debug_count += 1
        
        # İşle
        if process_pair(str(image_file), mask_path, output_image_path, 
                       output_mask_path, target_size, 
                       save_overlay=(overlay_path is not None), 
                       overlay_path=overlay_path):
            print(f"  ✓ Başarılı")
            success_count += 1
        else:
            error_count += 1
    
    print("\n" + "=" * 70)
    print("İşlem Tamamlandı!")
    print(f"  ✓ Başarılı: {success_count}")
    print(f"  ⚠ Maske Bulunamadı: {missing_mask_count}")
    print(f"  ✗ Hata: {error_count}")
    
    if save_debug_overlays and debug_count > 0:
        print(f"\nDebug overlay'ler kaydedildi: {debug_overlays}/ ({debug_count} adet)")
    
    print(f"\nİşlenmiş dosyalar:")
    print(f"  • Görüntüler: {output_images}/")
    print(f"  • Maskeler: {output_masks}/")


if __name__ == "__main__":
    print("=" * 70)
    print("Paired Image-Mask Dataset Preprocessing (512x512)")
    print("=" * 70)
    print()
    
    # Script'in bulunduğu dizindeki klasörleri kullan
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    images_folder = os.path.join(script_dir, "images_folder")
    masks_folder = os.path.join(script_dir, "masks_folder")
    output_images = os.path.join(script_dir, "output_images_512")
    output_masks = os.path.join(script_dir, "output_masks_512")
    debug_overlays = os.path.join(script_dir, "debug_overlays")
    
    # Batch işlemi başlat
    batch_process_dataset(
        images_folder=images_folder,
        masks_folder=masks_folder,
        output_images=output_images,
        output_masks=output_masks,
        debug_overlays=debug_overlays,
        target_size=512,
        save_debug_overlays=True,  # İlk 10 örnek için overlay kaydet
        max_debug_samples=10
    )
    
    print()
    input("Kapatmak için Enter'a basın...")
