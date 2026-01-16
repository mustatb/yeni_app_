import numpy as np
from PIL import Image
import os

def bos_maske_olustur(genislik: int, yukseklik: int) -> Image.Image:
    """Belirtilen boyutlarda siyah (0) bir maske oluşturur."""
    return Image.new("L", (genislik, yukseklik), 0)

def maske_yukle(maske_yolu: str) -> Image.Image:
    """Maske dosyasını yükler. Eğer dosya yoksa None döner."""
    if not os.path.exists(maske_yolu):
        return None
    try:
        img = Image.open(maske_yolu).convert("L")
        return img
    except Exception as e:
        print(f"Maske yüklenirken hata: {e}")
        return None

def maske_kaydet(maske: Image.Image, kayit_yolu: str) -> bool:
    """Maskeyi belirtilen yola kaydeder."""
    try:
        # Klasörün var olduğundan emin ol (dosya_yonetimi.py hallediyor ama garanti olsun)
        klasor = os.path.dirname(kayit_yolu)
        if not os.path.exists(klasor):
            os.makedirs(klasor, exist_ok=True)
            
        maske.save(kayit_yolu)
        return True
    except Exception as e:
        print(f"Maske kaydedilirken hata: {e}")
        return False

def numpy_dizisine_cevir(image: Image.Image) -> np.ndarray:
    """PIL görüntüsünü numpy dizisine çevirir."""
    return np.array(image)

def pil_goruntusune_cevir(array: np.ndarray) -> Image.Image:
    """Numpy dizisini PIL görüntüsüne çevirir."""
    return Image.fromarray(array.astype('uint8'), 'L')

def overlay_kaydet(orijinal_resim_yolu: str, maske: Image.Image, kayit_yolu: str, opaklik: float = 0.4, renk: tuple = (0, 0, 255)) -> bool:
    """
    Orijinal görüntünün üzerine maskeyi renkli ve yarı saydam olarak ekleyip kaydeder.
    renk: (B, G, R) formatında tuple. Varsayılan Kırmızı (0, 0, 255).
    """
    try:
        import cv2
        
        # Orijinal görüntüyü oku
        img = cv2.imread(orijinal_resim_yolu)
        if img is None:
            print(f"Orijinal görüntü okunamadı: {orijinal_resim_yolu}")
            return False
            
        # Maskeyi numpy array'e çevir
        maske_np = np.array(maske)
        
        # Maskeyi orijinal görüntü boyutuna getir (gerekirse)
        if maske_np.shape[:2] != img.shape[:2]:
            maske_np = cv2.resize(maske_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
        # Maskenin olduğu pikselleri bul
        mask_indices = maske_np > 0
        
        # Eğer maske boşsa, sadece orijinali kaydet (veya overlay yokmuş gibi)
        if not np.any(mask_indices):
            cv2.imwrite(kayit_yolu, img)
            return True
            
        # Renkli katman oluştur
        renkli_katman = np.zeros_like(img)
        renkli_katman[:] = renk
        
        # Tüm görüntü için blend işlemi yap
        # alpha blend: img * (1 - alpha) + color * alpha
        blended = cv2.addWeighted(img, 1 - opaklik, renkli_katman, opaklik, 0)
        
        # Sadece maskeli alanları blended haliyle değiştir
        final_output = img.copy()
        final_output[mask_indices] = blended[mask_indices]
        
        # Kaydet
        cv2.imwrite(kayit_yolu, final_output)
        return True
    except Exception as e:
        print(f"Overlay kaydedilirken hata: {e}")
        return False
