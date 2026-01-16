import os
import cv2
import numpy as np
from typing import Tuple, Any, Optional

def load_dicom_array(path: str) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    """
    DICOM dosyasını okur ve numpy array olarak döndürür.
    
    Args:
        path: Dosya yolu
        
    Returns:
        (image_array, original_size) tuple. 
        Hata durumunda (None, (0, 0)) döner.
    """
    try:
        import pydicom
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype(np.float32)
        
        # Normalize (0-255)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        h, w = img.shape[:2]
        return img, (h, w)
    except Exception as e:
        print(f"DICOM okuma hatası: {e}")
        return None, (0, 0)

def load_image_array(path: str) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    """
    Standart görüntü dosyasını okur.
    
    Args:
        path: Dosya yolu
        
    Returns:
        (image_array, original_size) tuple.
    """
    try:
        # Türkçe karakter sorununa karşı
        # cv2.imread bazen Türkçe yollarda sorun çıkarabilir, numpy ile okuyup decode etmek daha güvenli
        with open(path, "rb") as f:
            bytes_data = bytearray(f.read())
            numpy_array = np.asarray(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(numpy_array, cv2.IMREAD_GRAYSCALE)
            
        if img is None:
            return None, (0, 0)
            
        h, w = img.shape[:2]
        return img, (h, w)
    except Exception as e:
        print(f"Görüntü okuma hatası: {e}")
        return None, (0, 0)
