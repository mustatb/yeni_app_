"""
AI Model Modülü - Kalkaneus Kemiği Segmentasyonu
PyTorch U-Net (ResNet34 encoder) kullanarak otomatik segmentasyon yapar.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import os
from typing import Any, Dict, Optional
from mantik.image_utils import load_dicom_array, load_image_array


class CalcaneusSegmentationModel:
    """
    Kalkaneus kemiği segmentasyonu için AI model wrapper sınıfı.
    """
    
    def __init__(self, model_path: str):
        """
        Model yükleme ve başlatma.
        
        Args:
            model_path: .pth model dosyasının tam yolu
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"AI Model cihazı: {self.device}")
        
        # Model mimarisini oluştur (U-Net with ResNet34 encoder)
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,  # Önceden eğitilmiş ağırlık kullanmıyoruz
            in_channels=1,         # Grayscale görüntü
            classes=1,             # Binary segmentasyon
            activation='sigmoid'   # 0-1 arası çıktı
        )
        
        # Model ağırlıklarını yükle
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()  # Evaluation mode
            print(f"✓ AI Model başarıyla yüklendi: {os.path.basename(model_path)}")
        except Exception as e:
            raise RuntimeError(f"Model yüklenemedi: {e}")
    
    def preprocess_image(self, image_path: str) -> tuple[torch.Tensor, tuple]:
        """
        Görüntüyü model için hazırlar.
        
        Args:
            image_path: Görüntü dosyası yolu (DICOM, PNG, JPG, vb.)
            
        Returns:
            (preprocessed_tensor, original_size) tuple
        """
        # Görüntüyü oku
        if image_path.lower().endswith(('.dcm', '.dicom')):
            # DICOM dosyası
            try:
                import pydicom
                dcm = pydicom.dcmread(image_path)
                img = dcm.pixel_array.astype(np.float32)
                # Normalize (DICOM'da değer aralığı farklı olabilir)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            except Exception as e:
                raise ValueError(f"DICOM dosyası okunamadı: {e}")
        else:
            # Standart görüntü formatları
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Görüntü okunamadı: {image_path}")
        
        original_size = img.shape[:2]  # (height, width)
        
        # 512x512'ye resize et
        img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Normalize: [-1, 1] aralığına
        img_normalized = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # Tensor'a çevir: (1, 1, 512, 512) - (batch, channels, height, width)
        tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float()
        
        return tensor, original_size
    
    def run_inference(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Model inference yapar.
        
        Args:
            image_tensor: Preprocessed görüntü tensoru
            
        Returns:
            512x512 binary mask (0-255 numpy array)
        """
        with torch.no_grad():
            # GPU/CPU'ya gönder
            image_tensor = image_tensor.to(self.device)
            
            # Inference
            output = self.model(image_tensor)
            
            # Sigmoid zaten modelde var, çıktı [0, 1] aralığında
            # Threshold uygula (0.5)
            mask = (output.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            
            # 0-255 aralığına çevir
            mask = mask * 255
        
        return mask
    
    def postprocess_mask(self, mask: np.ndarray, original_size: tuple) -> np.ndarray:
        """
        Maskeyi post-process eder ve orijinal boyuta döndürür.
        
        Args:
            mask: 512x512 binary mask
            original_size: (height, width) orijinal görüntü boyutu
            
        Returns:
            Orijinal boyutta binary mask (0-255)
        """
        # Morphological opening (gürültü temizleme)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # En büyük contour'u bul
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Contour bulunamadıysa boş maske döndür
            return np.zeros(original_size, dtype=np.uint8)
        
        # En büyük contour'u seç
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Yeni temiz maske oluştur
        mask_final = np.zeros_like(mask_cleaned)
        
        # Contour içini doldur (Convex Hull kaldırıldı)
        cv2.drawContours(mask_final, [largest_contour], -1, 255, thickness=cv2.FILLED)
        
        # Orijinal boyuta geri ölçeklendir
        mask_resized = cv2.resize(mask_final, (original_size[1], original_size[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        return mask_resized
    
    def validate_mask(self, mask: np.ndarray, image_shape: tuple) -> tuple[bool, str]:
        """
        Maskenin validasyon kriterlerini kontrol eder.
        
        Args:
            mask: Binary mask
            image_shape: (height, width)
            
        Returns:
            (is_valid, message) tuple
        """
        h, w = image_shape
        total_pixels = h * w
        
        # Maskedeki beyaz pikselleri say
        mask_pixels = np.sum(mask > 0)
        
        if mask_pixels == 0:
            return False, "Maske boş (segmentasyon bulunamadı)"
        
        # 1. Minimum alan kontrolü: Görüntünün %0.3'ü
        min_area = total_pixels * 0.003
        if mask_pixels < min_area:
            return False, f"Maske çok küçük (Min: {min_area:.0f} px, Bulunan: {mask_pixels} px)"
        
        # Contour bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, "Contour bulunamadı"
        
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 2. Aspect ratio kontrolü: 1.2 - 5.0 arası
        x, y, rect_w, rect_h = cv2.boundingRect(largest_contour)
        aspect_ratio = max(rect_w, rect_h) / max(min(rect_w, rect_h), 1)
        
        if not (1.2 <= aspect_ratio <= 5.0):
            return False, f"Aspect ratio uygunsuz ({aspect_ratio:.2f}, beklenen: 1.2-5.0)"
        
        # 3. Solidity (kompaktlık) kontrolü: ≥0.60
        area = cv2.contourArea(largest_contour)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        if solidity < 0.60:
            return False, f"Solidity düşük ({solidity:.2f}, minimum: 0.60)"
        
        # 4. Konum kontrolü: Alt yarıda olmalı
        centroid_y = y + rect_h // 2
        if centroid_y < h * 0.3:  # En azından görüntünün üst %30'ında olmamalı
            return False, f"Kalkaneus kemiği görüntünün üst kısmında (beklenmedik konum)"
        
        # Tüm kriterler başarılı
        return True, "Validasyon başarılı"
    
    def segment(self, image_path: str) -> tuple[Image.Image, bool, str]:
        """
        Tam segmentasyon pipeline'ı.
        
        Args:
            image_path: Görüntü dosyası yolu
            
        Returns:
            (mask_image, is_valid, message) tuple
            - mask_image: PIL Image (L mode) binary mask
            - is_valid: Validasyon başarılı mı?
            - message: Bilgi/hata mesajı
        """
        try:
            # 1. Ön işleme
            image_tensor, original_size = self.preprocess_image(image_path)
            
            # 2. Inference
            mask_512 = self.run_inference(image_tensor)
            
            # 3. Post-processing
            mask_final = self.postprocess_mask(mask_512, original_size)
            
            # 4. Validasyon
            is_valid, message = self.validate_mask(mask_final, original_size)
            
            # PIL Image'e çevir
            mask_image = Image.fromarray(mask_final, mode='L')
            
            return mask_image, is_valid, message
            
        except Exception as e:
            # Hata durumunda boş maske döndür
            return None, False, f"Segmentasyon hatası: {e}"

    def analyze(self, image_data: Any, generate_debug: bool = False) -> Dict[str, Any]:
        """
        Main pipeline with AI prediction (Mask Only).
        """
        # 0. Load Image
        image = None
        if isinstance(image_data, str):
            ext = os.path.splitext(image_data)[1].lower()
            if ext in ['.dcm', '.dicom']:
                image, _ = load_dicom_array(image_data)
            else:
                image, _ = load_image_array(image_data)
                
            if image is None:
                 return {"error": f"Görüntü okunamadı: {image_data}"}
                 
        elif isinstance(image_data, np.ndarray):
            if len(image_data.shape) == 3:
                image = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            else:
                image = image_data
        else:
             return {"error": "Geçersiz giriş formatı"}
             
        if image is None:
            return {"error": "Görüntü okunamadı"}

        original_h, original_w = image.shape[:2]

        if self.model is None:
             return {"error": "Model yüklü değil"}

        # === AI PREDICTION ===
        # Mevcut preprocess fonksiyonumuz dosya yolu bekliyor, 
        # bu yüzden tensor dönüşümünü manuel yapıyoruz (image_utils'den gelen numpy array ile).
        
        # 512x512'ye resize et
        img_resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
        
        # Normalize: [-1, 1] aralığına
        img_normalized = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        
        # Tensor'a çevir: (1, 1, 512, 512)
        input_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            
        mask_tensor = torch.sigmoid(output) > 0.5
        mask_resized = mask_tensor.squeeze().cpu().numpy().astype(np.uint8) * 255
            
        # Resize mask back to original size
        mask_original = cv2.resize(mask_resized, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        # Post-process (optional cleanup)
        mask_original = self.postprocess_mask(mask_original, (original_h, original_w))
        
        # 5. Prepare Result (Mask Only)
        return {
            "status": "success",
            "visualized_image": None, # Görselleştirme istenmedi
            "angle": None,            # İstenmedi
            "diagnosis": None,        # İstenmedi
            "mask": mask_original
        }


# Global model instance (singleton pattern)
_model_instance = None


def load_model(model_path: str) -> CalcaneusSegmentationModel:
    """
    AI modelini yükler (singleton).
    
    Args:
        model_path: .pth model dosyasının tam yolu
        
    Returns:
        CalcaneusSegmentationModel instance
    """
    global _model_instance
    
    if _model_instance is None:
        _model_instance = CalcaneusSegmentationModel(model_path)
    
    return _model_instance


def get_model() -> CalcaneusSegmentationModel:
    """
    Yüklenmiş model instance'ını döndürür.
    
    Returns:
        CalcaneusSegmentationModel instance veya None
    """
    return _model_instance
