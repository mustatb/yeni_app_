import cv2
import numpy as np
from PIL import Image
import os
from mantik import akilli_makas

def otomatik_duzelt(goruntu_yolu: str, maske_pil: Image.Image) -> Image.Image:
    """
    Pipeline:
    1. Kaba maskeyi al.
    2. Intelligent Scissors (GrabCut) ile kenarlara oturt.
    3. Sonucu döndür.
    """
    try:
        # 1. Görüntü ve Maske Hazırlığı
        img = cv2.imread(goruntu_yolu) # BGR
        if img is None: raise ValueError("Görüntü okunamadı")
        
        maske_np = np.array(maske_pil)
        
        # Kaba maskenin içini doldur (Kullanıcı halka çizmiş olabilir)
        contours, _ = cv2.findContours(maske_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(maske_np, [max_contour], -1, 255, thickness=cv2.FILLED)
        
        # 2. ROI Belirleme
        y_indices, x_indices = np.where(maske_np > 0)
        if len(x_indices) == 0: return maske_pil
            
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        # Padding
        padding = 50
        h, w = maske_np.shape[:2]
        min_x = max(0, min_x - padding)
        max_x = min(w, max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(h, max_y + padding)
        
        # ROI Kırpma
        roi_img = img[min_y:max_y, min_x:max_x]
        roi_mask = maske_np[min_y:max_y, min_x:max_x]
        
        # 3. GrabCut Adımı
        # print("GrabCut ile kenarlara oturtuluyor (Snapping)...")
        refined_mask = akilli_makas.konturu_iyilestir(roi_img, roi_mask)
        
        # 4. Sonucu Yerleştir
        final_maske = np.zeros_like(maske_np)
        final_maske[min_y:max_y, min_x:max_x] = refined_mask
        
        return Image.fromarray(final_maske, mode='L')
        
    except Exception as e:
        print(f"Otomatik düzeltme hatası: {e}")
        import traceback
        traceback.print_exc()
        return maske_pil


