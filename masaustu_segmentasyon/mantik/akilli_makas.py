import numpy as np
import cv2

def konturu_iyilestir(img_np, kaba_maske_np):
    """
    Enhanced GrabCut - Optimize edilmiş CLAHE ile kontrast artırma.
    
    Strateji:
    1. Sadece ROI'ye güçlü CLAHE uygula (hız + etkinlik).
    2. 3px erosion ile 'Kesin Ön Plan' belirlenir.
    3. 7px dilation ile 'Kesin Arka Plan' belirlenir.
    4. GrabCut (3 iterasyon - hız için azaltıldı).
    5. Sonuç pürüzsüzleştirilerek döndürülür.
    """
    try:
        # 1. ROI Belirleme (Hız için - sadece ilgili bölgeyi işle)
        y_indices, x_indices = np.where(kaba_maske_np > 0)
        if len(y_indices) == 0:
            return kaba_maske_np
        
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Padding ekle
        padding = 20
        h, w = img_np.shape[:2]
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        
        # ROI'yi çıkar
        roi_img = img_np[y_min:y_max, x_min:x_max]
        roi_mask = kaba_maske_np[y_min:y_max, x_min:x_max]
        
        # 2. CLAHE Uygula (Güçlü - zayıf kenarları belirginleştir)
        if len(roi_img.shape) == 2:
            roi_img_gray = roi_img.copy()
        else:
            roi_img_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # Daha güçlü CLAHE (clipLimit artırıldı: 3.0 -> 6.0)
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
        roi_img_enhanced = clahe.apply(roi_img_gray)
        
        # BGR'ye çevir (GrabCut için gerekli)
        roi_img_bgr = cv2.cvtColor(roi_img_enhanced, cv2.COLOR_GRAY2BGR)
        
        # 3. Maske Hazırlığı (Trimap)
        mask = np.zeros(roi_img.shape[:2], dtype=np.uint8)
        mask.fill(cv2.GC_PR_BGD)
        
        # 4. Sure Foreground - Erosion (3px, 1 iter)
        kernel_erode = np.ones((3, 3), np.uint8)
        sure_fg = cv2.erode(roi_mask, kernel_erode, iterations=1)
        mask[sure_fg > 0] = cv2.GC_FGD
        
        # 5. Probable Foreground
        mask[roi_mask > 0] = cv2.GC_PR_FGD
        
        # 6. Sure Background - Dilation (7px, 1 iter)
        kernel_dilate = np.ones((7, 7), np.uint8)
        sure_bg = cv2.dilate(roi_mask, kernel_dilate, iterations=1)
        mask[sure_bg == 0] = cv2.GC_BGD
        
        # 7. GrabCut (3 iterasyon - hız için azaltıldı, CLAHE sayesinde yeter)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi_img_bgr, mask, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
        
        # 8. Sonucu İşle
        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
        
        if np.sum(mask2) == 0:
            # ROI'de sonuç yoksa orijinali döndür
            contours, _ = cv2.findContours(kaba_maske_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            yeni_maske = np.zeros_like(kaba_maske_np)
            if contours:
                cv2.drawContours(yeni_maske, contours, -1, 255, 2)
            return yeni_maske if np.sum(yeni_maske) > 0 else kaba_maske_np
        
        # 9. ROI Konturu
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return kaba_maske_np
        
        max_contour = max(contours, key=cv2.contourArea)
        
        # Pürüzsüzleştirme
        epsilon = 0.001 * cv2.arcLength(max_contour, True)
        approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)
        
        # ROI konturunu tam görüntü koordinatlarına çevir
        approx_contour[:, 0, 0] += x_min
        approx_contour[:, 0, 1] += y_min
        
        # 10. Tam boyutta maske oluştur
        yeni_maske = np.zeros_like(kaba_maske_np)
        cv2.drawContours(yeni_maske, [approx_contour], -1, 255, thickness=2)
        
        return yeni_maske
        
    except Exception as e:
        print(f"GrabCut hatası: {e}")
        import traceback
        traceback.print_exc()
        return kaba_maske_np
