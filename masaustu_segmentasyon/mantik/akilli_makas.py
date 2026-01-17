import numpy as np
import cv2

def konturu_iyilestir(img_np, kaba_maske_np):
    """
    Enhanced GrabCut - CLAHE ile kontrast artırma.
    
    Strateji:
    1. Üst kısma CLAHE uygula (zayıf kenarları güçlendir).
    2. 3px erosion ile 'Kesin Ön Plan' belirlenir.
    3. 7px dilation ile 'Kesin Arka Plan' belirlenir.
    4. GrabCut (5 iterasyon) - Artık üstte de kenar bulabilir.
    5. Sonuç pürüzsüzleştirilerek döndürülür.
    """
    try:
        # 1. Görüntü Hazırlığı
        # Üst kısımda (düşük kontrast) CLAHE uygula - kenarları güçlendir
        if len(img_np.shape) == 2:
            img_gray = img_np.copy()
        else:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # Maskenin merkezini bul
        y_indices = np.where(kaba_maske_np > 0)[0]
        if len(y_indices) > 0:
            y_center = (np.min(y_indices) + np.max(y_indices)) // 2
        else:
            y_center = img_gray.shape[0] // 2
        # CLAHE uygula (tüm görüntüye - kontrast artırma)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        img_enhanced = clahe.apply(img_gray)
        
        # BGR'ye çevir (GrabCut için gerekli)
        img_bgr = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2BGR)
            
        # 2. Maske Hazırlığı (Trimap)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        mask.fill(cv2.GC_PR_BGD)  # Varsayılan: Muhtemel arka plan
        
        # 3. Sure Foreground - Erosion (3px, 1 iter)
        kernel_erode = np.ones((3, 3), np.uint8)
        sure_fg = cv2.erode(kaba_maske_np, kernel_erode, iterations=1)
        mask[sure_fg > 0] = cv2.GC_FGD  # Kesinlikle ön plan
        
        # 4. Probable Foreground - Kullanıcının maskesi
        mask[kaba_maske_np > 0] = cv2.GC_PR_FGD  # Muhtemelen ön plan
        
        # 5. Sure Background - Dilation (7px, 1 iter)
        kernel_dilate = np.ones((7, 7), np.uint8)
        sure_bg = cv2.dilate(kaba_maske_np, kernel_dilate, iterations=1)
        mask[sure_bg == 0] = cv2.GC_BGD  # Kesinlikle arka plan
        
        # 6. GrabCut (5 iterasyon) - Artık CLAHE sayesinde üstte de kenar bulabilir
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        # 7. Sonucu İşle
        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
        
        if np.sum(mask2) == 0:
            print("GrabCut sonucu boş. Orijinal maske döndürülüyor.")
            contours, _ = cv2.findContours(kaba_maske_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            yeni_maske = np.zeros_like(kaba_maske_np)
            if contours:
                cv2.drawContours(yeni_maske, contours, -1, 255, 1)
            return yeni_maske if np.sum(yeni_maske) > 0 else kaba_maske_np
        
        # 8. Konturu Çıkar ve Pürüzsüzleştir
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yeni_maske = np.zeros_like(kaba_maske_np)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Pürüzsüzleştirme
            epsilon = 0.001 * cv2.arcLength(max_contour, True)
            approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)
            
            cv2.drawContours(yeni_maske, [approx_contour], -1, 255, thickness=2)
        else:
            return kaba_maske_np
        
        return yeni_maske
        
    except Exception as e:
        print(f"GrabCut hatası: {e}")
        import traceback
        traceback.print_exc()
        return kaba_maske_np
