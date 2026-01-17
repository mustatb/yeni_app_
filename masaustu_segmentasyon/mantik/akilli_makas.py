import numpy as np
import cv2

def konturu_iyilestir(img_np, kaba_maske_np):
    """
    Optimized GrabCut - Kullanıcı çizimlerine tam güven.
    
    Strateji:
    1. Kullanıcının tüm çizimi (AI + manuel) doğru kabul edilir.
    2. Erosion YOK - İnce manuel eklemeler korunur.
    3. Dilation ile 'Kesin Arka Plan' belirlenir.
    4. GrabCut sadece dar bir kenarda çalışır.
    5. Sonuç pürüzsüzleştirilerek döndürülür.
    """
    try:
        # 1. Görüntü Hazırlığı
        if len(img_np.shape) == 2:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_np
            
        # 2. Maske Hazırlığı (Trimap)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        mask.fill(cv2.GC_PR_BGD)  # Varsayılan: Muhtemel arka plan
        
        # 3. Sure Foreground - EROSION YOK
        # Kullanıcının çizdiği her şey (AI + manuel düzeltmeler) kesin doğru
        # Özellikle ince manuel çizimler erosion'da kaybolmasın diye
        mask[kaba_maske_np > 0] = cv2.GC_FGD  # Tüm maske = Kesinlikle ön plan
        
        # 4. Sure Background - Dilation (7px, 1 iter)
        # Sadece maskenin dışını işaretle
        kernel_dilate = np.ones((7, 7), np.uint8)
        sure_bg = cv2.dilate(kaba_maske_np, kernel_dilate, iterations=1)
        mask[sure_bg == 0] = cv2.GC_BGD  # Kesinlikle arka plan
        
        # 6. GrabCut - Daha fazla iterasyon (5) - Kenarları daha iyi oturtur
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
