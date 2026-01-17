import numpy as np
import cv2

def konturu_iyilestir(img_np, kaba_maske_np):
    """
    Enhanced GrabCut - Morphological Gradient + CLAHE.
    
    Strateji:
    1. Sadece ROI'ye morphological gradient + güçlü CLAHE (kenar oluşturma).
    2. 3px erosion ile 'Kesin Ön Plan' belirlenir.
    3. 7px dilation ile 'Kesin Arka Plan' belirlenir.
    4. GrabCut (3 iterasyon - optimize).
    5. Bu sayede neredeyse hiç kenarı olmayan bölgelerde bile segmentasyon yapılabilir.
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
        
        # 2. Görüntüyü griye çevir
        if len(roi_img.shape) == 2:
            roi_img_gray = roi_img.copy()
        else:
            roi_img_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        # 3. Bölgesel Kenar Güçlendirme (Region-aware Edge Enhancement)
        # Posterior-alt: Edge ağırlığı ↑ (düşük kontrast)
        # Üst: Intensity ağırlığı ↑ (iyi kontrast)
        
        # Aşama 1: Bilateral filter (gürültü azalt, kenarları koru)
        roi_img_filtered = cv2.bilateralFilter(roi_img_gray, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Aşama 2: Sobel edge detection (X ve Y yönlerinde)
        sobel_x = cv2.Sobel(roi_img_filtered, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(roi_img_filtered, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
        
        # Aşama 3: Morphological gradient (yapısal sınırlar)
        kernel_morph = np.ones((3, 3), np.uint8)
        morph_gradient = cv2.morphologyEx(roi_img_filtered, cv2.MORPH_GRADIENT, kernel_morph)
        
        # Aşama 4: Sobel ve Morph gradient'i birleştir
        combined_edges = cv2.addWeighted(sobel_combined, 0.5, morph_gradient, 0.5, 0)
        
        # Aşama 5: Bölgesel Ağırlıklandırma
        # Posterior-alt bölge: Edge ağırlığı artır (%80)
        # Üst bölge: Normal ağırlık (%40)
        roi_h, roi_w = roi_img_gray.shape
        y_mid_roi = roi_h // 2
        x_mid_roi = roi_w // 2
        
        edge_weight_map = np.ones_like(roi_img_gray, dtype=np.float32) * 0.4
        edge_weight_map[y_mid_roi:, x_mid_roi:] = 0.8  # Posterior-alt: Yüksek edge weight
        
        # Weighted edge blending
        weighted_edges = (combined_edges.astype(float) * edge_weight_map).astype(np.uint8)
        roi_img_boosted = cv2.addWeighted(roi_img_filtered, 0.6, weighted_edges, 0.4, 0)
        
        # Aşama 6: Çok güçlü CLAHE
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4,4))
        roi_img_enhanced = clahe.apply(roi_img_boosted)
        
        # BGR'ye çevir (GrabCut için gerekli)
        roi_img_bgr = cv2.cvtColor(roi_img_enhanced, cv2.COLOR_GRAY2BGR)
        
        # 3. Maske Hazırlığı (Trimap)
        mask = np.zeros(roi_img.shape[:2], dtype=np.uint8)
        mask.fill(cv2.GC_PR_BGD)
        
        # 4. Erosion - Posterior HARİÇ
        # KRITIK: Posterior bölgede (üst-sağ) erosion YAPMA
        # GrabCut'un posterior'da gradient'leri takip etmesine izin ver
        
        h_roi, w_roi = roi_mask.shape
        y_mid = h_roi // 2
        x_mid = w_roi // 2
        
        # Posterior OLMAYAN bölgelere erosion
        non_posterior_mask = roi_mask.copy()
        non_posterior_mask[:y_mid, x_mid:] = 0  # Üst-sağı (posterior) sıfırla
        
        # Non-posterior bölgeler: 2px erosion
        sure_fg = np.zeros_like(roi_mask)
        if np.sum(non_posterior_mask) > 0:
            kernel_erode = np.ones((2, 2), np.uint8)
            eroded = cv2.erode(non_posterior_mask, kernel_erode, iterations=1)
            sure_fg[eroded > 0] = 255
        
        mask[sure_fg > 0] = cv2.GC_FGD
        
        # 5. Probable Foreground
        mask[roi_mask > 0] = cv2.GC_PR_FGD
        
        # 6. Sure Background - Anatomik Aware Dilation
        # Üst taraf (Talus bölgesi): Dar dilation (tight)
        # Alt/arka taraf: Geniş dilation (loose)
        
        # Maskenin y merkezini bul
        y_indices_mask = np.where(roi_mask > 0)[0]
        if len(y_indices_mask) > 0:
            y_min_mask = np.min(y_indices_mask)
            y_max_mask = np.max(y_indices_mask)
            y_center = (y_min_mask + y_max_mask) // 2
        else:
            y_center = roi_mask.shape[0] // 2
        
        # Loose dilation (alt taraf) - Eski koddan: 20x20, 3 iter
        kernel_loose = np.ones((20, 20), np.uint8)
        sure_bg_loose = cv2.dilate(roi_mask, kernel_loose, iterations=3)
        
        # Tight dilation (üst taraf - Talus bölgesi) - Eski koddan: 10x10, 2 iter
        kernel_tight = np.ones((10, 10), np.uint8)
        sure_bg_tight = cv2.dilate(roi_mask, kernel_tight, iterations=2)
        
        # Birleştir: Üst kısımda tight, alt kısımda loose
        sure_bg = sure_bg_loose.copy()
        sure_bg[:y_center, :] = sure_bg_tight[:y_center, :]
        
        mask[sure_bg == 0] = cv2.GC_BGD
        
        # 7. GrabCut (Eski koddan: 5 iterasyon daha iyi sonuç veriyor)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(roi_img_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
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
