import numpy as np
import cv2

def konturu_iyilestir(img_np, kaba_maske_np):
    """
    Gelişmiş GrabCut (Trimap) ile segmentasyon.
    
    Strateji:
    1. Kullanıcının çizimi (dolu maske) alınır.
    2. Erozyon ile 'Kesin Ön Plan' (Sure Foreground) belirlenir (Kemiğin ortası).
    3. Genişletme (Dilation) ile 'Kesin Arka Plan' (Sure Background) belirlenir (Kemiğin uzağı).
    4. Arada kalan bölge 'Muhtemel Alan' olarak GrabCut'a verilir.
    5. Sonuç pürüzsüzleştirilerek sadece kontur olarak döndürülür.
    """
    try:
        # 1. Görüntü Hazırlığı
        if len(img_np.shape) == 2:
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = img_np
            
        # 2. Maske Hazırlığı (Trimap)
        mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
        
        # Tüm alan varsayılan olarak Muhtemel Arka Plan (2)
        mask.fill(cv2.GC_PR_BGD)
        
        # Kaba maske (kullanıcının çizimi) -> Muhtemel Ön Plan (3)
        mask[kaba_maske_np > 0] = cv2.GC_PR_FGD
        
        # Kesin Ön Plan (Sure Foreground) - Erozyon ile
        # 2a. Maske/ROI Boyut Analizi
        h, w = kaba_maske_np.shape[:2]
        diag = np.sqrt(h**2 + w**2)
        
        # 2b. Maskenin merkezini bul (Erken hesaplama gerekli)
        y_ind, x_ind = np.where(kaba_maske_np > 0)
        if len(y_ind) > 0:
            y_min, y_max = np.min(y_ind), np.max(y_ind)
            y_center = (y_min + y_max) // 2
        else:
            y_center = kaba_maske_np.shape[0] // 2
        
        # Adaptif Kernel Boyutları
        # Erozyon (Sure Foreground) için çok küçük bir değer kullanıyoruz ki
        # kullanıcının çizdiği sınıra (kırmızı çizgi) olabildiğince güvenelim.
        erode_size = max(3, int(diag * 0.005)) # Örn: 1000px için 5px
        kernel_erode = np.ones((erode_size, erode_size), np.uint8)
        
        # Dilation (Sure Background) - Loose
        dilate_loose_size = max(5, int(diag * 0.03)) # Örn: 1000px için 30px
        kernel_loose = np.ones((dilate_loose_size, dilate_loose_size), np.uint8)
        
        # Dilation (Sure Background) - Tight (Üst taraf için)
        dilate_tight_size = max(3, int(diag * 0.01)) # Örn: 1000px için 10px
        kernel_tight = np.ones((dilate_tight_size, dilate_tight_size), np.uint8)

        # 3. Kesin Ön Plan (Sure Foreground) - Erozyon ile
        # Maskenin iç kısmını kesin kemik olarak işaretle
        
        # Strateji: Alt tarafta (Plantar yüzey) kullanıcının çizdiği çizgiye tam sadık kalmak istiyoruz.
        # Bu yüzden alt tarafta erozyon yapmayacağız veya çok az yapacağız.
        # Üst tarafta (Talus) ise erozyon yaparak algoritmanın oturtmasına izin vereceğiz.
        
        full_eroded = cv2.erode(kaba_maske_np, kernel_erode, iterations=1)
        
        sure_fg = full_eroded.copy()
        
        # Alt yarıda (y > y_center) erozyonu iptal et -> Kullanıcının çizdiği tam maskeyi kullan
        # Bu sayede alt sınır "Sure Foreground" olarak işaretlenir ve GrabCut burayı kesemez.
        sure_fg[y_center:, :] = kaba_maske_np[y_center:, :]

        mask[sure_fg > 0] = cv2.GC_FGD # 1
        
        # 4. Kesin Arka Plan (Sure Background) - Dilation ile
        # Maskenin çok dışını kesin arka plan olarak işaretle
        
        # Strateji: Kalkaneusun üst kısmı (Talus tarafı) için daha kısıtlı bir alan (tight),
        # alt ve arka kısımlar için daha geniş bir alan (loose) kullanalım.
        # Bu sayede üst tarafta talusa "sıçrama" yapması engellenir.
        
        # 4a. Maskenin merkezi zaten hesaplandı (y_center)

        # 4b. Geniş (Loose) Dilation - Alt ve Arka taraf için
        sure_bg_loose = cv2.dilate(kaba_maske_np, kernel_loose, iterations=1)
        
        # 4c. Dar (Tight) Dilation - Üst taraf için
        sure_bg_tight = cv2.dilate(kaba_maske_np, kernel_tight, iterations=1)
        
        # 4. Maskeleri Birleştir
        sure_bg_area = sure_bg_loose.copy()
        
        # Üst yarıda (y < y_center) tight maskeyi kullan
        # Ancak sadece "dışarıdaki" alanları kısıtlıyoruz.
        # sure_bg_area'nın 0 olduğu yerler kesin arka plan.
        # Tight maskede 0 olan yerler (daha geniş siyah alan) loose maskede 1 olabilir (beyaz alan).
        # Biz üst tarafta siyah alanın (arka planın) daha "içeri" girmesini istiyoruz.
        # Yani sure_bg_tight'ı kullanacağız.
        
        sure_bg_area[:y_center, :] = sure_bg_tight[:y_center, :]
        
        # sure_bg_area'nın dışı (0 olan yerler) kesin arka plandır
        mask[sure_bg_area == 0] = cv2.GC_BGD # 0
        
        # 3. GrabCut Çalıştır
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(img_bgr, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
        
        # 4. Sonucu İşle
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        if np.sum(mask2) == 0:
            print("GrabCut sonucu boş. Orijinal maske döndürülüyor.")
            contours, _ = cv2.findContours(kaba_maske_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            yeni_maske = np.zeros_like(kaba_maske_np)
            cv2.drawContours(yeni_maske, contours, -1, 255, 1)
            return yeni_maske

        # 5. Konturu Çıkar ve Pürüzsüzleştir
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        yeni_maske = np.zeros_like(kaba_maske_np)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            
            # Pürüzsüzleştirme (Smoothing)
            # epsilon ne kadar büyükse o kadar düzleşir.
            epsilon = 0.001 * cv2.arcLength(max_contour, True)
            approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)
            
            cv2.drawContours(yeni_maske, [approx_contour], -1, 255, thickness=2)
        
        return yeni_maske
        
    except Exception as e:
        print(f"GrabCut hatası: {e}")
        import traceback
        traceback.print_exc()
        return kaba_maske_np
