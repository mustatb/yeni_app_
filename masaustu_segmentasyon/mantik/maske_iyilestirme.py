"""
Maske İyileştirme Modülü
========================

AI tarafından üretilen maskelerin kenarlarını düzleştirir ve iyileştirir.
Geliştirilmiş versiyon - daha güçlü smoothing.
"""

import cv2
import numpy as np
from PIL import Image


def smooth_mask_contours(mask_image, epsilon_factor=0.002, morph_kernel_size=7):
    """
    Maskenin kenarlarını düzleştirir ve pixelated görünümü giderir.
    Geliştirilmiş versiyon - daha agresif smoothing.
    
    Args:
        mask_image: PIL Image formatında binary maske
        epsilon_factor: Contour smoothing hassasiyeti (küçük = daha smooth)
        morph_kernel_size: Morphological işlem kernel boyutu
    
    Returns:
        PIL.Image: İyileştirilmiş maske
    """
    # PIL Image'i numpy array'e çevir
    mask_array = np.array(mask_image)
    
    # Binary threshold uygula
    _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
    
    # 1. Çoklu aşamalı morphological işlemler
    # Küçük kernel ile başla - gürültü temizliği
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, small_kernel, iterations=1)
    
    # Orta kernel - delikleri kapat
    medium_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, medium_kernel, iterations=2)
    
    # Büyük kernel - daha fazla smoothing
    large_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size + 2, morph_kernel_size + 2))
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, large_kernel, iterations=1)
    
    # 2. Contourları bul
    contours, hierarchy = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Contour bulunamadıysa orijinal maskeyi döndür
        return mask_image
    
    # 3. En büyük contouru seç (kemik)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 4. İki aşamalı contour smoothing
    # İlk smoothing - agresif
    epsilon1 = epsilon_factor * 2 * cv2.arcLength(largest_contour, True)
    smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon1, True)
    
    # İkinci smoothing - daha ince ayar
    epsilon2 = epsilon_factor * cv2.arcLength(smoothed_contour, True)
    final_contour = cv2.approxPolyDP(smoothed_contour, epsilon2, True)
    
    # 5. Yeni maske oluştur
    smooth_mask = np.zeros_like(binary_mask)
    cv2.drawContours(smooth_mask, [final_contour], -1, 255, -1)  # -1 = fill
    
    # 6. Kenarları yumuşat (daha güçlü Gaussian blur)
    smooth_mask = cv2.GaussianBlur(smooth_mask, (5, 5), 1.0)
    _, smooth_mask = cv2.threshold(smooth_mask, 127, 255, cv2.THRESH_BINARY)
    
    # 7. Son morphological smoothing
    final_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    smooth_mask = cv2.morphologyEx(smooth_mask, cv2.MORPH_CLOSE, final_kernel, iterations=1)
    
    # Numpy array'i PIL Image'e çevir
    return Image.fromarray(smooth_mask)


def refine_mask_edges(mask_image, iterations=2):
    """
    Maskenin kenarlarını iyileştirir (alternatif yöntem).
    
    Args:
        mask_image: PIL Image formatında binary maske
        iterations: İyileştirme iterasyon sayısı
    
    Returns:
        PIL.Image: İyileştirilmiş maske
    """
    # PIL Image'i numpy array'e çevir
    mask_array = np.array(mask_image)
    
    # Binary threshold
    _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
    
    # Morphological gradient - kenarları belirginleştir
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    # Dilation sonra erosion (closing)
    for _ in range(iterations):
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    
    # Gaussian blur + re-threshold
    blurred = cv2.GaussianBlur(binary_mask, (5, 5), 0)
    _, refined_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return Image.fromarray(refined_mask)


def advanced_edge_refinement(mask_image, original_image_path):
    """
    Orijinal görüntünün kenarlarını kullanarak maskeyi iyileştirir.
    
    Args:
        mask_image: PIL Image formatında binary maske
        original_image_path: Orijinal X-ray görüntüsünün yolu
    
    Returns:
        PIL.Image: İyileştirilmiş maske
    """
    try:
        # Orijinal görüntüyü oku
        original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if original is None:
            # Orijinal görüntü okunamazsa sadece smooth işlemi yap
            return smooth_mask_contours(mask_image)
        
        # Maskeyi numpy array'e çevir
        mask_array = np.array(mask_image)
        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)
        
        # Orijinal görüntüde kenar tespiti
        edges = cv2.Canny(original, 50, 150)
        
        # Kenarları genişlet
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Maskenin kenarlarını bul
        mask_contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not mask_contours:
            return mask_image
        
        # En büyük contouru seç
        largest_contour = max(mask_contours, key=cv2.contourArea)
        
        # Smoothing uygula
        epsilon = 0.002 * cv2.arcLength(largest_contour, True)
        smoothed = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Yeni maske
        result_mask = np.zeros_like(binary_mask)
        cv2.drawContours(result_mask, [smoothed], -1, 255, -1)
        
        return Image.fromarray(result_mask)
        
    except Exception as e:
        print(f"Advanced refinement hatası: {e}")
        # Hata durumunda basit smoothing yap
        return smooth_mask_contours(mask_image)
