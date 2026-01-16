from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt6.QtCore import Qt, pyqtSignal, QPointF, QRectF
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QBrush, QCursor
import numpy as np
import cv2
from PIL import Image

class SegmentasyonAlani(QGraphicsView):
    """
    Görüntü ve maske overlay'ini gösteren, zoom/pan ve boyama işlemlerini yöneten widget.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Görüntü ve Maske itemları
        self.goruntu_item = QGraphicsPixmapItem()
        self.maske_item = QGraphicsPixmapItem()
        
        self.scene.addItem(self.goruntu_item)
        self.scene.addItem(self.maske_item)
        
        # Ayarlar
        self.firca_boyutu = 10
        self.maske_opaklik = 0.6
        self.arac_modu = "firca" # "firca", "silgi", "pan"
        self.boyama_aktif = False
        
        # Undo (Geri Al) Yığını
        self.geri_al_yigin = [] # QImage kopyalarını tutacak
        self.ileri_al_yigin = [] # Redo (İleri Al) Yığını
        self.max_undo = 10
        
        # Maske verisi (QImage olarak tutacağız, kaydederken PIL'e çevireceğiz)
        self.maske_image = None
        
        # Görünüm ayarları
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False) # Piksel piksel görmek için
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        
        # Maske rengi (Varsayılan Mavi: #3C50DC)
        self.maske_rengi = QColor(60, 80, 220, 255)

    def goruntu_yukle(self, dosya_yolu: str) -> bool:
        """Görüntüyü yükler ve sahneye ekler. Başarılıysa True döner."""
        pixmap = QPixmap(dosya_yolu)
        if pixmap.isNull():
            return False
            
        self.goruntu_item.setPixmap(pixmap)
        
        # Sahne boyutunu güncelle
        self.scene.setSceneRect(QRectF(pixmap.rect()))
        
        # Maske boyutunu da görüntüye eşitle (henüz maske yoksa boş oluştur)
        if self.maske_image is None or self.maske_image.size() != pixmap.size():
            self.maske_temizle()
        
        self.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        return True

    def maske_yukle(self, pil_image: Image.Image):
        """PIL Image nesnesinden maskeyi yükler."""
        if pil_image is None:
            self.maske_temizle()
            return
            
        # PIL (L mode) -> QImage (Format_Grayscale8 veya Alpha8)
        # Ancak boyama kolaylığı için Format_ARGB32 kullanıp sadece alpha kanalını veya rengi manipüle edebiliriz.
        # Daha basit yöntem: Maskeyi siyah-beyaz (L) olarak alıp, QImage'e çevirip,
        # boyanan yerleri kırmızı yapıp diğer yerleri şeffaf yapmak.
        
        # PIL Image'i RGBA'ya çevirip işleyelim
        img_rgba = pil_image.convert("RGBA")
        data = np.array(img_rgba)
        
        # Beyaz olan yerleri (255) Kırmızı yap, Siyah olan yerleri Şeffaf yap
        # data[:, :, 0] = R, 1=G, 2=B, 3=A
        # Maske (L) verisi aslında R=G=B kanallarında aynıdır.
        
        # Maske piksellerini bul (R kanalı > 128 diyelim)
        maske_indeksleri = data[:, :, 0] > 128
        
        # Maske rengi ile boya
        data[maske_indeksleri, 0] = self.maske_rengi.red()
        data[maske_indeksleri, 1] = self.maske_rengi.green()
        data[maske_indeksleri, 2] = self.maske_rengi.blue()
        data[maske_indeksleri, 3] = 255 # Tam opak (item opacity ile kontrol edilecek)
        
        # Maske olmayan yerleri şeffaf yap
        data[~maske_indeksleri, 3] = 0
        
        height, width, channel = data.shape
        bytes_per_line = 4 * width
        
        # QImage oluştur
        qimg = QImage(data.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        self.maske_image = qimg.copy() # Copy önemli, buffer referansı kaybolmasın
        self.maske_guncelle()

    def maske_temizle(self):
        """Maskeyi sıfırlar (tamamen şeffaf)."""
        rect = self.goruntu_item.pixmap().rect()
        self.maske_image = QImage(rect.size(), QImage.Format.Format_RGBA8888)
        self.maske_image.fill(QColor(0, 0, 0, 0)) # Şeffaf
        self.maske_guncelle()

    def maske_guncelle(self):
        """QImage maskeyi sahnedeki item'a aktarır."""
        if self.maske_image:
            self.maske_item.setPixmap(QPixmap.fromImage(self.maske_image))
            self.maske_item.setOpacity(self.maske_opaklik)

    def maske_al(self) -> Image.Image:
        """Mevcut maskeyi PIL Image (L mode) olarak döndürür."""
        if not self.maske_image:
            return None
            
        # QImage -> PIL
        # QImage buffer'ını al
        ptr = self.maske_image.bits()
        ptr.setsize(self.maske_image.sizeInBytes())
        arr = np.array(ptr).reshape(self.maske_image.height(), self.maske_image.width(), 4)
        
        # Alpha kanalı > 0 olan yerler maske (beyaz), diğerleri siyah
        # Veya R kanalı 255 olanlar.
        
        maske_arr = np.zeros((self.maske_image.height(), self.maske_image.width()), dtype=np.uint8)
        # Kırmızı olan yerler (veya alpha > 0)
        maske_arr[arr[:, :, 3] > 0] = 255
        
        return Image.fromarray(maske_arr, mode="L")

    def set_arac_modu(self, mod: str):
        self.arac_modu = mod
        if mod == "pan":
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)

    def durumu_kaydet(self):
        """Mevcut maskeyi undo yığınına ekler ve redo yığınını temizler."""
        if self.maske_image:
            self.geri_al_yigin.append(self.maske_image.copy())
            self.ileri_al_yigin.clear() # Yeni işlem yapıldığında redo yığını temizlenmeli
            if len(self.geri_al_yigin) > self.max_undo:
                self.geri_al_yigin.pop(0)

    def geri_al(self):
        """Son işlemi geri alır."""
        if self.geri_al_yigin:
            # Şu anki hali redo yığınına at
            if self.maske_image:
                self.ileri_al_yigin.append(self.maske_image.copy())
            
            eski_maske = self.geri_al_yigin.pop()
            self.maske_image = eski_maske
            self.maske_guncelle()

    def ileri_al(self):
        """Geri alınan işlemi ileri alır."""
        if self.ileri_al_yigin:
            # Şu anki hali undo yığınına at
            if self.maske_image:
                self.geri_al_yigin.append(self.maske_image.copy())
            
            yeni_maske = self.ileri_al_yigin.pop()
            self.maske_image = yeni_maske
            self.maske_guncelle()

    def set_firca_boyutu(self, boyut: int):
        self.firca_boyutu = boyut

    def set_maske_opaklik(self, deger: float):
        self.maske_opaklik = deger
        self.maske_item.setOpacity(deger)

    def set_maske_rengi(self, renk: QColor):
        """Maske rengini değiştirir ve mevcut maskeyi günceller."""
        self.maske_rengi = renk
        
        if self.maske_image:
            # Mevcut maskeyi yeni renge boya
            if self.maske_image.format() != QImage.Format.Format_RGBA8888:
                self.maske_image = self.maske_image.convertToFormat(QImage.Format.Format_RGBA8888)
                
            ptr = self.maske_image.bits()
            ptr.setsize(self.maske_image.sizeInBytes())
            arr = np.array(ptr).reshape(self.maske_image.height(), self.maske_image.width(), 4)
            
            # Dolu olan pikselleri bul (Alpha > 0)
            dolu_pikseller = arr[:, :, 3] > 0
            
            # Rengi güncelle
            arr[dolu_pikseller, 0] = self.maske_rengi.red()
            arr[dolu_pikseller, 1] = self.maske_rengi.green()
            arr[dolu_pikseller, 2] = self.maske_rengi.blue()
            
            self.maske_guncelle()

    def mousePressEvent(self, event):
        if self.arac_modu == "pan" or (event.button() == Qt.MouseButton.RightButton):
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.pan_aktif = True
            self.last_pan_pos = event.pos()
            if event.button() == Qt.MouseButton.RightButton:
                return
            super().mousePressEvent(event)
            return

        if self.arac_modu == "kova":
            self.doldur(event.pos())
            return

        if self.arac_modu in ["firca", "silgi"]:
            # Çizim başlamadan önce durumu kaydet
            self.durumu_kaydet()
            scene_pos = self.mapToScene(event.pos())
            self.last_draw_pos = scene_pos
            self.boya(scene_pos)

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if hasattr(self, 'pan_aktif') and self.pan_aktif:
            super().mouseMoveEvent(event)
            return

        if self.arac_modu in ["firca", "silgi"] and event.buttons() & Qt.MouseButton.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            if hasattr(self, 'last_draw_pos') and self.last_draw_pos:
                self.cizgi_cek(self.last_draw_pos, scene_pos)
            else:
                self.boya(scene_pos)
            self.last_draw_pos = scene_pos

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self, 'pan_aktif') and self.pan_aktif:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.pan_aktif = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        if self.arac_modu in ["firca", "silgi"]:
            # Durumu zaten Press'te kaydettik, burada kaydetmeye gerek yok
            self.last_draw_pos = None

        super().mouseReleaseEvent(event)

    def cizgi_cek(self, p1, p2):
        if not self.maske_image:
            return

        painter = QPainter(self.maske_image)
        renk = self.maske_rengi if self.arac_modu == "firca" else QColor(0, 0, 0, 0)
        
        if self.arac_modu == "silgi":
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        else:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        pen = QPen(renk)
        pen.setWidth(self.firca_boyutu)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        painter.drawLine(p1, p2)
        painter.end()
        
        self.maske_guncelle()

    def boya(self, pos):
        self.cizgi_cek(pos, pos)

    def maske_guncelle(self):
        if self.maske_item and self.maske_image:
            self.maske_item.setPixmap(QPixmap.fromImage(self.maske_image))
            self.maske_item.setOpacity(self.maske_opaklik)

    def doldur(self, pos):
        if not self.maske_image:
            return
            
        # İşlemden önce durumu kaydet
        self.durumu_kaydet()

        scene_pos = self.mapToScene(pos)
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        
        w = self.maske_image.width()
        h = self.maske_image.height()
        
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        if self.maske_image.format() != QImage.Format.Format_RGBA8888:
            self.maske_image = self.maske_image.convertToFormat(QImage.Format.Format_RGBA8888)

        ptr = self.maske_image.bits()
        ptr.setsize(self.maske_image.sizeInBytes())
        
        # np.array(ptr) kopya oluşturabilir, bu yüzden açıkça copy() diyerek niyetimizi belli edelim
        # ve işlem sonunda QImage'i güncelleyelim.
        arr = np.array(ptr).reshape(h, w, 4).copy()
        
        maske_tek_kanal = np.zeros((h, w), dtype=np.uint8)
        maske_tek_kanal[arr[:, :, 3] > 0] = 255
        
        if maske_tek_kanal[y, x] == 255:
            return 
            
        h_mask, w_mask = h + 2, w + 2
        flood_mask = np.zeros((h_mask, w_mask), dtype=np.uint8)
        
        cv2.floodFill(maske_tek_kanal, flood_mask, (x, y), 255)
        
        yeni_dolu_yerler = maske_tek_kanal == 255
        
        arr[yeni_dolu_yerler, 0] = self.maske_rengi.red()
        arr[yeni_dolu_yerler, 1] = self.maske_rengi.green()
        arr[yeni_dolu_yerler, 2] = self.maske_rengi.blue()
        arr[yeni_dolu_yerler, 3] = 255
        
        # Güncellenmiş array'den yeni QImage oluştur ve kaydet
        # arr.data -> buffer
        # bytesPerLine = w * 4 (RGBA)
        self.maske_image = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888).copy()
        
        self.maske_guncelle()
        self.scene.update()
        self.viewport().update()

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)
