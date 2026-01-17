from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QListWidget, QLabel, QSlider, QFileDialog, 
                             QMessageBox, QCheckBox, QGroupBox, QApplication, QColorDialog,
                             QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QMouseEvent
import os
import sys

# Proje kökünü path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arayuz.segmentasyon_alani import SegmentasyonAlani
from mantik import dosya_yonetimi, maske_islemleri, otomatik_segmentasyon
from mantik.ai_model import load_model



class ColorButton(QPushButton):
    """
    Özel renk seçimi için dairesel buton.
    """
    doubleClicked = pyqtSignal()
    colorSelected = pyqtSignal(QColor)

    def __init__(self, color, is_custom=False, parent=None):
        super().__init__(parent)
        self.setFixedSize(24, 24)
        self._color = QColor(color)
        self.is_custom = is_custom
        self.is_active = False
        
        # Gölge efekti (Glow)
        self.shadow = QGraphicsDropShadowEffect()
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(QColor(255, 255, 255, 0)) # Başlangıçta görünmez
        self.shadow.setOffset(0, 0)
        self.setGraphicsEffect(self.shadow)
        
        self.update_style()

    def color(self):
        return self._color

    def set_color(self, color):
        self._color = QColor(color)
        self.update_style()

    def set_active(self, active):
        self.is_active = active
        if active:
            self.shadow.setColor(QColor(255, 255, 255, 200)) # Parlama
        else:
            self.shadow.setColor(QColor(255, 255, 255, 0))
        self.update_style()

    def update_style(self):
        border = "2px solid white" if self.is_active else "none"
        # Hover durumunu stylesheet içinde hallediyoruz
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self._color.name()};
                border-radius: 12px;
                border: {border};
            }}
            QPushButton:hover {{
                border: 2px solid rgba(255, 255, 255, 150);
            }}
        """)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.colorSelected.emit(self._color)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if self.is_custom and event.button() == Qt.MouseButton.LeftButton:
            self.doubleClicked.emit()
        super().mouseDoubleClickEvent(event)

class AnaPencere(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Masaüstü Segmentasyon Uygulaması")
        self.resize(1200, 800)
        
        # Durum değişkenleri
        self.aktif_klasor = ""
        self.dosya_listesi = []
        self.aktif_dosya_indeksi = -1
        
        # Renk Paleti Durumu
        self.custom_color = QColor(128, 128, 128) # Başlangıç gri
        self.color_buttons = []
        
        # AI Model yükleme
        self.ai_model = None
        self._yukle_ai_model()
        
        # Arayüzü kur
        self.arayuzu_olustur()
        


    def arayuzu_olustur(self):
        ana_widget = QWidget()
        self.setCentralWidget(ana_widget)
        ana_layout = QHBoxLayout(ana_widget)
        
        # --- SOL PANEL (Dosya Listesi ve Klasör Seçimi) ---
        sol_panel = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.btn_klasor_sec = QPushButton("Klasör Seç")
        self.btn_klasor_sec.clicked.connect(self.klasor_sec)
        btn_layout.addWidget(self.btn_klasor_sec)
        
        self.btn_dosya_ac = QPushButton("Dosya Aç")
        self.btn_dosya_ac.clicked.connect(self.dosya_ac)
        btn_layout.addWidget(self.btn_dosya_ac)
        
        sol_panel.addLayout(btn_layout)
        
        self.lbl_klasor = QLabel("Henüz bir klasör veya dosya seçilmedi")
        self.lbl_klasor.setWordWrap(True)
        sol_panel.addWidget(self.lbl_klasor)
        
        self.liste_widget = QListWidget()
        self.liste_widget.currentRowChanged.connect(self.dosya_secildi)
        sol_panel.addWidget(self.liste_widget)
        
        nav_layout = QHBoxLayout()
        self.btn_onceki = QPushButton("Önceki")
        self.btn_onceki.clicked.connect(self.onceki_goruntu)
        
        self.lbl_sayac = QLabel("0 / 0")
        self.lbl_sayac.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Biraz stil verelim
        self.lbl_sayac.setStyleSheet("font-weight: bold; font-size: 14px;")
        
        self.btn_sonraki = QPushButton("Sonraki")
        self.btn_sonraki.clicked.connect(self.sonraki_goruntu)
        
        nav_layout.addWidget(self.btn_onceki)
        nav_layout.addWidget(self.lbl_sayac)
        nav_layout.addWidget(self.btn_sonraki)
        sol_panel.addLayout(nav_layout)
        
        ana_layout.addLayout(sol_panel, 1) # Sol panel oranı 1
        
        # --- ORTA PANEL (Segmentasyon Alanı) ---
        orta_panel = QVBoxLayout()
        self.segmentasyon_alani = SegmentasyonAlani()
        orta_panel.addWidget(self.segmentasyon_alani)
        ana_layout.addLayout(orta_panel, 4) # Orta panel oranı 4
        
        # --- SAĞ PANEL (Araçlar) ---
        sag_panel = QVBoxLayout()
        
        # Araçlar Grubu
        grp_araclar = QGroupBox("Araçlar")
        layout_araclar = QVBoxLayout()
        
        self.btn_firca = QPushButton("Fırça")
        self.btn_firca.setCheckable(True)
        self.btn_firca.setChecked(True)
        self.btn_firca.clicked.connect(lambda: self.arac_degistir("firca"))
        layout_araclar.addWidget(self.btn_firca)
        
        self.btn_silgi = QPushButton("Silgi")
        self.btn_silgi.setCheckable(True)
        self.btn_silgi.clicked.connect(lambda: self.arac_degistir("silgi"))
        layout_araclar.addWidget(self.btn_silgi)
        
        self.btn_pan = QPushButton("Pan (Taşı)")
        self.btn_pan.setCheckable(True)
        self.btn_pan.clicked.connect(lambda: self.arac_degistir("pan"))
        layout_araclar.addWidget(self.btn_pan)

        self.btn_kova = QPushButton("Boya Kovası")
        self.btn_kova.setCheckable(True)
        self.btn_kova.clicked.connect(lambda: self.arac_degistir("kova"))
        layout_araclar.addWidget(self.btn_kova)
        
        grp_araclar.setLayout(layout_araclar)
        sag_panel.addWidget(grp_araclar)
        
        # Ayarlar Grubu
        grp_ayarlar = QGroupBox("Ayarlar")
        layout_ayarlar = QVBoxLayout()
        
        layout_ayarlar.addWidget(QLabel("Fırça Boyutu"))
        self.sld_firca = QSlider(Qt.Orientation.Horizontal)
        self.sld_firca.setRange(1, 50)
        self.sld_firca.setValue(10)
        self.sld_firca.valueChanged.connect(self.segmentasyon_alani.set_firca_boyutu)
        layout_ayarlar.addWidget(self.sld_firca)
        
        # --- RENK PALETİ ---
        layout_ayarlar.addWidget(QLabel("Kalem Rengi"))
        layout_renkler = QHBoxLayout()
        layout_renkler.setSpacing(10)
        layout_renkler.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Sabit Renkler
        sabit_renkler = [
            "#000000", # Siyah
            "#FF0000", # Kırmızı
            "#00FF00", # Yeşil
            "#007AFF", # Mavi
            "#FFD600", # Sarı
            "#3434CB"  # Özel Mavi (Görselden)
        ]
        
        for hex_code in sabit_renkler:
            btn = ColorButton(hex_code)
            btn.colorSelected.connect(self.renk_secildi)
            layout_renkler.addWidget(btn)
            self.color_buttons.append(btn)
            
        # Özel Renk Butonu
        self.btn_custom_color = ColorButton(self.custom_color, is_custom=True)
        self.btn_custom_color.colorSelected.connect(self.renk_secildi)
        self.btn_custom_color.doubleClicked.connect(self.ozel_renk_secici_ac)
        layout_renkler.addWidget(self.btn_custom_color)
        self.color_buttons.append(self.btn_custom_color)
        
        layout_ayarlar.addLayout(layout_renkler)
        # -------------------
        
        layout_ayarlar.addWidget(QLabel("Maske Saydamlığı"))
        self.sld_opaklik = QSlider(Qt.Orientation.Horizontal)
        self.sld_opaklik.setRange(0, 100)
        self.sld_opaklik.setValue(60)
        self.sld_opaklik.valueChanged.connect(lambda v: self.segmentasyon_alani.set_maske_opaklik(v / 100.0))
        layout_ayarlar.addWidget(self.sld_opaklik)
        
        self.chk_maske_goster = QCheckBox("Maskeyi Göster")
        self.chk_maske_goster.setChecked(True)
        self.chk_maske_goster.toggled.connect(self.maske_gorunurluk_degistir)
        layout_ayarlar.addWidget(self.chk_maske_goster)
        
        grp_ayarlar.setLayout(layout_ayarlar)
        sag_panel.addWidget(grp_ayarlar)
        
        # İşlemler Grubu
        grp_islemler = QGroupBox("İşlemler")
        layout_islemler = QVBoxLayout()
        
        self.btn_otomatik = QPushButton("Otomatik Kemiği Düzelt")
        self.btn_otomatik.clicked.connect(self.otomatik_duzelt)
        # Biraz renklendirelim
        self.btn_otomatik.setStyleSheet("background-color: #2a82da; color: white; font-weight: bold;")
        layout_islemler.addWidget(self.btn_otomatik)
        
        self.btn_geri_al = QPushButton("Geri Al (Undo)")
        self.btn_geri_al.clicked.connect(self.segmentasyon_alani.geri_al)
        layout_islemler.addWidget(self.btn_geri_al)
        
        self.btn_ileri_al = QPushButton("İleri Al (Redo)")
        self.btn_ileri_al.clicked.connect(self.segmentasyon_alani.ileri_al)
        layout_islemler.addWidget(self.btn_ileri_al)
        
        self.btn_kaydet = QPushButton("Maskeyi Kaydet")
        self.btn_kaydet.clicked.connect(self.maskeyi_kaydet)
        layout_islemler.addWidget(self.btn_kaydet)
        
        self.btn_temizle = QPushButton("Maskeyi Temizle")
        self.btn_temizle.clicked.connect(self.maskeyi_temizle)
        layout_islemler.addWidget(self.btn_temizle)
        
        grp_islemler.setLayout(layout_islemler)
        sag_panel.addWidget(grp_islemler)
        
        sag_panel.addStretch()
        ana_layout.addLayout(sag_panel, 1) # Sağ panel oranı 1

    def klasor_sec(self):
        klasor = QFileDialog.getExistingDirectory(self, "Görüntü Klasörü Seç")
        if klasor:
            self.klasoru_yukle(klasor)

    def dosya_ac(self):
        dosya_yolu, _ = QFileDialog.getOpenFileName(self, "Görüntü Dosyası Aç", "", "Tüm Görüntüler (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;PNG Dosyaları (*.png);;JPG Dosyaları (*.jpg *.jpeg);;Tüm Dosyalar (*)")
        if dosya_yolu:
            klasor = os.path.dirname(dosya_yolu)
            dosya_adi = os.path.basename(dosya_yolu)
            self.klasoru_yukle(klasor)
            
            # Seçilen dosyayı listede bul ve seç
            items = self.liste_widget.findItems(dosya_adi, Qt.MatchFlag.MatchExactly)
            if items:
                self.liste_widget.setCurrentItem(items[0])

    def klasoru_yukle(self, klasor):
        self.aktif_klasor = klasor
        self.lbl_klasor.setText(klasor)
        self.dosya_listesini_guncelle()
        


    def dosya_listesini_guncelle(self):
        self.dosya_listesi = dosya_yonetimi.goruntu_listesini_getir(self.aktif_klasor)
        self.liste_widget.clear()
        self.liste_widget.addItems(self.dosya_listesi)
        
        if self.dosya_listesi:
            self.liste_widget.setCurrentRow(0)
        else:
            QMessageBox.information(self, "Bilgi", "Klasörde PNG dosyası bulunamadı.")

    def dosya_secildi(self, index):
        if index < 0 or index >= len(self.dosya_listesi):
            return
            
        self.aktif_dosya_indeksi = index
        
        # Sayacı güncelle
        self.lbl_sayac.setText(f"{index + 1} / {len(self.dosya_listesi)}")
        
        dosya_adi = self.dosya_listesi[index]
        tam_yol = os.path.join(self.aktif_klasor, dosya_adi)
        
        # Görüntüyü yükle
        basari = self.segmentasyon_alani.goruntu_yukle(tam_yol)
        if not basari:
            QMessageBox.warning(self, "Hata", f"Görüntü yüklenemedi veya format desteklenmiyor:\n{dosya_adi}")
            return
        
        # Varsa kayıtlı maskeyi yükle, yoksa AI segmentasyon yap
        maske_yolu = dosya_yonetimi.maske_yolunu_olustur(self.aktif_klasor, dosya_adi)
        if dosya_yonetimi.maske_var_mi(maske_yolu):
            # Kayıtlı maske var, onu yükle
            maske = maske_islemleri.maske_yukle(maske_yolu)
            self.segmentasyon_alani.maske_yukle(maske)
            self.statusBar().showMessage(f"Kayıtlı maske yüklendi: {dosya_adi}", 2000)
        else:
            # Kayıtlı maske yok, AI segmentasyon yap
            self.segmentasyon_alani.maske_temizle()
            self._ai_otomatik_segmentasyon(tam_yol)

    def onceki_goruntu(self):
        if self.aktif_dosya_indeksi > 0:
            self.liste_widget.setCurrentRow(self.aktif_dosya_indeksi - 1)

    def sonraki_goruntu(self):
        if self.aktif_dosya_indeksi < len(self.dosya_listesi) - 1:
            self.liste_widget.setCurrentRow(self.aktif_dosya_indeksi + 1)

    def arac_degistir(self, arac):
        self.btn_firca.setChecked(arac == "firca")
        self.btn_silgi.setChecked(arac == "silgi")
        self.btn_pan.setChecked(arac == "pan")
        self.btn_kova.setChecked(arac == "kova")
        self.segmentasyon_alani.set_arac_modu(arac)

    def maske_gorunurluk_degistir(self, gorunur):
        self.segmentasyon_alani.maske_item.setVisible(gorunur)

    def renk_secildi(self, renk):
        """Renk butonuna tıklandığında çağrılır."""
        # Tüm butonların aktifliğini kaldır
        for btn in self.color_buttons:
            btn.set_active(False)
            
        # Gönderen butonu bul ve aktif yap
        sender = self.sender()
        if isinstance(sender, ColorButton):
            sender.set_active(True)
            
        # Segmentasyon alanını güncelle
        self.segmentasyon_alani.set_maske_rengi(renk)

    def ozel_renk_secici_ac(self):
        """Özel renk butonuna çift tıklandığında açılır."""
        renk = QColorDialog.getColor(self.custom_color, self, "Özel Renk Seç")
        if renk.isValid():
            self.custom_color = renk
            self.btn_custom_color.set_color(renk)
            
            # Rengi seç ve aktif yap
            self.renk_secildi(renk)
            # Manuel olarak aktif yap çünkü sender() burada farklı olabilir veya renk_secildi'yi doğrudan çağırmak daha temiz
            for btn in self.color_buttons:
                btn.set_active(False)
            self.btn_custom_color.set_active(True)

    def maskeyi_kaydet(self):
        if self.aktif_dosya_indeksi == -1:
            return
            
        dosya_adi = self.dosya_listesi[self.aktif_dosya_indeksi]
        maske_yolu = dosya_yonetimi.maske_yolunu_olustur(self.aktif_klasor, dosya_adi)
        
        # Overlay için yol oluştur: <dosyaAdi>_seg.png
        isim, uzanti = os.path.splitext(dosya_adi)
        seg_dosya_adi = f"{isim}_seg.png" # Her zaman PNG olsun, kullanıcı isteği
        seg_yolu = os.path.join(self.aktif_klasor, seg_dosya_adi)
        
        # Orijinal resim yolu
        orijinal_yol = os.path.join(self.aktif_klasor, dosya_adi)
        
        maske_image = self.segmentasyon_alani.maske_al()
        if maske_image:
            # 1. Binary Maskeyi Kaydet
            basari_maske = maske_islemleri.maske_kaydet(maske_image, maske_yolu)
            
            # 2. Overlay Görüntüyü Kaydet
            # Opaklık %40 (0.4)
            # QColor RGB'dir, OpenCV BGR ister.
            r = self.segmentasyon_alani.maske_rengi.red()
            g = self.segmentasyon_alani.maske_rengi.green()
            b = self.segmentasyon_alani.maske_rengi.blue()
            
            basari_seg = maske_islemleri.overlay_kaydet(orijinal_yol, maske_image, seg_yolu, opaklik=self.segmentasyon_alani.maske_opaklik, renk=(b, g, r))
            
            if basari_maske and basari_seg:
                QMessageBox.information(self, "Başarılı", f"Dosyalar başarıyla kaydedildi:\n\n1. {os.path.basename(maske_yolu)}\n2. {os.path.basename(seg_yolu)}")
            elif basari_maske:
                QMessageBox.warning(self, "Kısmi Başarı", f"Maske kaydedildi ancak overlay kaydedilemedi.\nMaske: {maske_yolu}")
            else:
                QMessageBox.critical(self, "Hata", "Maske kaydedilemedi!")

    def maskeyi_temizle(self):
        cevap = QMessageBox.question(self, "Onay", "Maskeyi tamamen temizlemek istediğinize emin misiniz?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if cevap == QMessageBox.StandardButton.Yes:
            self.segmentasyon_alani.durumu_kaydet() # Temizlemeden önce kaydet
            self.segmentasyon_alani.maske_temizle()

    def otomatik_duzelt(self):
        if self.aktif_dosya_indeksi == -1:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir görüntü seçin.")
            return
            
        # Mevcut maskeyi al
        mevcut_maske = self.segmentasyon_alani.maske_al()
        if not mevcut_maske:
            QMessageBox.warning(self, "Uyarı", "Otomatik düzeltme için önce kabaca bir çizim yapmalısınız.")
            return
            
        dosya_adi = self.dosya_listesi[self.aktif_dosya_indeksi]
        tam_yol = os.path.join(self.aktif_klasor, dosya_adi)
        
        try:
            # İşlem başlıyor mesajı
            self.statusBar().showMessage("Otomatik segmentasyon yapılıyor...")
            QApplication.processEvents() 
            
            yeni_maske = otomatik_segmentasyon.otomatik_duzelt(tam_yol, mevcut_maske)
            
            # Değişiklik kontrolü
            import numpy as np
            eski_np = np.array(mevcut_maske)
            yeni_np = np.array(yeni_maske)
            
            fark = np.sum(eski_np != yeni_np)
            toplam_piksel = eski_np.size
            degisim_orani = fark / toplam_piksel
            
            # Eğer yeni maske tamamen boşsa
            if np.sum(yeni_np) == 0:
                self.statusBar().showMessage("Otomatik segmentasyon başarısız (Boş maske). Kaba çizim korunuyor.", 3000)
                QMessageBox.warning(self, "Uyarı", "Otomatik segmentasyon sonucu boş döndü. Kaba çizim korundu.")
            else:
                # Her durumda uygula (küçük fark olsa bile)
                self.segmentasyon_alani.durumu_kaydet() # Değiştirmeden önce kaydet
                self.segmentasyon_alani.maske_yukle(yeni_maske)
                self.statusBar().showMessage("Otomatik segmentasyon uygulandı.", 3000)
                # Kullanıcıyı her seferinde popup ile rahatsız etmeyelim, status bar yeterli olabilir
                # Ama kullanıcı "uygulandı" mesajını net görmek istiyor
                # QMessageBox.information(self, "Bilgi", "Otomatik segmentasyon uygulandı.")
                
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Otomatik segmentasyon sırasında hata oluştu:\n{e}")
            self.statusBar().showMessage("Hata oluştu.")
    
    def _yukle_ai_model(self):
        """
        AI modelini yükler (uygulama başlangıcında).
        """
        try:
            # Model dosyası yolu: yeni_app klasöründe
            model_dosyasi = "calcaneus_ultimate_model.pth"
            
            # Ana uygulama dizinini bul (masaustu_segmentasyon)
            app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_yolu = os.path.join(app_dir, model_dosyasi)
            
            if not os.path.exists(model_yolu):
                print(f"UYARI: AI Model dosyası bulunamadı: {model_yolu}")
                print("AI segmentasyon özelliği devre dışı.")
                return
            
            # Status bar'da bilgi ver
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage("AI Model yükleniyor...", 5000)
            
            print(f"AI Model yükleniyor: {model_yolu}")
            
            # Modeli yükle
            self.ai_model = load_model(model_yolu)
            
            print("✓ AI Model başarıyla yüklendi ve hazır.")
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage("AI Model hazır.", 3000)
                
        except Exception as e:
            print(f"AI Model yüklenirken hata oluştu: {e}")
            import traceback
            traceback.print_exc()
            
            # Kullanıcıya bilgi ver
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage("AI Model yüklenemedi. Manuel segmentasyon modu.", 5000)
    
    def _ai_otomatik_segmentasyon(self, goruntu_yolu: str):
        """
        Görüntü için AI otomatik segmentasyon yapar.
        
        Args:
            goruntu_yolu: Görüntü dosyasının tam yolu
        """
        if self.ai_model is None:
            # Model yüklenmemişse sessizce atla
            return
        
        try:
            # Status bar güncelle
            self.statusBar().showMessage("AI segmentasyonu yapılıyor...", 0)
            QApplication.processEvents()
            
            # AI segmentasyon yap
            maske, basarili, mesaj = otomatik_segmentasyon.ai_segmentasyon(goruntu_yolu)
            
            if basarili and maske is not None:
                # Başarılı, maskeyi yükle
                self.segmentasyon_alani.maske_yukle(maske)
                self.statusBar().showMessage(f"AI segmentasyonu tamamlandı: {mesaj}", 3000)
            else:
                # Başarısız veya validasyon geçmedi
                self.statusBar().showMessage(f"AI segmentasyonu: {mesaj}", 4000)
                # Boş maske bırak, kullanıcı manuel çizebilir
                
        except Exception as e:
            print(f"AI segmentasyon hatası: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage("AI segmentasyon hatası.", 3000)
