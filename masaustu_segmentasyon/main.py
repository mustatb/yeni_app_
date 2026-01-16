import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor, QImageReader
from PyQt6.QtCore import Qt
from arayuz.ana_pencere import AnaPencere

def main():
    try:
        app = QApplication(sys.argv)
        
        # Desteklenen formatları yazdır (Debug için)
        formats = [f.data().decode("ascii") for f in QImageReader.supportedImageFormats()]
        print(f"DEBUG: Desteklenen Görüntü Formatları: {formats}")
        
        # Stil ayarları (Dark Mode)
        app.setStyle("Fusion")
        
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        app.setPalette(dark_palette)
        
        pencere = AnaPencere()
        pencere.show()
        
        sys.exit(app.exec())
    except Exception as e:
        print(f"KRİTİK HATA: Uygulama başlatılamadı!\n{e}")
        import traceback
        traceback.print_exc()
        input("Kapatmak için Enter'a basın...") # Konsolun hemen kapanmasını önle

if __name__ == "__main__":
    main()
