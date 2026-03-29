from ultralytics import YOLO
import cv2

# 1. ONNX modelini yükle
# Ultralytics, ONNX uzantısını otomatik olarak tanır ve çalıştırır.
model = YOLO('models/moon.onnx', task='detect')

# 2. Çıkarım (Inference) yap
# Buraya elindeki bir Ay görüntüsünün yolunu ver
image_path = "test/images/test3.png" 
results = model(image_path)

# 3. Sonuçları incele ve kaydet
for result in results:
    # Bulunan kraterlerin koordinatlarını ve güven skorlarını (0-1 arası) yazdır
    boxes = result.boxes
    print(f"Toplam {len(boxes)} krater bulundu.")
    
    # Görüntüyü üzerinde kutularla birlikte ekranda göster (Colab'de çalışmaz, lokalde çalışır)
    # result.show() 
    
    # Görüntüyü diske kaydet
    # labels=False ile yazıları (etiket ve skor) gizleyip, line_width=1 ile kutuları inceltebiliriz.
    plotted_img = result.plot(labels=False, line_width=1)
    cv2.imwrite("tespit_sonucu.jpg", plotted_img)
    print("Sonuç 'tespit_sonucu.jpg' olarak kaydedildi.")