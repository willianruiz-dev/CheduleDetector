"""
Versión simplificada: OCR sobre toda la imagen y extracción por posición.
No usa regiones predefinidas, deja que EasyOCR encuentre todo el texto.
"""
import sys
import json
from pathlib import Path
import cv2
import easyocr

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.perspective_corrector import PerspectiveCorrector
from src.preprocessing.image_enhancer import ImageEnhancer

img_path = Path("samples/capturas/cc_anverso.jpg")
img = cv2.imread(str(img_path))
print(f"Imagen: {img.shape}")

# Corrección de perspectiva
pc = PerspectiveCorrector()
warped, corrected, _ = pc.correct(img)
if corrected:
    img = warped
    print("✅ Perspectiva corregida")

# Enhancement
enhancer = ImageEnhancer()
enhanced = enhancer.enhance(img)
enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

# OCR en toda la imagen
print("Ejecutando EasyOCR en imagen completa...")
reader = easyocr.Reader(["es"], gpu=False, verbose=False)
results = reader.readtext(enhanced_bgr)

print(f"\nSe encontraron {len(results)} bloques de texto:\n")
print("=" * 70)
for bbox, text, conf in results:
    # bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    x = int(bbox[0][0])
    y = int(bbox[0][1])
    w = int(bbox[2][0] - bbox[0][0])
    h = int(bbox[2][1] - bbox[0][1])
    print(f"  Pos=({x:4d},{y:4d}) {w:4d}x{h:4d} | conf={conf:.2f} | '{text}'")

# Guardar imagen con todo el texto detectado
out = enhanced_bgr.copy()
for bbox, text, conf in results:
    pts = [[int(p[0]), int(p[1])] for p in bbox]
    cv2.polylines(out, [np.array(pts)], True, (0, 255, 0), 2)
    cv2.putText(out, text, (int(bbox[0][0]), int(bbox[0][1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

import numpy as np
cv2.imwrite(str(Path("samples/capturas/test_full_ocr.jpg")), out)
print(f"\n✅ Imagen guardada: samples/capturas/test_full_ocr.jpg")
