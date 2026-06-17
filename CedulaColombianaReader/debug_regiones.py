"""Debug: muestra las regiones detectadas sobre la imagen."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.perspective_corrector import PerspectiveCorrector
from src.preprocessing.image_enhancer import ImageEnhancer
from src.detection.region_detector import RegionDetector, RegionName

def dibujar_regiones(img, coords):
    colors = {
        RegionName.FOTO: (0, 255, 0),
        RegionName.NUMERO_CEDULA: (255, 0, 0),
        RegionName.APELLIDOS_NOMBRES: (255, 100, 0),
        RegionName.FECHA_NACIMIENTO: (0, 255, 255),
        RegionName.LUGAR_NACIMIENTO: (255, 0, 255),
        RegionName.ESTATURA: (128, 0, 128),
        RegionName.GRUPO_SANGUINEO: (0, 128, 128),
        RegionName.SEXO: (128, 128, 0),
        RegionName.FECHA_EXPEDICION: (0, 0, 255),
        RegionName.FIRMA: (255, 255, 0),
        RegionName.HUELLA: (100, 100, 255),
        RegionName.PDF417: (0, 0, 0),
    }
    out = img.copy()
    for region_name, (x, y, w, h) in coords.items():
        color = colors.get(region_name, (200, 200, 200))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = str(region_name).replace("RegionName.", "")
        cv2.putText(out, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out

img_path = Path("samples/capturas/cc_anverso.jpg")
if not img_path.exists():
    print(f"No existe: {img_path}")
    sys.exit(1)

img = cv2.imread(str(img_path))
print(f"Imagen original: {img.shape}")

# Auto-rotar si está vertical
h_img, w_img = img.shape[:2]
if h_img > w_img:
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    print(f"Rotada a: {img.shape}")

# Corrección de perspectiva
pc = PerspectiveCorrector()
warped, corrected, _ = pc.correct(img)
if corrected:
    img = warped
    print("Perspectiva corregida")

# Detectar tipo y regiones
detector = RegionDetector()
tipo = detector.detect_document_type(img)
print(f"Tipo: {tipo.value}")

coords = detector.detect_regions(img, tipo)
print(f"Regiones detectadas: {len(coords)}")
for name, (x, y, w, h) in coords.items():
    print(f"  {name}: ({x},{y}) {w}x{h}")

# Guardar imagen anotada
annotated = dibujar_regiones(img, coords)
out_path = Path("samples/capturas/debug_regiones.jpg")
cv2.imwrite(str(out_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
print(f"\nGuardado: {out_path}")

# También guardar cada ROI
for name, (x, y, w, h) in coords.items():
    roi = img[y:y+h, x:x+w]
    if roi.size > 0:
        safe = str(name).replace(".", "_")
        cv2.imwrite(str(Path(f"samples/capturas/debug_{safe}.jpg")), roi)
