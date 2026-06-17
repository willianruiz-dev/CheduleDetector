"""Test: probar OCR con preprocesamiento agresivo para eliminar guilloché."""
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.perspective_corrector import PerspectiveCorrector
from src.preprocessing.image_enhancer import ImageEnhancer
from src.detection.region_detector import RegionDetector, RegionName

img_path = Path("samples/capturas/cc_anverso.jpg")
img = cv2.imread(str(img_path))
h, w = img.shape[:2]

# Corrección de perspectiva
pc = PerspectiveCorrector()
warped, corrected, _ = pc.correct(img)
if corrected:
    img = warped

# Enhancement
enhancer = ImageEnhancer()
enhanced = enhancer.enhance(img)

# Detectar regiones
detector = RegionDetector()
tipo = detector.detect_document_type(img)
coords = detector.detect_regions(img, tipo)

# Probar diferentes preprocesamientos en APELLIDOS_NOMBRES
x, y, rw, rh = coords[RegionName.APELLIDOS_NOMBRES]
roi_color = img[y:y+rh, x:x+rw]
roi_gray = enhanced[y:y+rh, x:x+rw]

# Método 1: CLAHE + morphological opening para eliminar líneas finas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened = cv2.morphologyEx(roi_gray, cv2.MORPH_OPEN, kernel)

# Método 2: Binarización adaptativa + limpieza
binary = cv2.adaptiveThreshold(roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 31, 10)
# Invertir: texto negro sobre blanco
binary_clean = cv2.bitwise_not(binary)

# Método 3: Desenfoque bilateral (preserva bordes) + threshold
bilateral = cv2.bilateralFilter(roi_gray, 9, 75, 75)
_, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Guardar versiones
out_dir = Path("samples/capturas")
cv2.imwrite(str(out_dir / "test_roi_original.jpg"), roi_color)
cv2.imwrite(str(out_dir / "test_roi_gray.jpg"), roi_gray)
cv2.imwrite(str(out_dir / "test_roi_opened.jpg"), opened)
cv2.imwrite(str(out_dir / "test_roi_binary.jpg"), binary_clean)
cv2.imwrite(str(out_dir / "test_roi_otsu.jpg"), otsu)
print("Guardadas versiones de prueba en samples/capturas/test_roi_*.jpg")

# Probar EasyOCR en cada versión
try:
    import easyocr
    reader = easyocr.Reader(["es"], gpu=False, verbose=False)

    for name, img_test in [
        ("original (color)", roi_color),
        ("gray CLAHE", roi_gray),
        ("morph open", opened),
        ("binary clean", binary_clean),
        ("bilateral+otsu", otsu),
    ]:
        results = reader.readtext(img_test)
        texts = [r[1] for r in results] if results else ["(nada)"]
        print(f"\n{name}: {texts}")
except Exception as e:
    print(f"Error OCR: {e}")
