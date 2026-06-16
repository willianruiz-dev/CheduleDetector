"""
Demo visual del pipeline - muestra el preprocesamiento y detección de regiones
sin necesidad de ejecutar EasyOCR (que es muy pesado en CPU).

Uso:
    python demo_visual.py samples/cc_anverso.jpg
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.perspective_corrector import PerspectiveCorrector
from src.preprocessing.image_enhancer import ImageEnhancer
from src.detection.region_detector import RegionDetector, RegionName


def dibujar_regiones(img, coords):
    """Dibuja rectángulos de colores sobre las regiones detectadas."""
    colors = {
        RegionName.FOTO: (0, 255, 0),           # verde
        RegionName.NUMERO_CEDULA: (255, 0, 0),  # azul
        RegionName.APELLIDOS_NOMBRES: (255, 100, 0),  # naranja
        RegionName.FECHA_NACIMIENTO: (0, 255, 255),   # amarillo
        RegionName.LUGAR_NACIMIENTO: (255, 0, 255),   # magenta
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
    return out


def main():
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
    else:
        img_path = Path("samples/cc_anverso.jpg")

    if not img_path.exists():
        print(f"❌ No existe: {img_path}")
        print("Uso: python demo_visual.py [ruta_imagen]")
        sys.exit(1)

    print("=" * 60)
    print("  DEMO VISUAL - LECTOR DE CÉDULAS COLOMBIANAS")
    print("=" * 60)

    # ── Cargar ─────────────────────────────────────────────────
    original = cv2.imread(str(img_path))
    if original is None:
        print(f"❌ No se pudo cargar: {img_path}")
        sys.exit(1)

    print(f"\n📷 Imagen: {img_path.name}  ({original.shape[1]}x{original.shape[0]})")

    # ── 1. Perspectiva ────────────────────────────────────────
    print("\n1️⃣  Corrección de perspectiva...")
    pc = PerspectiveCorrector()
    warped, corrected, matrix = pc.correct(original)
    print(f"   {'✅ Corregida' if corrected else '⚠ Sin cambios'}")

    # ── 2. Mejora ─────────────────────────────────────────────
    print("\n2️⃣  Mejora de imagen (CLAHE + denoise + sharpen)...")
    enhancer = ImageEnhancer()
    enhanced_gray = enhancer.enhance(warped)
    enhanced_color = enhancer.enhance_color(warped)
    print("   ✅ Imagen mejorada (CLAHE 2.0, fastNlMeans)")

    # ── 3. Detección tipo ─────────────────────────────────────
    print("\n3️⃣  Detección de tipo de documento...")
    detector = RegionDetector()
    tipo = detector.detect_document_type(warped if corrected else original)
    print(f"   ✅ Tipo: {tipo.value}")

    # ── 4. Regiones ───────────────────────────────────────────
    print("\n4️⃣  Detección de regiones...")
    detector = RegionDetector()

    # Intentar encontrar el documento en el encuadre
    doc_rect = detector._find_document(warped)
    if doc_rect:
        dx, dy, dw, dh = doc_rect
        print(f"   📄 Documento encontrado en ({dx},{dy}) {dw}x{dh}")
        cv2.rectangle(warped, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 3)

    coords = detector.detect_regions(warped, tipo)
    print(f"   ✅ {len(coords)} regiones detectadas:")
    for name, (x, y, w, h) in coords.items():
        print(f"     • {name}: ({x},{y}) {w}x{h}")

    # ── Guardar visualizaciones ────────────────────────────
    output_dir = Path("samples/demo_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Regiones sobre original
    annotated = dibujar_regiones(warped, coords)
    cv2.imwrite(str(output_dir / "01_regiones_detectadas.jpg"),
                annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\n   💾 Regiones detectadas: {output_dir / '01_regiones_detectadas.jpg'}")

    # Recortes de cada región
    for name, (x, y, w, h) in coords.items():
        roi = enhanced_gray[y:y + h, x:x + w]
        if roi.size > 0:
            safe_name = str(name).replace(".", "_")
            cv2.imwrite(str(output_dir / f"roi_{safe_name}.jpg"),
                        roi, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"   💾 ROIs guardados en: {output_dir}/")

    # ── Comparativa ────────────────────────────────────────
    comparativa = np.hstack([
        cv2.resize(original, (640, 400)),
        cv2.resize(annotated, (640, 400)),
    ])
    cv2.imwrite(str(output_dir / "00_comparativa.jpg"),
                comparativa, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"   💾 Comparativa: {output_dir / '00_comparativa.jpg'}")

    print("\n✅ Demo completada. Revisa la carpeta samples/demo_output/\n")

    # ── Resumen ────────────────────────────────────────────
    print("Resumen de datos que extraería el OCR:")
    print(f"  • Tipo: {tipo.value}")
    print(f"  • Regiones detectadas: {len(coords)}")
    if tipo.value == "CC":
        print("  • Campos esperados en ANVERSO: número, apellidos, nombres, firma")


if __name__ == "__main__":
    main()
