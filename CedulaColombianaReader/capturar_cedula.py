"""
Captura tu cédula colombiana con la webcam y la procesa.

Uso:
    python capturar_cedula.py
    python capturar_cedula.py --ocr easyocr
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.perspective_corrector import PerspectiveCorrector
from src.preprocessing.image_enhancer import ImageEnhancer
from src.detection.region_detector import RegionDetector, RegionName


def dibujar_guias(frame):
    """Dibuja guías para alinear la cédula y texto informativo."""
    h, w = frame.shape[:2]

    # Rectángulo guía (proporción ~1.6:1 de la cédula)
    guia_w = int(w * 0.75)
    guia_h = int(guia_w / 1.6)
    guia_x = (w - guia_w) // 2
    guia_y = (h - guia_h) // 2

    cv2.rectangle(frame, (guia_x, guia_y),
                  (guia_x + guia_w, guia_y + guia_h), (0, 255, 255), 2)

    # Esquinas
    corner_len = 40
    color = (0, 255, 255)
    for cx, cy in [(guia_x, guia_y), (guia_x + guia_w, guia_y),
                   (guia_x, guia_y + guia_h), (guia_x + guia_w, guia_y + guia_h)]:
        dx = corner_len if cx == guia_x else -corner_len
        dy = corner_len if cy == guia_y else -corner_len
        cv2.line(frame, (cx, cy), (cx + dx, cy), color, 3)
        cv2.line(frame, (cx, cy), (cx, cy + dy), color, 3)

    # Texto informativo
    cv2.putText(frame, "CENTRA tu cedula en el recuadro", (guia_x, guia_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "[ESPACIO] Capturar ANVERSO  |  [R] Capturar REVERSO  |  [ESC] Salir",
                (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def dibujar_regiones(img, coords):
    """Dibuja rectángulos de colores sobre las regiones detectadas."""
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
        label = str(region_name).replace("RegionName.", "")[:12]
        cv2.putText(out, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    return out


def procesar_imagen(image, ocr_mode=True):
    """Pipeline completo de procesamiento."""
    resultados_dir = Path("samples/capturas")
    resultados_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("  PROCESANDO CÉDULA CAPTURADA")
    print("=" * 55)

    # ── 1. Corrección de perspectiva ────────────────────────
    print("\n1️⃣  Corrección de perspectiva...")
    pc = PerspectiveCorrector()
    warped, corrected, _ = pc.correct(image)

    if corrected:
        print("   ✅ Perspectiva corregida")
    else:
        print("   ⚠ No se detectaron 4 esquinas, usando imagen original")

    img_proc = warped if corrected else image
    cv2.imwrite(str(resultados_dir / "01_corregida.jpg"), img_proc,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # ── 2. Mejora ───────────────────────────────────────────
    print("2️⃣  Mejora de imagen (CLAHE + denoise)...")
    enhancer = ImageEnhancer()
    enhanced = enhancer.enhance(img_proc)

    # ── 3. Tipo ──────────────────────────────────────────────
    print("3️⃣  Detectando tipo...")
    detector = RegionDetector()
    tipo = detector.detect_document_type(img_proc)
    print(f"   ✅ {tipo.value}")

    # ── 4. Regiones ──────────────────────────────────────────
    print("4️⃣  Detectando regiones...")
    detector = RegionDetector()

    # Verificar si encuentra el documento en el encuadre
    doc_rect = detector._find_document(img_proc)
    if doc_rect:
        dx, dy, dw, dh = doc_rect
        print(f"   📄 Documento encontrado en ({dx},{dy}) {dw}x{dh}")
        cv2.rectangle(img_proc, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 3)

    coords = detector.detect_regions(img_proc, tipo)
    print(f"   ✅ {len(coords)} regiones")

    # Guardar imagen con regiones
    annotated = dibujar_regiones(img_proc, coords)
    cv2.imwrite(str(resultados_dir / "02_regiones.jpg"), annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Guardar ROIs
    print("\n   ROIs extraídos:")
    rois = {}
    for name, (x, y, w, h) in coords.items():
        roi = enhanced[y:y + h, x:x + w]
        if roi.size > 0:
            rois[str(name)] = True
            safe = str(name).replace(".", "_")
            cv2.imwrite(str(resultados_dir / f"roi_{safe}.jpg"), roi,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"     • {name}: {w}x{h}")

    # ── 5. OCR (si se pide) ─────────────────────────────────
    if ocr_mode and rois:
        print("\n5️⃣  Ejecutando OCR...")
        try:
            from src.pipeline.cedula_reader_pipeline import CedulaReaderPipeline
            pipe = CedulaReaderPipeline(ocr_backend="easyocr")
            result = pipe.procesar_cedula(img_proc)
            print("\n📋 Datos extraídos:")
            print(result.to_json(indent=2))
        except Exception as e:
            print(f"   ⚠ OCR no disponible: {e}")
            print("   (Las regiones se guardaron igual, puedes usar otro motor)")

    print(f"\n✅ Resultados en: {resultados_dir.resolve()}/")
    print(f"   • 01_corregida.jpg - Imagen con perspectiva corregida")
    print(f"   • 02_regiones.jpg - Regiones detectadas")
    print(f"   • roi_*.jpg - Recortes de cada campo")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Captura tu cédula con la webcam")
    parser.add_argument("--ocr", action="store_true", help="Ejecutar OCR tras capturar")
    parser.add_argument("--camara", type=int, default=0, help="Índice de cámara")
    args = parser.parse_args()

    print("\n📷 INICIANDO CÁMARA...")
    print("=" * 55)
    print("  CAPTURA DE CÉDULA COLOMBIANA")
    print("=" * 55)
    print()
    print("  🟡 [ESPACIO] = Capturar ANVERSO (foto + datos)")
    print("  🔵 [R]      = Capturar REVERSO (huella + PDF417)")
    print("  ⚪ [ESC]    = Salir")
    print()

    cap = cv2.VideoCapture(args.camara)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara. Prueba con --camara 1")
        sys.exit(1)

    # Intentar máxima resolución
    for res in [(1920, 1080), (1280, 720), (800, 600), (640, 480)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w >= 800:
            break
    print(f"   📹 Cámara abierta: {w}x{h}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al leer de la cámara")
            break

        display = frame.copy()
        display = dibujar_guias(display)

        cv2.imshow("Captura de Cedula Colombiana", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("\n👋 Saliendo...")
            break
        elif key == 32:  # ESPACIO - Anverso
            print("\n📸 ¡Captura ANVERSO!")
            cv2.imwrite("samples/capturas/anverso_capturado.jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            procesar_imagen(frame, ocr_mode=args.ocr)
        elif key == ord('r') or key == ord('R'):  # Reverso
            print("\n📸 ¡Captura REVERSO!")
            cv2.imwrite("samples/capturas/reverso_capturado.jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            procesar_imagen(frame, ocr_mode=args.ocr)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
