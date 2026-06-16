"""
Ejemplo de uso del lector de cédulas colombianas.

Uso:
    python main.py
    python main.py --imagen ruta/cédula.jpg
    python main.py --backend paddleocr
    python main.py --lote carpeta_con_cedulas/
"""

import argparse
import json
import sys
from pathlib import Path

# Asegurar que src está en el path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline.cedula_reader_pipeline import CedulaReaderPipeline
from src.ocr.ocr_adapter import OCRBackend


def main():
    parser = argparse.ArgumentParser(
        description="Lector de Cédulas Colombianas con TensorFlow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python main.py                                # Demo con texto informativo
  python main.py --imagen mi_cedula.jpg         # Procesar una imagen
  python main.py --imagen cedula.jpg --backend paddleocr
  python main.py --lote ./carpeta_cedulas/      # Procesar lote
        """,
    )

    parser.add_argument(
        "--imagen", "-i",
        type=str,
        help="Ruta a la imagen de la cédula a procesar.",
    )
    parser.add_argument(
        "--lote", "-l",
        type=str,
        help="Ruta a carpeta con imágenes de cédulas para procesar en lote.",
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="auto",
        choices=["auto", "easyocr", "paddleocr"],
        help="Motor de OCR a usar (default: auto).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Usar GPU para OCR si está disponible.",
    )
    parser.add_argument(
        "--umbral", "-u",
        type=float,
        default=0.5,
        help="Umbral de confianza mínima (default: 0.5).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Ruta para guardar el resultado JSON.",
    )

    args = parser.parse_args()

    # ── Inicializar pipeline ───────────────────────────────────
    print("=" * 60)
    print("  LECTOR DE CÉDULAS COLOMBIANAS")
    print("  Powered by TensorFlow & OpenCV")
    print("=" * 60)

    backend_map = {
        "auto": OCRBackend.AUTO,
        "easyocr": OCRBackend.EASYOCR,
        "paddleocr": OCRBackend.PADDLEOCR,
    }

    pipeline = CedulaReaderPipeline(
        ocr_backend=backend_map[args.backend],
        use_gpu=args.gpu,
        confidence_threshold=args.umbral,
    )

    # ── Procesar imagen individual ─────────────────────────────
    if args.imagen:
        print(f"\n📄 Procesando: {args.imagen}")
        cedula = pipeline.procesar_cedula(args.imagen)

        print("\n" + "─" * 40)
        print("  RESULTADO")
        print("─" * 40)
        print(cedula.to_json())

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(cedula.to_json())
            print(f"\n💾 Resultado guardado en: {args.output}")

        return

    # ── Procesar lote ──────────────────────────────────────────
    if args.lote:
        carpeta = Path(args.lote)
        if not carpeta.is_dir():
            print(f"❌ No es una carpeta válida: {carpeta}")
            sys.exit(1)

        imagenes = sorted(carpeta.glob("*"))
        imagenes = [
            str(p) for p in imagenes
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
        ]

        if not imagenes:
            print(f"❌ No se encontraron imágenes en: {carpeta}")
            sys.exit(1)

        print(f"\n📦 Procesando {len(imagenes)} imágenes...")
        resultados = pipeline.procesar_lote(imagenes)

        # Reporte resumen
        print("\n" + "=" * 60)
        print("  RESUMEN DEL LOTE")
        print("=" * 60)

        exitosas = sum(1 for r in resultados if r.es_valida)
        print(f"  ✅ Procesadas exitosamente: {exitosas}/{len(resultados)}")
        print(f"  ⏱  Tiempo promedio: {sum(r.processing_time_ms for r in resultados) // max(len(resultados), 1)}ms")

        # Guardar resultados
        salida = args.output or str(carpeta / "resultados.json")
        resultados_json = [r.to_dict() for r in resultados]

        with open(salida, "w", encoding="utf-8") as f:
            json.dump(resultados_json, f, ensure_ascii=False, indent=2)

        print(f"  💾 Resultados guardados en: {salida}")
        return

    # ── Modo demo (sin argumentos) ─────────────────────────────
    print("\n📋 MODO DEMO")
    print("─" * 40)
    print("No se proporcionó ninguna imagen.")
    print()
    print("Para procesar una cédula, use:")
    print("  python main.py --imagen ruta/a/tu/cedula.jpg")
    print()
    print("Para procesar varias cédulas:")
    print("  python main.py --lote carpeta_con_cedulas/")
    print()
    print("Opciones adicionales:")
    print("  --backend paddleocr   Usar PaddleOCR (alta precisión)")
    print("  --backend easyocr     Usar EasyOCR (rápido)")
    print("  --gpu                 Usar aceleración GPU")
    print("  --umbral 0.7          Subir umbral de confianza")
    print("  --output resultado.json  Guardar en archivo")
    print()
    print("Requisitos:")
    print("  pip install tensorflow opencv-python numpy Pillow scikit-image")
    print("  pip install easyocr    # opcional, OCR rápido")
    print("  pip install paddleocr  # opcional, OCR de alta precisión")
    print("  pip install pyzbar     # opcional, decodificación PDF417")


if __name__ == "__main__":
    main()
