"""
Pipeline principal de lectura de cédulas colombianas.

Orquesta todo el flujo:
1. Carga de imagen
2. Preprocesamiento (mejora, corrección de perspectiva, binarización)
3. Decodificación PDF417
4. Detección de tipo de documento y regiones
5. OCR sobre cada región
6. Extracción de campos
7. Validación cruzada OCR vs PDF417
"""

import time
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src.preprocessing.image_enhancer import ImageEnhancer
from src.preprocessing.perspective_corrector import PerspectiveCorrector
from src.preprocessing.binarizer import Binarizer
from src.pdf417.barcode_decoder import BarcodeDecoder
from src.pdf417.barcode_parser import BarcodeParser
from src.detection.region_detector import RegionDetector, RegionName
from src.ocr.ocr_adapter import TextRecognizer, OCRBackend
from src.extraction.field_extractor import FieldExtractor
from src.extraction.validators.cross_validator import CrossValidator
from src.models.cedula_colombiana import CedulaColombiana, TipoDocumento


class CedulaReaderPipeline:
    """
    Pipeline completo para leer y extraer datos de cédulas colombianas.

    Uso:
        pipeline = CedulaReaderPipeline()
        cedula = pipeline.procesar_cedula("ruta/cédula.jpg")
        print(cedula.to_json())
    """

    def __init__(
        self,
        ocr_backend: OCRBackend = OCRBackend.AUTO,
        use_gpu: bool = False,
        confidence_threshold: float = 0.5,
        use_tf_region_detector: bool = False,
        tf_region_model_path: Optional[str] = None,
    ):
        """
        Args:
            ocr_backend: Motor de OCR (AUTO, EASYOCR, PADDLEOCR, CRNN).
            use_gpu: Usar GPU para OCR si está disponible.
            confidence_threshold: Umbral mínimo de confianza global (0-1).
            use_tf_region_detector: Usar TensorFlow para detección de regiones.
            tf_region_model_path: Ruta al modelo TF de regiones.
        """
        # ── Módulos del pipeline ───────────────────────────────
        self.enhancer = ImageEnhancer()
        self.perspective_corrector = PerspectiveCorrector()
        self.binarizer = Binarizer()
        self.barcode_decoder = BarcodeDecoder()
        self.barcode_parser = BarcodeParser()
        self.region_detector = RegionDetector(
            use_tensorflow=use_tf_region_detector,
            model_path=tf_region_model_path,
        )
        self.text_recognizer = TextRecognizer(
            backend=ocr_backend,
            use_gpu=use_gpu,
        )
        self.field_extractor = FieldExtractor()
        self.cross_validator = CrossValidator()

        # ── Configuración ──────────────────────────────────────
        self.confidence_threshold = confidence_threshold

    def procesar_cedula(
        self,
        imagen: Union[str, Path, np.ndarray, bytes],
    ) -> CedulaColombiana:
        """
        Procesa una cédula colombiana y extrae todos sus datos.

        Args:
            imagen: Puede ser:
                - Ruta a archivo (str o Path).
                - Array numpy (imagen ya cargada en memoria).
                - Bytes de imagen (JPEG/PNG en memoria).

        Returns:
            CedulaColombiana con los datos extraídos.

        Raises:
            FileNotFoundError: Si la ruta no existe.
            ValueError: Si el formato no es soportado o la calidad es baja.
        """
        t_start = time.time()

        # ── 0. Cargar imagen ───────────────────────────────────
        image = self._cargar_imagen(imagen)

        # ── 1. Preprocesamiento ────────────────────────────────
        # 1a. Corrección de perspectiva
        warped, corrected, _ = self.perspective_corrector.correct(image)
        if corrected:
            image = warped
            print("[Pipeline] Perspectiva corregida.")

        # 1b. Mejora de imagen (CLAHE + denoising + sharpen)
        enhanced = self.enhancer.enhance(image)

        # 1c. Mantener copia a color para detección de tipo
        enhanced_color = image  # imagen corregida pero sin pasar a gris

        # ── 2. Decodificar PDF417 ──────────────────────────────
        print("[Pipeline] Decodificando PDF417...")
        pdf417_data = self.barcode_decoder.decode_first(image)
        datos_pdf417 = {}
        if pdf417_data:
            datos_pdf417 = self.barcode_parser.parse(pdf417_data)
            print(f"[Pipeline] PDF417 decodificado: {len(datos_pdf417)} campos.")
        else:
            print("[Pipeline] PDF417 no detectado o ilegible. Continuando solo con OCR.")

        # ── 3. Detectar tipo de documento y regiones ────────────
        print("[Pipeline] Detectando tipo de documento y regiones...")
        tipo = self.region_detector.detect_document_type(enhanced_color)
        print(f"[Pipeline] Tipo detectado: {tipo.value}")

        coords = self.region_detector.detect_regions(image, tipo)

        # ── 4. OCR sobre cada región ────────────────────────────
        print("[Pipeline] Ejecutando OCR sobre regiones...")
        ocr_results = {}
        for region_name, (x, y, w, h) in coords.items():
            # Extraer región
            roi = enhanced[y:y + h, x:x + w]

            # OCR
            texto, confianza = self.text_recognizer.recognize(roi)

            ocr_results[str(region_name)] = (texto, confianza)

            if texto:
                print(f"  [{region_name}]: '{texto}' (conf={confianza:.2f})")

        # ── 5. Extraer campos ───────────────────────────────────
        print("[Pipeline] Extrayendo campos...")
        cedula = self.field_extractor.extract(ocr_results, tipo)

        # ── 6. Validación cruzada OCR vs PDF417 ─────────────────
        if datos_pdf417:
            print("[Pipeline] Validando OCR vs PDF417...")
            self.cross_validator.apply_validation(cedula, datos_pdf417)

            if cedula.tiene_discrepancias:
                print(f"[Pipeline] ⚠ {len(cedula.discrepancias_pdf417)} discrepancias:")
                for d in cedula.discrepancias_pdf417:
                    print(f"    - {d}")
            else:
                print("[Pipeline] ✅ OCR y PDF417 coinciden.")

        # ── 7. Metadata ─────────────────────────────────────────
        cedula.processing_time_ms = int((time.time() - t_start) * 1000)

        # ── 8. Verificar calidad ────────────────────────────────
        if cedula.confidence_score < self.confidence_threshold:
            print(
                f"[Pipeline] ⚠ Confianza baja ({cedula.confidence_score:.2f}). "
                f"Umbral: {self.confidence_threshold}"
            )

        print(
            f"[Pipeline] ✅ Procesamiento completado en "
            f"{cedula.processing_time_ms}ms. "
            f"Confianza: {cedula.confidence_score:.2f}"
        )

        return cedula

    def procesar_lote(
        self,
        imagenes: list,
    ) -> list:
        """
        Procesa un lote de cédulas.

        Args:
            imagenes: Lista de rutas, arrays numpy, o bytes.

        Returns:
            Lista de CedulaColombiana.
        """
        resultados = []
        for i, img in enumerate(imagenes):
            print(f"\n── Procesando {i + 1}/{len(imagenes)} ──")
            try:
                cedula = self.procesar_cedula(img)
                resultados.append(cedula)
            except Exception as e:
                print(f"[Pipeline] ERROR en imagen {i + 1}: {e}")
                # Crear objeto vacío con error
                cedula = CedulaColombiana()
                cedula.raw_text = f"ERROR: {str(e)}"
                resultados.append(cedula)

        return resultados

    def _cargar_imagen(
        self,
        imagen: Union[str, Path, np.ndarray, bytes],
    ) -> np.ndarray:
        """
        Carga una imagen desde diferentes fuentes.

        Returns:
            Array numpy BGR.
        """
        if isinstance(imagen, np.ndarray):
            return imagen

        if isinstance(imagen, (str, Path)):
            path = Path(imagen)
            if not path.exists():
                raise FileNotFoundError(f"No se encontró la imagen: {path}")

            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"No se pudo leer la imagen: {path} (formato no soportado)")
            return img

        if isinstance(imagen, bytes):
            nparr = np.frombuffer(imagen, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("No se pudo decodificar la imagen desde bytes")
            return img

        raise ValueError(f"Tipo de imagen no soportado: {type(imagen)}")
