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

        # ── 0b. Detectar lado por orientación original ─────────
        # Regla: vertical (h > w) = anverso, horizontal (w >= h) = reverso
        h_img, w_img = image.shape[:2]
        es_reverso = w_img >= h_img
        print(f"[Pipeline] Lado detectado por orientación: {'REVERSO' if es_reverso else 'ANVERSO'}")

        # ── 0c. Auto-rotar si está vertical ────────────────────
        # Anverso: la foto se toma en vertical pero el texto corre
        # horizontalmente -> ROTATE_90_COUNTERCLOCKWISE alinea el texto.
        if h_img > w_img:
            print("[Pipeline] Imagen vertical -> rotando 90 deg CCW")
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # ── 1. Preprocesamiento ────────────────────────────────
        # 1a. Corrección de perspectiva (solo anverso)
        if not es_reverso:
            warped, corrected, _ = self.perspective_corrector.correct(image)
            if corrected:
                image = warped
                print("[Pipeline] Perspectiva corregida.")

        # 1b. Mejora de imagen (diferente para anverso vs reverso)
        if es_reverso:
            enhanced = self._enhance_reverso(image)
        else:
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

        # ── 3. Detectar tipo de documento ─────────────────────
        print("[Pipeline] Detectando tipo de documento...")
        tipo = self.region_detector.detect_document_type(image)
        print(f"[Pipeline] Tipo detectado: {tipo.value}")

        # ── 4. OCR en página completa ─────────────────────────
        print("[Pipeline] Ejecutando OCR en página completa...")
        # Convertir enhanced (grayscale) a BGR para EasyOCR
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        h_img, w_img = enhanced_bgr.shape[:2]

        all_blocks = self.text_recognizer.recognize_full_page(enhanced_bgr)
        print(f"[Pipeline] Bloques OCR detectados: {len(all_blocks)}")

        # Mostrar todos los bloques encontrados (coordenadas originales)
        for b in all_blocks:
            print(f"  ({b['x']:4d},{b['y']:4d}) {b['w']:4d}x{b['h']:4d} "
                  f"| conf={b['conf']:.2f} | '{b['text']}'")

        # ── 5. Clasificar por posicion ────────────────────────
        # Probar los 4 clasificadores y elegir el que devuelva mas campos.
        # No hay deteccion previa de formato: cada clasificador sabe a que
        # zonas prestar atencion.
        from src.extraction.position_classifier import (
            classify_blocks, classify_blocks_reverso,
            classify_blocks_nueva, classify_blocks_reverso_nueva,
        )

        candidates = [
            ("ANVERSO CLASICO", classify_blocks(all_blocks, w_img, h_img), False),
            ("REVERSO CLASICO", classify_blocks_reverso(all_blocks, w_img, h_img), True),
            ("ANVERSO NUEVO", classify_blocks_nueva(all_blocks, w_img, h_img), False),
            ("REVERSO NUEVO", classify_blocks_reverso_nueva(all_blocks, w_img, h_img), True),
        ]

        # Contar campos no vacios (mrz_raw del reverso nuevo cuenta como +3)
        def score(res):
            s = sum(1 for v in res.values() if v)
            if res.get("mrz_raw"):
                s += 3
            return s

        best_name, campos_por_posicion, es_reverso = max(candidates, key=lambda c: score(c[1]))

        print(f"[Pipeline] Clasificador elegido: {best_name} "
              f"({score(campos_por_posicion)} campos)")

        print("[Pipeline] Campos clasificados por posición:")
        for campo, valor in campos_por_posicion.items():
            if valor:
                print(f"  [{campo}]: '{valor}'")

        # ── 6. Construir CedulaColombiana ─────────────────────
        cedula = CedulaColombiana()
        cedula.tipo_documento = tipo

        # Usar el field_extractor para limpiar cada campo
        from src.extraction.regex_patterns import RegexPatterns
        patterns = RegexPatterns()

        raw_parts = []

        # ── 5b. Fallback: parsear texto humano del PDF417 ─────
        if not pdf417_data and es_reverso:
            # Buscar en los bloques OCR el texto legible del código de barras
            barcode_texts = [
                b["text"] for b in all_blocks
                if b["y"] > h_img * 0.85 and any(c.isdigit() for c in b["text"])
            ]
            if barcode_texts:
                barcode_line = " ".join(barcode_texts)
                print(f"[Pipeline] Texto humano del PDF417 detectado: '{barcode_line}'")
                datos_pdf417 = self.barcode_parser.parse_text_line(barcode_line)
                if datos_pdf417:
                    print(f"[Pipeline] Datos extraídos del texto: {datos_pdf417}")

        # Numero de cedula
        if campos_por_posicion["numero_cedula"]:
            num = patterns.extract_cedula(campos_por_posicion["numero_cedula"])
            cedula.numero_cedula = num
            raw_parts.append(f"NUMERO: {campos_por_posicion['numero_cedula']}")
        elif "numero_cedula" in datos_pdf417:
            cedula.numero_cedula = datos_pdf417["numero_cedula"]
            raw_parts.append(f"NUMERO (PDF417): {datos_pdf417['numero_cedula']}")

        # Apellidos y nombres
        if campos_por_posicion["apellidos_nombres"]:
            p_ap, s_ap, noms = patterns.split_nombre_completo(
                campos_por_posicion["apellidos_nombres"]
            )
            cedula.primer_apellido = p_ap
            cedula.segundo_apellido = s_ap
            cedula.nombres = noms
            raw_parts.append(f"NOMBRE: {campos_por_posicion['apellidos_nombres']}")

        # Fecha de nacimiento
        if campos_por_posicion["fecha_nacimiento"]:
            fecha = patterns.extract_fecha(campos_por_posicion["fecha_nacimiento"])
            if fecha:
                cedula.fecha_nacimiento = patterns.normalize_fecha_to_iso(*fecha)
            raw_parts.append(f"FECHA_NAC: {campos_por_posicion['fecha_nacimiento']}")

        # Lugar de nacimiento
        if campos_por_posicion["lugar_nacimiento"]:
            cedula.lugar_nacimiento = patterns.extract_lugar(
                campos_por_posicion["lugar_nacimiento"]
            )
            raw_parts.append(f"LUGAR: {campos_por_posicion['lugar_nacimiento']}")

        # Estatura
        if campos_por_posicion["estatura"]:
            cedula.estatura = patterns.extract_estatura(
                campos_por_posicion["estatura"]
            )
            raw_parts.append(f"ESTATURA: {campos_por_posicion['estatura']}")

        # Grupo sanguíneo
        if campos_por_posicion["grupo_sanguineo"]:
            cedula.grupo_sanguineo = patterns.extract_grupo_sanguineo(
                campos_por_posicion["grupo_sanguineo"]
            )
            raw_parts.append(f"GS: {campos_por_posicion['grupo_sanguineo']}")

        # Sexo
        if campos_por_posicion["sexo"]:
            cedula.sexo = patterns.extract_sexo(campos_por_posicion["sexo"])
            raw_parts.append(f"SEXO: {campos_por_posicion['sexo']}")

        # Fecha de expedición
        if campos_por_posicion["fecha_expedicion"]:
            fecha = patterns.extract_fecha(campos_por_posicion["fecha_expedicion"])
            if fecha:
                cedula.fecha_expedicion = patterns.normalize_fecha_to_iso(*fecha)
            raw_parts.append(f"FECHA_EXP: {campos_por_posicion['fecha_expedicion']}")

        cedula.raw_text = " | ".join(raw_parts) if raw_parts else ""

        # ── 6b. MRZ (cédula digital nueva) ──────────────────
        if campos_por_posicion.get("mrz_raw"):
            cedula.raw_text += f" || MRZ: {campos_por_posicion['mrz_raw']}"

        # ── 7. Validación cruzada OCR vs PDF417 ─────────────
        if datos_pdf417:
            print("[Pipeline] Validando OCR vs PDF417...")
            self.cross_validator.apply_validation(cedula, datos_pdf417)

            if cedula.tiene_discrepancias:
                print(f"[Pipeline] [!] {len(cedula.discrepancias_pdf417)} discrepancias:")
                for d in cedula.discrepancias_pdf417:
                    print(f"    - {d}")
            else:
                print("[Pipeline] [OK] OCR y PDF417 coinciden.")

        # ── 8. Calcular confianza ───────────────────────────
        campos_encontrados = sum(1 for v in [
            cedula.numero_cedula, cedula.primer_apellido, cedula.nombres,
            cedula.fecha_nacimiento, cedula.lugar_nacimiento,
        ] if v)
        cedula.confidence_score = min(campos_encontrados / 5.0, 1.0)

        # ── 9. Metadata ─────────────────────────────────────
        cedula.processing_time_ms = int((time.time() - t_start) * 1000)

        if cedula.confidence_score < self.confidence_threshold:
            print(
                f"[Pipeline] [!] Confianza baja ({cedula.confidence_score:.2f}). "
                f"Umbral: {self.confidence_threshold}"
            )

        print(
            f"[Pipeline] [OK] Procesamiento completado en "
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

    def _detectar_reverso(self, image: np.ndarray) -> bool:
        """
        Detecta si la imagen es el reverso de la cédula.
        Heurística: el reverso tiene fondo claro uniforme, sin foto.
        El anverso tiene una foto (región oscura grande) en un lado.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # El reverso típicamente tiene fondo más claro y uniforme
        mean_val = gray.mean()

        # Detectar presencia de foto (región oscura grande) = anverso
        # La foto ocupa ~25-35% del área y tiene pixeles oscuros (< 80)
        dark_mask = gray < 80
        dark_ratio = dark_mask.sum() / (h * w)

        # Si hay una región oscura significativa (> 8% del área), es anverso (tiene foto)
        if dark_ratio > 0.08:
            return False

        # Buscar región densa de bordes en la mitad inferior (PDF417)
        lower_half = gray[h // 2:, :]
        edges = cv2.Canny(lower_half, 50, 150)
        edge_density = edges.mean()

        # Heurística combinada: fondo claro + densidad de bordes alta abajo
        # + sin región oscura grande (foto del anverso)
        if mean_val > 110 and edge_density > 5:
            return True

        return False

    def _enhance_reverso(self, image: np.ndarray) -> np.ndarray:
        """
        El reverso no necesita preprocesamiento.
        La imagen cruda ya tiene texto negro sobre fondo claro con buen contraste.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray

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
