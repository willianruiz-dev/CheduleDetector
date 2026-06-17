"""
Detector de regiones de interés (ROI) para cédulas colombianas.

Detecta zonas específicas del layout colombiano usando heurísticas
de posición y color, y opcionalmente un modelo de TensorFlow.
Determina además el tipo de documento (CC, CE, TI).
"""

import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List
from enum import Enum

from src.models.cedula_colombiana import TipoDocumento


class RegionName(str, Enum):
    """Nombres de las regiones de interés en la cédula colombiana."""
    FOTO = "foto"
    NUMERO_CEDULA = "numero_cedula"
    APELLIDOS_NOMBRES = "apellidos_nombres"
    FECHA_NACIMIENTO = "fecha_nacimiento"
    LUGAR_NACIMIENTO = "lugar_nacimiento"
    ESTATURA = "estatura"
    GRUPO_SANGUINEO = "grupo_sanguineo"
    SEXO = "sexo"
    FECHA_EXPEDICION = "fecha_expedicion"
    FIRMA = "firma"
    HUELLA = "huella"
    PDF417 = "pdf417"


class RegionDetector:
    """
    Detecta regiones de interés en cédulas colombianas.

    Usa heurísticas basadas en el layout conocido:
    - Foto: lado izquierdo (~30-40% del ancho).
    - Datos: lado derecho (~60-70% del ancho).
    - PDF417: franja inferior (~15% de la altura).
    """

    def __init__(self, use_tensorflow: bool = False, model_path: Optional[str] = None):
        self.use_tensorflow = use_tensorflow
        self.model_path = model_path
        self._tf_model = None

        if use_tensorflow and model_path:
            self._load_model(model_path)

    def detect_document_type(self, image: np.ndarray) -> TipoDocumento:
        """
        Detecta el tipo de documento basado en características visuales.

        Heurísticas:
        - CC amarilla: fondo amarillo cálido predominante.
        - CC azul: fondo azul oscuro predominante.
        - CE: similar a CC pero con tonos diferentes.
        - TI: formato azul con rosa, tamaño menor.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Calcular porcentaje de píxeles en rangos de color
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # Amarillo: H entre 20 y 40
        amarillo_mask = cv2.inRange(hsv, (20, 50, 50), (40, 255, 255))
        pct_amarillo = np.sum(amarillo_mask > 0) / total_pixels

        # Azul: H entre 90 y 130
        azul_mask = cv2.inRange(hsv, (90, 50, 50), (130, 255, 255))
        pct_azul = np.sum(azul_mask > 0) / total_pixels

        # Rosa/magenta: H entre 140 y 170
        rosa_mask = cv2.inRange(hsv, (140, 50, 50), (170, 255, 255))
        pct_rosa = np.sum(rosa_mask > 0) / total_pixels

        if pct_rosa > 0.15 and pct_azul > 0.10:
            return TipoDocumento.TI
        elif pct_azul > 0.20:
            return TipoDocumento.CE
        elif pct_amarillo > 0.15:
            return TipoDocumento.CC
        else:
            # Fallback: asumir CC (más común)
            return TipoDocumento.CC

    def detect_regions(
        self, image: np.ndarray, tipo: TipoDocumento
    ) -> Dict[RegionName, Tuple[int, int, int, int]]:
        """
        Detecta todas las regiones de interés.

        Primero intenta encontrar el documento dentro de la imagen
        (útil cuando la foto no es solo la cédula sino el entorno).
        Luego aplica heurísticas de layout relativas al documento.

        Args:
            image: Imagen BGR ya corregida de perspectiva.
            tipo: Tipo de documento detectado.

        Returns:
            Diccionario {RegionName: (x, y, w, h)} con las coordenadas
            de cada región detectada.
        """
        if self.use_tensorflow and self._tf_model is not None:
            return self._detect_with_tf(image)

        # Intentar encontrar el documento dentro de la imagen
        doc_rect = self._find_document(image)
        if doc_rect is not None:
            dx, dy, dw, dh = doc_rect
            # Usar las coordenadas del documento como referencia
            return self._detect_heuristic_relative(dx, dy, dw, dh, tipo)

        h, w = image.shape[:2]
        return self._detect_heuristic(h, w, tipo)

    def _find_document(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Intenta encontrar el rectángulo del documento en la imagen.

        Usa detección de bordes + contornos para aislar la cédula
        del fondo/entorno.

        Returns:
            (x, y, w, h) del documento, o None si no se detecta.
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        best_rect = None
        best_area = 0

        # Probar varios umbrales de Canny para mejor detección
        for low_t in [30, 50, 80]:
            edges = cv2.Canny(blurred, low_t, low_t * 3)

            # Dilatar para conectar bordes cercanos
            kernel = np.ones((7, 7), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=3)

            # Encontrar contornos
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            img_area = w * h

            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Debe ocupar entre 15% y 95% de la imagen
                if area < img_area * 0.15 or area > img_area * 0.95:
                    continue

                # Aproximar a polígono
                peri = cv2.arcLength(cnt, True)
                for eps in [0.02, 0.05, 0.08]:
                    approx = cv2.approxPolyDP(cnt, eps * peri, True)

                    # Debe tener 4 esquinas (rectangular)
                    if len(approx) >= 4 and area > best_area:
                        # Obtener bounding box
                        x, y, rw, rh = cv2.boundingRect(approx)

                        # Verificar proporción razonable (1.2-2.0 para cédula)
                        ratio = rw / rh if rh > 0 else 0
                        if 1.2 <= ratio <= 2.2:
                            best_area = area
                            best_rect = (x, y, rw, rh)

            if best_rect is not None:
                break

        return best_rect

    def extract_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Extrae un recorte de la imagen para una región dada.

        Args:
            image: Imagen BGR.
            region: (x, y, w, h).

        Returns:
            Recorte de la región.
        """
        x, y, w, h = region
        return image[y:y + h, x:x + w].copy()

    def extract_all_regions(
        self, image: np.ndarray
    ) -> Tuple[TipoDocumento, Dict[RegionName, np.ndarray]]:
        """
        Pipeline completo: detecta tipo y extrae todas las regiones.

        Returns:
            Tuple con el tipo de documento y un diccionario de recortes.
        """
        tipo = self.detect_document_type(image)
        coords = self.detect_regions(image, tipo)

        regions = {}
        for name, rect in coords.items():
            regions[name] = self.extract_region(image, rect)

        return tipo, regions

    def _detect_heuristic(
        self, h: int, w: int, tipo: TipoDocumento
    ) -> Dict[RegionName, Tuple[int, int, int, int]]:
        """
        Detección por heurísticas de posición basadas en el layout
        estándar de la cédula colombiana.

        Layout aproximado (porcentajes del ancho y alto):
        ┌──────────────────────────────────────────┐
        │  FOTO     │  NÚMERO DE CÉDULA           │
        │ (30%)     │  APELLIDOS Y NOMBRES        │
        │           │  FECHA DE NACIMIENTO        │
        │  HUELLA   │  LUGAR DE NACIMIENTO        │
        │           │  ESTATURA  │ GS  │ SEXO     │
        │           │  FECHA DE EXPEDICIÓN        │
        │           │  FIRMA                       │
        │           │                              │
        │             PDF417 (15% inferior)        │
        └──────────────────────────────────────────┘
        """
        regions = {}

        # ── Foto: 5%-30% ancho, 5%-55% alto ──
        regions[RegionName.FOTO] = (
            int(w * 0.04), int(h * 0.04),
            int(w * 0.26), int(h * 0.52),
        )

        # ── Número de cédula: esquina superior derecha ──
        regions[RegionName.NUMERO_CEDULA] = (
            int(w * 0.32), int(h * 0.02),
            int(w * 0.64), int(h * 0.07),
        )

        # ── Apellidos y nombres: centro-derecha (el texto más grande) ──
        regions[RegionName.APELLIDOS_NOMBRES] = (
            int(w * 0.32), int(h * 0.10),
            int(w * 0.64), int(h * 0.14),
        )

        # ── Fecha de nacimiento ──
        regions[RegionName.FECHA_NACIMIENTO] = (
            int(w * 0.32), int(h * 0.24),
            int(w * 0.30), int(h * 0.06),
        )

        # ── Lugar de nacimiento ──
        regions[RegionName.LUGAR_NACIMIENTO] = (
            int(w * 0.32), int(h * 0.30),
            int(w * 0.64), int(h * 0.06),
        )

        # ── Estatura ──
        regions[RegionName.ESTATURA] = (
            int(w * 0.32), int(h * 0.38),
            int(w * 0.14), int(h * 0.05),
        )

        # ── Grupo sanguíneo ──
        regions[RegionName.GRUPO_SANGUINEO] = (
            int(w * 0.48), int(h * 0.38),
            int(w * 0.10), int(h * 0.05),
        )

        # ── Sexo ──
        regions[RegionName.SEXO] = (
            int(w * 0.62), int(h * 0.38),
            int(w * 0.10), int(h * 0.05),
        )

        # ── Fecha de expedición ──
        regions[RegionName.FECHA_EXPEDICION] = (
            int(w * 0.32), int(h * 0.46),
            int(w * 0.35), int(h * 0.06),
        )

        # ── Firma: inferior derecha ──
        regions[RegionName.FIRMA] = (
            int(w * 0.52), int(h * 0.56),
            int(w * 0.44), int(h * 0.14),
        )

        # ── Huella: inferior izquierda ──
        regions[RegionName.HUELLA] = (
            int(w * 0.02), int(h * 0.58),
            int(w * 0.24), int(h * 0.25),
        )

        # ── PDF417: franja inferior ──
        regions[RegionName.PDF417] = (
            int(w * 0.05), int(h * 0.80),
            int(w * 0.90), int(h * 0.15),
        )

        return regions

    def _detect_heuristic_relative(
        self, dx: int, dy: int, dw: int, dh: int, tipo: TipoDocumento
    ) -> Dict[RegionName, Tuple[int, int, int, int]]:
        """
        Detecta regiones usando heurísticas relativas al rectángulo
        del documento ya encontrado en la imagen.

        Args:
            dx, dy: esquina superior izquierda del documento en la imagen.
            dw, dh: ancho y alto del documento.
            tipo: Tipo de documento.

        Returns:
            Diccionario de regiones con coordenadas absolutas en la imagen.
        """
        rel_regions = self._detect_heuristic(dh, dw, tipo)
        absolute_regions = {}
        for name, (rx, ry, rw, rh) in rel_regions.items():
            absolute_regions[name] = (dx + rx, dy + ry, rw, rh)
        return absolute_regions

    def _detect_with_tf(
        self, image: np.ndarray
    ) -> Dict[RegionName, Tuple[int, int, int, int]]:
        """
        Detección de regiones usando un modelo de TensorFlow.

        NOTA: Requiere un modelo entrenado (SSD, EfficientDet o YOLO)
        exportado a formato SavedModel o .h5.
        """
        if self._tf_model is None:
            return self._detect_heuristic(
                image.shape[0], image.shape[1], TipoDocumento.CC
            )

        try:
            import tensorflow as tf

            # Preprocesar para el modelo
            input_tensor = tf.convert_to_tensor(image)
            input_tensor = tf.image.resize(input_tensor, (640, 640))
            input_tensor = tf.expand_dims(input_tensor, 0)
            input_tensor = tf.cast(input_tensor, tf.uint8)

            # Inferencia
            detections = self._tf_model(input_tensor)

            # Parsear detecciones a diccionario de regiones
            return self._parse_tf_detections(detections, image.shape)

        except Exception as e:
            print(f"[RegionDetector] Error en TF: {e}. Usando heurísticas.")
            return self._detect_heuristic(
                image.shape[0], image.shape[1], TipoDocumento.CC
            )

    def _load_model(self, model_path: str) -> None:
        """Carga un modelo de TensorFlow desde disco."""
        try:
            import tensorflow as tf
            self._tf_model = tf.saved_model.load(model_path)
            print(f"[RegionDetector] Modelo TF cargado: {model_path}")
        except Exception as e:
            print(f"[RegionDetector] Error cargando modelo: {e}")
            self._tf_model = None

    def _parse_tf_detections(
        self, detections: dict, image_shape: Tuple[int, int, int]
    ) -> Dict[RegionName, Tuple[int, int, int, int]]:
        """Convierte las detecciones del modelo TF a coordenadas de regiones."""
        h, w = image_shape[:2]

        # Mapeo de IDs de clase a nombres de región
        id_to_region = {
            1: RegionName.FOTO,
            2: RegionName.NUMERO_CEDULA,
            3: RegionName.APELLIDOS_NOMBRES,
            4: RegionName.FECHA_NACIMIENTO,
            5: RegionName.LUGAR_NACIMIENTO,
            6: RegionName.ESTATURA,
            7: RegionName.GRUPO_SANGUINEO,
            8: RegionName.SEXO,
            9: RegionName.FECHA_EXPEDICION,
            10: RegionName.FIRMA,
            11: RegionName.HUELLA,
            12: RegionName.PDF417,
        }

        regions = {}
        boxes = detections.get("detection_boxes", [])
        classes = detections.get("detection_classes", [])
        scores = detections.get("detection_scores", [])

        for box, cls, score in zip(boxes, classes, scores):
            if score < 0.5:
                continue

            class_id = int(cls)
            if class_id not in id_to_region:
                continue

            # Convertir coordenadas normalizadas a píxeles
            ymin, xmin, ymax, xmax = box
            x = int(xmin * w)
            y = int(ymin * h)
            rw = int((xmax - xmin) * w)
            rh = int((ymax - ymin) * h)

            regions[id_to_region[class_id]] = (x, y, rw, rh)

        return regions
