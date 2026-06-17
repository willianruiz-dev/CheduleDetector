"""
Adaptador OCR multi-backend para cédulas colombianas.

Soporta:
- CRNN personalizado (TensorFlow/Keras).
- EasyOCR (pre-entrenado, buen soporte para español).
- PaddleOCR (alta precisión, soporte multilingüe).

Selecciona automáticamente el mejor backend disponible.
"""

import numpy as np
from typing import Optional, List, Tuple
from enum import Enum


class OCRBackend(str, Enum):
    CRNN = "crnn"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    AUTO = "auto"


class TextRecognizer:
    """
    Reconocedor de texto multi-backend optimizado para cédulas colombianas.

    Prioridad de backends (AUTO):
    1. PaddleOCR (mayor precisión).
    2. EasyOCR (buen balance velocidad/precisión).
    3. CRNN personalizado (modelo propio entrenado).
    """

    def __init__(
        self,
        backend: OCRBackend = OCRBackend.AUTO,
        crnn_model_path: Optional[str] = None,
        use_gpu: bool = False,
    ):
        """
        Args:
            backend: Motor de OCR a usar.
            crnn_model_path: Ruta al modelo CRNN (.h5) si se usa ese backend.
            use_gpu: Usar GPU si está disponible.
        """
        self.backend = backend
        self.use_gpu = use_gpu

        # Inicializar backends según disponibilidad
        self._crnn = None
        self._easyocr_reader = None
        self._paddleocr_reader = None

        self._init_backends(crnn_model_path)

    def recognize(
        self, image: np.ndarray
    ) -> Tuple[str, float]:
        """
        Reconoce texto en una imagen de región de cédula.

        Args:
            image: Imagen BGR o escala de grises de una región.

        Returns:
            Tuple (texto_reconocido, confianza 0-1).
        """
        if self._paddleocr_reader is not None:
            return self._recognize_paddleocr(image)
        elif self._easyocr_reader is not None:
            return self._recognize_easyocr(image)
        elif self._crnn is not None:
            return self._crnn.predict(image)
        else:
            return "", 0.0

    def recognize_full_page(
        self, image: np.ndarray
    ) -> list:
        """
        Reconoce TODO el texto en una imagen completa, devolviendo
        cada bloque con su posición y confianza.

        Args:
            image: Imagen BGR completa.

        Returns:
            Lista de dicts: [{"text": str, "conf": float, "x": int, "y": int, "w": int, "h": int}, ...]
        """
        if self._easyocr_reader is not None:
            return self._recognize_full_easyocr(image)
        elif self._paddleocr_reader is not None:
            return self._recognize_full_paddleocr(image)
        else:
            return []

    def _recognize_full_easyocr(self, image: np.ndarray) -> list:
        """EasyOCR en página completa con posiciones."""
        results = self._easyocr_reader.readtext(image)
        blocks = []
        for bbox, text, conf in results:
            x = int(bbox[0][0])
            y = int(bbox[0][1])
            w = int(bbox[2][0] - bbox[0][0])
            h = int(bbox[2][1] - bbox[0][1])
            blocks.append({"text": text, "conf": float(conf), "x": x, "y": y, "w": w, "h": h})
        return blocks

    def _recognize_full_paddleocr(self, image: np.ndarray) -> list:
        """PaddleOCR en página completa con posiciones."""
        try:
            results = self._paddleocr_reader.ocr(image, cls=True)
        except (TypeError, ValueError):
            results = self._paddleocr_reader.ocr(image)

        if not results or not results[0]:
            return []

        blocks = []
        for line in results[0]:
            if isinstance(line, dict):
                text = line.get("text", "")
                conf = line.get("confidence", 0.0)
                bbox = line.get("bbox", [[0,0],[0,0],[0,0],[0,0]])
                x, y = int(bbox[0][0]), int(bbox[0][1])
                w = int(bbox[2][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])
            elif isinstance(line, (list, tuple)) and len(line) >= 2:
                bbox = line[0]
                x, y = int(bbox[0][0]), int(bbox[0][1])
                w = int(bbox[2][0] - bbox[0][0])
                h = int(bbox[2][1] - bbox[0][1])
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2 else 0.0
            else:
                continue
            blocks.append({"text": text, "conf": float(conf), "x": x, "y": y, "w": w, "h": h})
        return blocks

    def recognize_regions(
        self, regions: dict
    ) -> dict:
        """
        Reconoce texto en múltiples regiones.

        Args:
            regions: Diccionario {nombre_region: imagen_numpy}.

        Returns:
            Diccionario {nombre_region: (texto, confianza)}.
        """
        results = {}
        for name, img in regions.items():
            if img is not None and img.size > 0:
                text, conf = self.recognize(img)
                results[str(name)] = (text, conf)
            else:
                results[str(name)] = ("", 0.0)
        return results

    def _init_backends(self, crnn_model_path: Optional[str] = None) -> None:
        """Inicializa los backends disponibles en orden de preferencia."""
        if self.backend == OCRBackend.AUTO:
            # Probar en orden de preferencia
            if self._try_paddleocr():
                return
            if self._try_easyocr():
                return
            if crnn_model_path:
                self._try_crnn(crnn_model_path)
        elif self.backend == OCRBackend.PADDLEOCR:
            self._try_paddleocr()
        elif self.backend == OCRBackend.EASYOCR:
            self._try_easyocr()
        elif self.backend == OCRBackend.CRNN:
            if crnn_model_path:
                self._try_crnn(crnn_model_path)

    def _try_paddleocr(self) -> bool:
        """Intenta inicializar PaddleOCR."""
        try:
            from paddleocr import PaddleOCR
            # Soporte tanto para PaddleOCR 2.x como 3.x
            try:
                self._paddleocr_reader = PaddleOCR(
                    use_angle_cls=True,
                    lang="es",
                    use_gpu=self.use_gpu,
                    show_log=False,
                )
            except (TypeError, ValueError):
                self._paddleocr_reader = PaddleOCR(lang="es")
            print("[OCR] Backend: PaddleOCR (español)")
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"[OCR] PaddleOCR no disponible: {e}")
            return False

    def _try_easyocr(self) -> bool:
        """Intenta inicializar EasyOCR."""
        try:
            import easyocr
            self._easyocr_reader = easyocr.Reader(
                ["es"],
                gpu=self.use_gpu,
                verbose=False,
            )
            print("[OCR] Backend: EasyOCR (español)")
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"[OCR] EasyOCR no disponible: {e}")
            return False

    def _try_crnn(self, model_path: str) -> bool:
        """Intenta inicializar CRNN personalizado."""
        try:
            from src.ocr.text_recognizer import CRNNModel
            self._crnn = CRNNModel(model_path=model_path)
            print(f"[OCR] Backend: CRNN personalizado ({model_path})")
            return True
        except Exception as e:
            print(f"[OCR] CRNN no disponible: {e}")
            return False

    def _recognize_easyocr(
        self, image: np.ndarray
    ) -> Tuple[str, float]:
        """Reconoce texto con EasyOCR."""
        results = self._easyocr_reader.readtext(image)
        if not results:
            return "", 0.0

        # Concatenar todos los textos detectados
        texts = []
        confidences = []
        for bbox, text, conf in results:
            texts.append(text)
            confidences.append(conf)

        full_text = " ".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return full_text, avg_confidence

    def _recognize_paddleocr(
        self, image: np.ndarray
    ) -> Tuple[str, float]:
        """Reconoce texto con PaddleOCR."""
        try:
            results = self._paddleocr_reader.ocr(image, cls=True)
        except (TypeError, ValueError):
            results = self._paddleocr_reader.ocr(image)

        if not results or not results[0]:
            return "", 0.0

        texts = []
        confidences = []
        for line in results[0]:
            # PaddleOCR 2.x: line = [bbox, (text, confidence)]
            # PaddleOCR 3.x: line = {"text": ..., "confidence": ...} or similar
            if isinstance(line, dict):
                text = line.get("text", "")
                conf = line.get("confidence", 0.0)
            elif isinstance(line, (list, tuple)) and len(line) >= 2:
                text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                conf = line[1][1] if isinstance(line[1], (list, tuple)) and len(line[1]) >= 2 else 0.0
            else:
                continue
            texts.append(text)
            confidences.append(float(conf))

        full_text = " ".join(texts)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return full_text, avg_confidence

    @property
    def active_backend(self) -> str:
        """Retorna el nombre del backend activo."""
        if self._paddleocr_reader is not None:
            return "paddleocr"
        elif self._easyocr_reader is not None:
            return "easyocr"
        elif self._crnn is not None:
            return "crnn"
        return "ninguno"

    @property
    def is_ready(self) -> bool:
        """True si hay al menos un backend disponible."""
        return self.active_backend != "ninguno"
