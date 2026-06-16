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
            self._paddleocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang="es",
                use_gpu=self.use_gpu,
                show_log=False,
            )
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
        results = self._paddleocr_reader.ocr(image, cls=True)

        if not results or not results[0]:
            return "", 0.0

        texts = []
        confidences = []
        for line in results[0]:
            text = line[1][0]
            conf = line[1][1]
            texts.append(text)
            confidences.append(conf)

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
