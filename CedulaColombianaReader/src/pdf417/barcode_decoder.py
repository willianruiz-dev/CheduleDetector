"""
Decodificador del código de barras PDF417 de la cédula colombiana.

La cédula colombiana contiene un PDF417 en la parte inferior
que codifica los datos del ciudadano. Este módulo lo decodifica
usando pyzbar (wrapper de ZBar) o, como fallback, una búsqueda
por región de interés + decodificación dedicada.
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple


class BarcodeDecoder:
    """
    Decodifica el código PDF417 de una cédula colombiana.

    Estrategia:
    1. Intentar con pyzbar (rápido, buena tasa de acierto).
    2. Si falla, buscar en el tercio inferior de la imagen (ubicación típica).
    3. Si aún falla, intentar en toda la imagen.
    """

    def __init__(self):
        self._pyzbar_available = self._check_pyzbar()

    def decode(self, image: np.ndarray) -> List[dict]:
        """
        Decodifica todos los códigos PDF417 encontrados en la imagen.

        Args:
            image: Imagen BGR o escala de grises.

        Returns:
            Lista de dicts con 'data' (bytes), 'rect' (x,y,w,h) y 'type'.
        """
        results = []

        # Intentar con pyzbar
        if self._pyzbar_available:
            results = self._decode_with_pyzbar(image)

        # Si no encontró nada, buscar específicamente en el tercio inferior
        if not results:
            results = self._decode_lower_third(image)

        return results

    def decode_first(self, image: np.ndarray) -> Optional[bytes]:
        """
        Retorna los datos del primer PDF417 encontrado, o None.
        """
        results = self.decode(image)
        if results:
            return results[0].get("data")
        return None

    def _check_pyzbar(self) -> bool:
        """Verifica si pyzbar está instalado."""
        try:
            import pyzbar.pyzbar  # noqa: F401
            return True
        except ImportError:
            return False

    def _decode_with_pyzbar(self, image: np.ndarray) -> List[dict]:
        """Decodifica usando pyzbar (ZBar)."""
        import pyzbar.pyzbar as pyzbar

        # pyzbar funciona mejor con escala de grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        decoded = pyzbar.decode(gray)

        results = []
        for obj in decoded:
            # Solo nos interesan PDF417
            if obj.type in ("PDF417", "CODABAR"):
                results.append({
                    "data": obj.data,
                    "rect": (
                        obj.rect.left,
                        obj.rect.top,
                        obj.rect.width,
                        obj.rect.height,
                    ),
                    "type": obj.type,
                })
        return results

    def _decode_lower_third(self, image: np.ndarray) -> List[dict]:
        """
        Busca PDF417 en el tercio inferior de la imagen.

        La cédula colombiana ubica el PDF417 en una franja
        horizontal en la parte baja del documento.
        """
        h, w = image.shape[:2]

        # Recortar tercio inferior
        lower = image[int(h * 0.65):, :]

        # Si pyzbar está disponible, intentar en el recorte
        if self._pyzbar_available:
            results = self._decode_with_pyzbar(lower)
            # Ajustar coordenadas al offset
            for r in results:
                x, y, rw, rh = r["rect"]
                r["rect"] = (x, y + int(h * 0.65), rw, rh)
            return results

        return []

    def detect_barcode_region(
        self, image: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detecta la ubicación aproximada del PDF417 sin decodificar.

        Retorna (x, y, width, height) o None.
        Útil para alimentar al OCR con la región del código.
        """
        # Buscar región densa horizontal en el tercio inferior
        h, w = image.shape[:2]
        lower = image[int(h * 0.65):, :]

        if len(lower.shape) == 3:
            gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)
        else:
            gray = lower

        # El PDF417 tiene alta densidad de bordes horizontales
        edges = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        edges = np.abs(edges).astype(np.uint8)

        # Proyectar horizontalmente para encontrar la franja
        projection = np.mean(edges, axis=1)
        threshold = np.mean(projection) * 1.5

        rows = np.where(projection > threshold)[0]
        if len(rows) == 0:
            return None

        y_start = rows[0] + int(h * 0.65)
        y_end = rows[-1] + int(h * 0.65)

        return (0, y_start, w, y_end - y_start)
