"""
Corrector de perspectiva para documentos de identidad colombianos.

Detecta los 4 vértices del documento mediante Canny + contornos
y aplica una transformación homográfica para enderezar la imagen.
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class PerspectiveCorrector:
    """
    Detecta y corrige la perspectiva de una foto de cédula.

    La cédula colombiana tiene proporción horizontal ~1.6:1
    (ancho ~8.5 cm, alto ~5.4 cm aproximadamente).
    """

    # Proporción estándar de la cédula colombiana (ancho/alto)
    ASPECT_RATIO_CC = 1.6
    OUTPUT_WIDTH = 1280   # ancho de salida en píxeles
    OUTPUT_HEIGHT = int(OUTPUT_WIDTH / ASPECT_RATIO_CC)  # ~800 px

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_contour_area: float = 0.3,  # 30% del área de la imagen
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_area = min_contour_area

    def correct(
        self, image: np.ndarray
    ) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
        """
        Detecta el rectángulo del documento y corrige la perspectiva.

        Args:
            image: Imagen BGR.

        Returns:
            Tuple con:
                - Imagen corregida (o la original si falla la detección).
                - Booleano: True si se detectó y corrigió exitosamente.
                - Matriz de transformación homográfica (3x3) o None.
        """
        h, w = image.shape[:2]
        img_area = h * w

        # 1. Escala de grises + suavizado + Canny
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)

        # 2. Dilatar bordes para cerrar contornos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        # 3. Encontrar contornos
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 4. Buscar el contorno más grande con 4 vértices
        document_contour = None
        max_area = 0
        min_area_threshold = img_area * self.min_contour_area

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(contour)

            if len(approx) == 4 and area > min_area_threshold and area > max_area:
                document_contour = approx
                max_area = area

        if document_contour is None:
            return image, False, None

        # 5. Ordenar vértices (superior-izquierdo, superior-derecho,
        #    inferior-derecho, inferior-izquierdo)
        pts = document_contour.reshape(4, 2)
        rect = self._order_points(pts)

        # 6. Calcular destino y transformar
        dst = np.array([
            [0, 0],
            [self.OUTPUT_WIDTH - 1, 0],
            [self.OUTPUT_WIDTH - 1, self.OUTPUT_HEIGHT - 1],
            [0, self.OUTPUT_HEIGHT - 1],
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect.astype("float32"), dst)
        warped = cv2.warpPerspective(
            image, M, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT)
        )

        return warped, True, M

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Ordena 4 puntos en orden:
        superior-izquierdo, superior-derecho, inferior-derecho, inferior-izquierdo.
        """
        rect = np.zeros((4, 2), dtype="float32")

        # suma mínima = superior-izquierdo, suma máxima = inferior-derecho
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        # diferencia mínima = superior-derecho, diferencia máxima = inferior-izquierdo
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]

        return rect
