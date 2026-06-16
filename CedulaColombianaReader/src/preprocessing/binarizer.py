"""
Binarizador adaptativo para cédulas colombianas.

Convierte imágenes en escala de grises a blanco y negro puro,
optimizado para tinta negra sobre fondo claro (amarillo o azul).
"""

import cv2
import numpy as np
from enum import Enum


class BinarizationMethod(str, Enum):
    OTSU = "otsu"
    SAUVOLA = "sauvola"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"
    ADAPTIVE_MEAN = "adaptive_mean"


class Binarizer:
    """
    Binariza imágenes de cédulas usando métodos adaptativos.

    Cada método tiene ventajas según la calidad de la imagen:
    - OTSU: rápido, bueno para imágenes con buena iluminación.
    - Sauvola: robusto ante iluminación no uniforme.
    - Adaptive: bueno para texto fino sobre fondos degradados.
    """

    def __init__(
        self,
        method: BinarizationMethod = BinarizationMethod.ADAPTIVE_GAUSSIAN,
        block_size: int = 31,
        c_constant: int = 7,
    ):
        """
        Args:
            method: Método de binarización.
            block_size: Tamaño del bloque (debe ser impar).
            c_constant: Constante de ajuste de umbral.
        """
        self.method = method
        self.block_size = block_size if block_size % 2 == 1 else block_size + 1
        self.c_constant = c_constant

    def binarize(self, gray: np.ndarray) -> np.ndarray:
        """
        Aplica binarización a una imagen en escala de grises.

        Args:
            gray: Imagen en escala de grises (uint8).

        Returns:
            Imagen binarizada (0 = negro/texto, 255 = blanco/fondo).
        """
        if self.method == BinarizationMethod.OTSU:
            return self._otsu(gray)
        elif self.method == BinarizationMethod.SAUVOLA:
            return self._sauvola(gray)
        elif self.method == BinarizationMethod.ADAPTIVE_GAUSSIAN:
            return cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                self.block_size,
                self.c_constant,
            )
        elif self.method == BinarizationMethod.ADAPTIVE_MEAN:
            return cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV,
                self.block_size,
                self.c_constant,
            )
        else:
            raise ValueError(f"Método no soportado: {self.method}")

    def _otsu(self, gray: np.ndarray) -> np.ndarray:
        """Binarización global de Otsu."""
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return binary

    def _sauvola(self, gray: np.ndarray) -> np.ndarray:
        """
        Implementación del algoritmo de Sauvola.

        Umbral local: T = mean * (1 + k * (stddev / R - 1))
        donde R = 128 (rango dinámico de grises), k = 0.2.
        """
        k = 0.2
        R = 128.0

        # Media y desviación estándar locales
        mean = cv2.boxFilter(
            gray.astype(np.float32), -1, (self.block_size, self.block_size),
            normalize=True
        )
        sq_mean = cv2.boxFilter(
            (gray.astype(np.float32) ** 2), -1,
            (self.block_size, self.block_size), normalize=True
        )
        stddev = np.sqrt(np.maximum(sq_mean - mean ** 2, 0))

        # Umbral de Sauvola
        threshold = mean * (1.0 + k * (stddev / R - 1.0))

        binary = np.where(gray > threshold, 0, 255).astype(np.uint8)
        return binary
