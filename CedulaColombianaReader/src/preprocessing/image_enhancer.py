"""
Mejorador de imagen para cédulas colombianas.

Aplica CLAHE, reducción de ruido y mejora de contraste para
optimizar la lectura OCR sobre el fondo amarillo o azul de la cédula.
"""

import cv2
import numpy as np


class ImageEnhancer:
    """
    Mejora imágenes de cédulas colombianas para OCR.

    Aplica ecualización adaptativa (CLAHE), filtros de nitidez
    y atenuación de elementos de seguridad (hologramas, guilloché).
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple = (8, 8),
        denoise_strength: float = 10.0,
    ):
        """
        Args:
            clip_limit: Límite de contraste para CLAHE.
            tile_grid_size: Tamaño de grilla para CLAHE.
            denoise_strength: Fuerza del filtro de reducción de ruido.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.denoise_strength = denoise_strength
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline completo de mejora de imagen.

        Args:
            image: Imagen BGR o escala de grises.

        Returns:
            Imagen mejorada en escala de grises.
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 1. Reducir ruido preservando bordes (importante para OCR)
        gray = self._denoise(gray)

        # 2. Aplicar CLAHE para mejorar contraste local
        #    Crucial para cédulas amarillas donde el texto negro
        #    puede tener poco contraste con el fondo
        gray = self._clahe.apply(gray)

        # 3. Nitidez suave para resaltar caracteres
        gray = self._sharpen(gray)

        return gray

    def enhance_color(self, image: np.ndarray) -> np.ndarray:
        """
        Versión que preserva color. Útil para clasificar tipo de documento
        (amarillo CC vs azul TI/CE) antes del OCR.
        """
        # Convertir a LAB para aplicar CLAHE solo al canal L (luminosidad)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _denoise(self, gray: np.ndarray) -> np.ndarray:
        """Reducción de ruido preservando bordes."""
        return cv2.fastNlMeansDenoising(
            gray, None, self.denoise_strength, 7, 21
        )

    def _sharpen(self, gray: np.ndarray) -> np.ndarray:
        """Nitidez suave con kernel de realce."""
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(gray, -1, kernel)
