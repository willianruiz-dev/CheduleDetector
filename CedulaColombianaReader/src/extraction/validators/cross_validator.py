"""
Validador cruzado: OCR vs PDF417 para cédulas colombianas.

Compara los datos extraídos por OCR con los decodificados
del código PDF417 y reporta discrepancias. El PDF417 sirve
como fuente de verdad complementaria (ground truth parcial).
"""

from typing import Dict, Optional, List

from src.models.cedula_colombiana import CedulaColombiana


class CrossValidator:
    """
    Compara resultados OCR contra datos del PDF417.

    Genera alertas cuando hay discrepancias significativas
    entre ambas fuentes de datos.
    """

    # Campos a comparar y sus nombres legibles
    FIELD_LABELS = {
        "numero_cedula": "Número de cédula",
        "primer_apellido": "Primer apellido",
        "segundo_apellido": "Segundo apellido",
        "nombres": "Nombres",
        "fecha_nacimiento": "Fecha de nacimiento",
        "lugar_nacimiento": "Lugar de nacimiento",
        "estatura": "Estatura",
        "grupo_sanguineo": "Grupo sanguíneo",
        "sexo": "Sexo",
        "nacionalidad": "Nacionalidad",
        "fecha_expedicion": "Fecha de expedición",
    }

    # Campos donde una diferencia de 1 carácter es aceptable
    FUZZY_FIELDS = {"nombres", "primer_apellido", "segundo_apellido", "lugar_nacimiento"}

    def __init__(self, max_edit_distance: int = 2):
        """
        Args:
            max_edit_distance: Máxima distancia de Levenshtein permitida
                               para campos fuzzy antes de reportar discrepancia.
        """
        self.max_edit_distance = max_edit_distance

    def validate(
        self,
        cedula: CedulaColombiana,
        datos_pdf417: Dict[str, Optional[str]],
    ) -> List[str]:
        """
        Compara los campos OCR con los del PDF417.

        Args:
            cedula: Objeto CedulaColombiana con datos OCR.
            datos_pdf417: Diccionario con datos del PDF417.

        Returns:
            Lista de strings con las discrepancias encontradas.
            Lista vacía si todo coincide.
        """
        discrepancias = []

        if not datos_pdf417:
            return discrepancias

        # Mapeo de nombres de campo OCR -> nombres en PDF417
        field_map = self._build_field_map()

        for ocr_field, pdf417_keys in field_map.items():
            ocr_value = getattr(cedula, ocr_field, None)
            if ocr_value is None:
                continue

            # Buscar el valor correspondiente en PDF417
            pdf417_value = None
            for key in pdf417_keys:
                if key in datos_pdf417 and datos_pdf417[key]:
                    pdf417_value = datos_pdf417[key]
                    break

            if pdf417_value is None:
                continue

            # Normalizar para comparación
            ocr_norm = self._normalize(ocr_value)
            pdf417_norm = self._normalize(pdf417_value)

            if ocr_norm == pdf417_norm:
                continue  # OK, coinciden exactamente

            # Para campos fuzzy, permitir pequeña diferencia
            if ocr_field in self.FUZZY_FIELDS:
                dist = self._levenshtein_distance(ocr_norm, pdf417_norm)
                if dist <= self.max_edit_distance:
                    continue  # Diferencia aceptable

            # Reportar discrepancia
            label = self.FIELD_LABELS.get(ocr_field, ocr_field)
            discrepancias.append(
                f"{label}: OCR='{ocr_value}' vs PDF417='{pdf417_value}'"
            )

        return discrepancias

    def apply_validation(
        self,
        cedula: CedulaColombiana,
        datos_pdf417: Dict[str, Optional[str]],
    ) -> CedulaColombiana:
        """
        Aplica validación y guarda las discrepancias en el objeto cedula.

        Args:
            cedula: CedulaColombiana a validar (modificado in-place).
            datos_pdf417: Datos del PDF417.

        Returns:
            El mismo objeto cedula con las discrepancias registradas.
        """
        cedula.datos_pdf417 = datos_pdf417
        cedula.discrepancias_pdf417 = self.validate(cedula, datos_pdf417)
        return cedula

    def _build_field_map(self) -> Dict[str, List[str]]:
        """
        Construye el mapeo de campos del modelo a posibles
        claves en el PDF417.
        """
        return {
            "numero_cedula": [
                "numero_cedula", "numero", "cedula", "NUIP",
                "numero_cedula_sin_puntos",
            ],
            "primer_apellido": [
                "primer_apellido", "APELLIDO1", "apellido1",
            ],
            "segundo_apellido": [
                "segundo_apellido", "APELLIDO2", "apellido2",
            ],
            "nombres": [
                "nombres", "nombre", "NOMBRES",
            ],
            "fecha_nacimiento": [
                "fecha_nacimiento", "nacimiento", "FECHA_NACIMIENTO",
            ],
            "lugar_nacimiento": [
                "lugar_nacimiento", "municipio", "LUGAR_NACIMIENTO",
            ],
            "estatura": [
                "estatura", "altura",
            ],
            "grupo_sanguineo": [
                "grupo_sanguineo", "rh", "sangre",
            ],
            "sexo": [
                "sexo", "genero",
            ],
            "nacionalidad": [
                "nacionalidad",
            ],
            "fecha_expedicion": [
                "fecha_expedicion", "expedicion",
            ],
        }

    def _normalize(self, value: str) -> str:
        """Normaliza un string para comparación."""
        if value is None:
            return ""
        return (
            value.upper()
            .strip()
            .replace(".", "")
            .replace(",", "")
            .replace("-", "")
            .replace(" ", "")
        )

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calcula la distancia de Levenshtein entre dos strings.
        (Número mínimo de inserciones, eliminaciones o sustituciones).
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Costo de inserción, eliminación, sustitución
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row

        return prev_row[-1]
