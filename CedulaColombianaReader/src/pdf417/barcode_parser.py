"""
Parser de datos del código PDF417 de la cédula colombiana.

El PDF417 de la cédula colombiana contiene los datos del ciudadano
en un formato estructurado definido por la Registraduría Nacional.
Este módulo extrae y mapea esos campos.
"""

import re
from typing import Optional, Dict


class BarcodeParser:
    """
    Parsea los datos crudos del PDF417 de una cédula colombiana.

    El formato típico del PDF417 contiene campos delimitados por
    saltos de línea o caracteres de control, con etiquetas como
    NUMERO=, APELLIDO1=, APELLIDO2=, NOMBRES=, etc.
    """

    # Mapeo de claves conocidas en el PDF417 a campos del modelo
    KEY_MAPPING = {
        # Número de cédula
        "NUMERO": "numero_cedula",
        "CEDULA": "numero_cedula",
        "NUIP": "numero_cedula",
        # Apellidos
        "APELLIDO1": "primer_apellido",
        "PRIMER_APELLIDO": "primer_apellido",
        "APELLIDO2": "segundo_apellido",
        "SEGUNDO_APELLIDO": "segundo_apellido",
        # Nombres
        "NOMBRES": "nombres",
        "NOMBRE": "nombres",
        # Fecha de nacimiento
        "FECHA_NACIMIENTO": "fecha_nacimiento",
        "FECHANACIMIENTO": "fecha_nacimiento",
        "NACIMIENTO": "fecha_nacimiento",
        # Lugar de nacimiento
        "LUGAR_NACIMIENTO": "lugar_nacimiento",
        "LUGARNACIMIENTO": "lugar_nacimiento",
        "MUNICIPIO": "lugar_nacimiento",
        # Estatura
        "ESTATURA": "estatura",
        "ALTURA": "estatura",
        # Grupo sanguíneo
        "GRUPO_SANGUINEO": "grupo_sanguineo",
        "RH": "grupo_sanguineo",
        "SANGRE": "grupo_sanguineo",
        # Sexo
        "SEXO": "sexo",
        "GENERO": "sexo",
        # Fecha de expedición
        "FECHA_EXPEDICION": "fecha_expedicion",
        "EXPEDICION": "fecha_expedicion",
        # Nacionalidad
        "NACIONALIDAD": "nacionalidad",
    }

    def parse(self, raw_data: bytes) -> Dict[str, Optional[str]]:
        """
        Parsea los bytes del PDF417 a un diccionario estructurado.

        Args:
            raw_data: Bytes decodificados del PDF417.

        Returns:
            Diccionario con campos normalizados a los nombres del modelo.
        """
        result: Dict[str, Optional[str]] = {}

        # Decodificar bytes a string (UTF-8 o Latin-1)
        try:
            text = raw_data.decode("utf-8")
        except UnicodeDecodeError:
            text = raw_data.decode("latin-1", errors="replace")

        # Guardar raw para debug
        result["_raw"] = text

        # Intentar parseo por líneas con formato CLAVE=VALOR
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Formato CLAVE=VALOR
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip().upper()
                value = value.strip()

                if key in self.KEY_MAPPING:
                    mapped_key = self.KEY_MAPPING[key]
                    result[mapped_key] = value
                else:
                    # Guardar claves desconocidas
                    result[key.lower()] = value
            else:
                # Intentar parseo por posición (respaldo)
                self._parse_positional(line, result)

        # Normalizar fechas a ISO 8601
        self._normalize_dates(result)

        # Normalizar número de cédula (quitar puntos)
        self._normalize_cedula(result)

        return result

    def _parse_positional(
        self, line: str, result: Dict[str, Optional[str]]
    ) -> None:
        """
        Parseo por posición para formatos antiguos sin etiquetas.
        Intenta extraer datos por patrones conocidos.
        """
        # Patrón: número de cédula con puntos
        cedula_match = re.search(
            r"\b(\d{1,3}(?:\.\d{3}){2,3})\b", line
        )
        if cedula_match and "numero_cedula" not in result:
            result["numero_cedula"] = cedula_match.group(1)

        # Patrón: fecha DD-MMM-AAAA o DD/MM/AAAA
        fecha_match = re.search(
            r"\b(\d{2}[-/](?:ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|OCT|NOV|DIC)[-/]\d{4})\b",
            line, re.IGNORECASE,
        )
        if fecha_match and "fecha_nacimiento" not in result:
            result["fecha_nacimiento"] = fecha_match.group(1)

    def _normalize_dates(self, result: Dict[str, Optional[str]]) -> None:
        """
        Convierte fechas del formato colombiano (DD-MMM-AAAA)
        a ISO 8601 (AAAA-MM-DD).
        """
        MESES_ES = {
            "ENE": "01", "FEB": "02", "MAR": "03",
            "ABR": "04", "MAY": "05", "JUN": "06",
            "JUL": "07", "AGO": "08", "SEP": "09",
            "OCT": "10", "NOV": "11", "DIC": "12",
        }

        date_fields = ["fecha_nacimiento", "fecha_expedicion"]

        for field in date_fields:
            value = result.get(field)
            if not value:
                continue

            # Formato: "15-MAY-1990"
            match = re.match(
                r"(\d{2})\s*[-/]\s*([A-Z]{3})\s*[-/]\s*(\d{4})",
                value, re.IGNORECASE
            )
            if match:
                dia, mes_abr, anio = match.groups()
                mes = MESES_ES.get(mes_abr.upper())
                if mes:
                    result[field] = f"{anio}-{mes}-{dia}"
                    continue

            # Formato: "15/05/1990"
            match = re.match(r"(\d{2})/(\d{2})/(\d{4})", value)
            if match:
                dia, mes, anio = match.groups()
                result[field] = f"{anio}-{mes}-{dia}"

    def _normalize_cedula(self, result: Dict[str, Optional[str]]) -> None:
        """Quita puntos del número de cédula si es necesario."""
        cedula = result.get("numero_cedula")
        if cedula:
            # Guardar versión con puntos y sin puntos
            result["numero_cedula_raw"] = cedula
            result["numero_cedula_sin_puntos"] = cedula.replace(".", "")
