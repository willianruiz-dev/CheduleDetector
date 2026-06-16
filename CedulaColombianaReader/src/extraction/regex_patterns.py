"""
Patrones de expresiones regulares para documentos colombianos.

Define los regex usados para extraer y validar cada campo
de la cédula a partir del texto reconocido por OCR.
"""

import re
from typing import Optional, List, Tuple


class RegexPatterns:
    """
    Colección de patrones regex para cédulas colombianas.

    Cada método retorna el patrón compilado o None si no
    encuentra coincidencia en el texto dado.
    """

    # ── Número de cédula ───────────────────────────────────────
    # Formato: 1.234.567.890 o 1234567890
    CEDULA_PATTERN = re.compile(
        r"\b(\d{1,3}(?:\.\d{3}){2,3})\b"  # con puntos
        r"|"
        r"\b(\d{8,11})\b",  # sin puntos (8-11 dígitos)
    )

    # ── Fechas ─────────────────────────────────────────────────
    # Formato colombiano en cédulas: 15-MAY-1990 o 15/05/1990
    FECHA_ES_PATTERN = re.compile(
        r"(\d{1,2})\s*[-/]\s*"
        r"(ENE|FEB|MAR|ABR|MAY|JUN|JUL|AGO|SEP|OCT|NOV|DIC)"
        r"\s*[-/]\s*"
        r"(\d{4})",
        re.IGNORECASE,
    )

    FECHA_NUM_PATTERN = re.compile(
        r"(\d{2})/(\d{2})/(\d{4})"
    )

    # ── Estatura ───────────────────────────────────────────────
    # Formato: 1.65 (metros con dos decimales)
    ESTATURA_PATTERN = re.compile(
        r"\b(\d)\.(\d{2})\b"
    )

    # ── Grupo sanguíneo ────────────────────────────────────────
    # Formato: O+, A-, AB+, B-
    GRUPO_SANGRE_PATTERN = re.compile(
        r"\b([ABO]{1,2})\s*([+-])\b",
        re.IGNORECASE,
    )

    # ── Sexo ───────────────────────────────────────────────────
    SEXO_PATTERN = re.compile(
        r"\b(MASCULINO|FEMENINO|[MF])\b",
        re.IGNORECASE,
    )

    # ── Nombre (apellidos y nombres) ───────────────────────────
    # Detecta patrones típicos: APELLIDO1 APELLIDO2 NOMBRES
    NOMBRE_LINEA_PATTERN = re.compile(
        r"^([A-ZÁÉÍÓÚÜÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÜÑ]{2,})*)$"
    )

    # ── Nacionalidad ───────────────────────────────────────────
    NACIONALIDAD_PATTERN = re.compile(
        r"\b(COLOMBIAN[OA]|EXTRANJER[OA]|[A-ZÁÉÍÓÚÜÑ]{3,}(?:ANO|ANA|EÑO|EÑA)?)\b",
        re.IGNORECASE,
    )

    # ── Lugares de Colombia ────────────────────────────────────
    # Diccionario de departamentos y ciudades para validación contextual
    DEPARTAMENTOS_COLOMBIA = {
        "AMAZONAS", "ANTIOQUIA", "ARAUCA", "ATLÁNTICO",
        "BOGOTÁ", "BOGOTA", "BOLÍVAR", "BOLIVAR", "BOYACÁ", "BOYACA",
        "CALDAS", "CAQUETÁ", "CAQUETA", "CASANARE", "CAUCA", "CESAR",
        "CHOCÓ", "CHOCO", "CÓRDOBA", "CORDOBA", "CUNDINAMARCA",
        "GUAINÍA", "GUAINIA", "GUAVIARE", "HUILA", "LA GUAJIRA",
        "MAGDALENA", "META", "NARIÑO", "NARINO",
        "NORTE DE SANTANDER", "PUTUMAYO", "QUINDÍO", "QUINDIO",
        "RISARALDA", "SAN ANDRÉS", "SAN ANDRES", "SANTANDER",
        "SUCRE", "TOLIMA", "VALLE DEL CAUCA", "VALLE", "VAUPÉS", "VAUPES",
        "VICHADA",
    }

    CIUDADES_COLOMBIA = {
        "BOGOTÁ", "BOGOTA", "MEDELLÍN", "MEDELLIN", "CALI",
        "BARRANQUILLA", "CARTAGENA", "CÚCUTA", "CUCUTA",
        "BUCARAMANGA", "PEREIRA", "SANTA MARTA", "IBAGUÉ", "IBAGUE",
        "MANIZALES", "VILLAVICENCIO", "PASTO", "MONTERÍA", "MONTERIA",
        "NEIVA", "ARMENIA", "POPAYÁN", "POPAYAN", "SINCELEJO",
        "VALLEDUPAR", "TUNJA", "RIOHACHA", "FLORENCIA",
        "BARRANCABERMEJA", "PALMIRA", "BUENAVENTURA",
        "SOLEDAD", "SOACHA", "BELLO", "ITAGÜÍ", "ITAGUI",
    }

    ABREVIATURAS_COLOMBIA = {
        "D.C.", "D.C", "DC", "MPIO", "DPTO", "DEPTO",
        "STDER", "B/QUILLA", "B/BERMEJA", "B/MANGA",
    }

    @classmethod
    def extract_cedula(cls, text: str) -> Optional[str]:
        """Extrae número de cédula del texto."""
        match = cls.CEDULA_PATTERN.search(text)
        if match:
            # Retornar el grupo que encontró (con o sin puntos)
            return match.group(1) or match.group(2)
        return None

    @classmethod
    def extract_fecha(cls, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Extrae fecha en formato colombiano.
        Retorna (dia, mes, año) o None.
        """
        # Primero intentar formato español (DD-MMM-AAAA)
        match = cls.FECHA_ES_PATTERN.search(text)
        if match:
            return match.group(1), match.group(2).upper(), match.group(3)

        # Luego formato numérico (DD/MM/AAAA)
        match = cls.FECHA_NUM_PATTERN.search(text)
        if match:
            return match.group(1), match.group(2), match.group(3)

        return None

    @classmethod
    def extract_estatura(cls, text: str) -> Optional[str]:
        """Extrae estatura en formato X.XX."""
        match = cls.ESTATURA_PATTERN.search(text)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        return None

    @classmethod
    def extract_grupo_sanguineo(cls, text: str) -> Optional[str]:
        """Extrae grupo sanguíneo (ej. O+)."""
        match = cls.GRUPO_SANGRE_PATTERN.search(text)
        if match:
            return f"{match.group(1).upper()}{match.group(2)}"
        return None

    @classmethod
    def extract_sexo(cls, text: str) -> Optional[str]:
        """Extrae sexo, normalizado a M o F."""
        match = cls.SEXO_PATTERN.search(text)
        if match:
            valor = match.group(1).upper()
            if valor.startswith("M"):
                return "M"
            elif valor.startswith("F"):
                return "F"
            return valor
        return None

    @classmethod
    def extract_nacionalidad(cls, text: str) -> Optional[str]:
        """Extrae nacionalidad del texto."""
        match = cls.NACIONALIDAD_PATTERN.search(text)
        if match:
            return match.group(1).upper()
        return None

    @classmethod
    def extract_lugar(cls, text: str) -> Optional[str]:
        """
        Extrae lugar de nacimiento buscando coincidencias con
        ciudades y departamentos colombianos conocidos.
        """
        text_upper = text.upper().strip()

        # Buscar patrón: "CIUDAD, DEPARTAMENTO" o "CIUDAD (DEPARTAMENTO)"
        # Primero buscar departamentos
        for depto in cls.DEPARTAMENTOS_COLOMBIA:
            if depto in text_upper:
                # Buscar ciudad antes del departamento
                idx = text_upper.find(depto)
                prefix = text_upper[:idx].strip().rstrip(",").rstrip("-")
                if prefix:
                    # Validar que el prefijo sea una ciudad conocida
                    for ciudad in cls.CIUDADES_COLOMBIA:
                        if ciudad in prefix:
                            return f"{ciudad.title()}, {depto.title()}"
                    return f"{prefix.title()}, {depto.title()}"
                return depto.title()

        # Buscar solo ciudad
        for ciudad in sorted(cls.CIUDADES_COLOMBIA, key=len, reverse=True):
            if ciudad in text_upper:
                return ciudad.title()

        # Si no coincide con nada conocido, retornar el texto tal cual
        # pero limpio de abreviaturas
        limpio = text_upper
        for abrev in cls.ABREVIATURAS_COLOMBIA:
            limpio = limpio.replace(abrev, "")

        limpio = re.sub(r"\s+", " ", limpio).strip().strip(",").strip("-")
        return limpio.title() if limpio else None

    @classmethod
    def split_nombre_completo(cls, text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Separa un texto de nombres en primer apellido, segundo apellido, nombres.

        Heurística: las primeras 1-2 palabras son apellidos,
        el resto son nombres.

        Returns:
            Tuple (primer_apellido, segundo_apellido, nombres).
        """
        text = text.strip().upper()
        palabras = text.split()

        if not palabras:
            return None, None, None

        if len(palabras) == 1:
            return palabras[0], None, None
        elif len(palabras) == 2:
            return palabras[0], None, palabras[1]
        elif len(palabras) >= 3:
            # Asumir que las primeras 2 palabras son apellidos
            # y el resto son nombres (convención colombiana)
            return (
                palabras[0],
                palabras[1],
                " ".join(palabras[2:]),
            )

        return None, None, None

    @classmethod
    def normalize_cedula(cls, cedula: str) -> str:
        """Normaliza número de cédula: quita puntos y espacios."""
        return re.sub(r"[.\s]", "", cedula)

    @classmethod
    def normalize_fecha_to_iso(
        cls, dia: str, mes: str, anio: str
    ) -> str:
        """
        Convierte fecha colombiana a ISO 8601 (AAAA-MM-DD).
        mes puede ser numérico (05) o abreviado (MAY).
        """
        MESES = {
            "ENE": "01", "FEB": "02", "MAR": "03",
            "ABR": "04", "MAY": "05", "JUN": "06",
            "JUL": "07", "AGO": "08", "SEP": "09",
            "OCT": "10", "NOV": "11", "DIC": "12",
        }

        mes_num = MESES.get(mes.upper(), mes.zfill(2))
        dia = dia.zfill(2)

        return f"{anio}-{mes_num}-{dia}"
