"""
Extractor de campos de cédula colombiana a partir de texto OCR.

Toma el texto reconocido de cada región y lo mapea
a los campos del modelo CedulaColombiana usando regex.
"""

from typing import Dict, Optional, Tuple

from src.models.cedula_colombiana import CedulaColombiana, TipoDocumento
from src.extraction.regex_patterns import RegexPatterns


class FieldExtractor:
    """
    Extrae campos de una cédula colombiana a partir de los
    resultados OCR de cada región.
    """

    def __init__(self):
        self.patterns = RegexPatterns()

    def extract(
        self,
        ocr_results: Dict[str, Tuple[str, float]],
        tipo: TipoDocumento = TipoDocumento.CC,
    ) -> CedulaColombiana:
        """
        Extrae todos los campos a partir de los resultados OCR por región.

        Args:
            ocr_results: Diccionario {nombre_region: (texto, confianza)}.
            tipo: Tipo de documento detectado.

        Returns:
            CedulaColombiana poblada con los campos extraídos.
        """
        cedula = CedulaColombiana()
        cedula.tipo_documento = tipo

        # ── Número de cédula ──────────────────────────────────
        if "numero_cedula" in ocr_results:
            texto, conf = ocr_results["numero_cedula"]
            cedula.numero_cedula = self.patterns.extract_cedula(texto)
            cedula.confidence_por_campo["numero_cedula"] = conf

        # ── Apellidos y nombres ────────────────────────────────
        if "apellidos_nombres" in ocr_results:
            texto, conf = ocr_results["apellidos_nombres"]
            p_ap, s_ap, noms = self.patterns.split_nombre_completo(texto)
            cedula.primer_apellido = p_ap
            cedula.segundo_apellido = s_ap
            cedula.nombres = noms
            cedula.confidence_por_campo["apellidos_nombres"] = conf

        # ── Fecha de nacimiento ────────────────────────────────
        if "fecha_nacimiento" in ocr_results:
            texto, conf = ocr_results["fecha_nacimiento"]
            fecha = self.patterns.extract_fecha(texto)
            if fecha:
                cedula.fecha_nacimiento = self.patterns.normalize_fecha_to_iso(*fecha)
            cedula.confidence_por_campo["fecha_nacimiento"] = conf

        # ── Lugar de nacimiento ────────────────────────────────
        if "lugar_nacimiento" in ocr_results:
            texto, conf = ocr_results["lugar_nacimiento"]
            cedula.lugar_nacimiento = self.patterns.extract_lugar(texto)
            cedula.confidence_por_campo["lugar_nacimiento"] = conf

        # ── Estatura ───────────────────────────────────────────
        if "estatura" in ocr_results:
            texto, conf = ocr_results["estatura"]
            cedula.estatura = self.patterns.extract_estatura(texto)
            cedula.confidence_por_campo["estatura"] = conf

        # ── Grupo sanguíneo ────────────────────────────────────
        if "grupo_sanguineo" in ocr_results:
            texto, conf = ocr_results["grupo_sanguineo"]
            cedula.grupo_sanguineo = self.patterns.extract_grupo_sanguineo(texto)
            cedula.confidence_por_campo["grupo_sanguineo"] = conf

        # ── Sexo ───────────────────────────────────────────────
        if "sexo" in ocr_results:
            texto, conf = ocr_results["sexo"]
            cedula.sexo = self.patterns.extract_sexo(texto)
            cedula.confidence_por_campo["sexo"] = conf

        # ── Fecha de expedición ────────────────────────────────
        if "fecha_expedicion" in ocr_results:
            texto, conf = ocr_results["fecha_expedicion"]
            fecha = self.patterns.extract_fecha(texto)
            if fecha:
                cedula.fecha_expedicion = self.patterns.normalize_fecha_to_iso(*fecha)
            cedula.confidence_por_campo["fecha_expedicion"] = conf

        # ── Nacionalidad ───────────────────────────────────────
        # Puede venir de varias regiones o del texto completo
        for region_key in ["nacionalidad", "apellidos_nombres", "raw_text"]:
            if region_key in ocr_results and not cedula.nacionalidad:
                texto, conf = ocr_results[region_key]
                nac = self.patterns.extract_nacionalidad(texto)
                if nac:
                    cedula.nacionalidad = nac

        # ── Calcular confianza global ──────────────────────────
        if cedula.confidence_por_campo:
            cedula.confidence_score = sum(
                cedula.confidence_por_campo.values()
            ) / len(cedula.confidence_por_campo)

        # ── Texto OCR completo (para debug) ────────────────────
        cedula.raw_text = " | ".join(
            f"[{k}]: {v[0]}"
            for k, v in ocr_results.items()
            if v[0]
        )

        return cedula
