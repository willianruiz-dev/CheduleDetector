"""
Clasificador de bloques OCR por posición en la cédula colombiana.

En vez de regiones fijas, clasifica cada texto detectado según
su ubicación (x%, y%) relativa en la imagen.
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

# Etiquetas fijas de la cédula que NO son datos del titular
_LABELS = [
    "APELLIDOS", "NOMBRES", "IDENTIFICACION PERSONAL",
    "REPUBLICA DE COLOMBIA", "CEDULA DE CIUDADANIA",
    "FIRMA", "FIRMA DEL TITULAR", "HUELLA", "HUELLA DACTILAR",
    "ESTATURA", "G.S.", "GRUPO SANGUINEO", "SEXO",
    "FECHA DE NACIMIENTO", "FECHA NACIMIENTO",
    "LUGAR DE NACIMIENTO",
    "FECHA DE EXPEDICION", "FECHA EXPEDICION",
    "NACIONALIDAD", "NACIONAL",
    "NO. DE CEDULA", "NUMERO DE CEDULA",
    "REGISTRADURIA NACIONAL", "REGISTRADURIA",
    "REPUBLICA DE COLOMBLA",  # variante OCR común
    "IDENTIFICACION PERSONAI",
]


def _is_label(text: str) -> bool:
    """
    Detecta si un texto OCR es una etiqueta fija del formato de cédula.
    Usa coincidencia exacta + coincidencia fuzzy por ratio de similitud.
    """
    t = text.upper().strip()
    # Solo textos razonables (al menos 4 caracteres para ser label)
    if len(t) < 4:
        return False

    for label in _LABELS:
        if t == label:
            return True
        # Calcular ratio de caracteres coincidentes
        ratio = _similarity_ratio(t, label)
        if ratio >= 0.75:
            return True
        # Un label contiene al otro (substring largo)
        if len(t) >= 6 and len(label) >= 6:
            if t in label or label in t:
                return True

    return False


def _similarity_ratio(a: str, b: str) -> float:
    """
    Ratio de similitud entre dos strings basado en la longitud de
    la subsecuencia común más larga (LCS).
    """
    n, m = len(a), len(b)
    if n == 0 and m == 0:
        return 1.0
    if n == 0 or m == 0:
        return 0.0

    # LCS via programación dinámica
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    lcs = dp[n][m]
    return (2.0 * lcs) / (n + m)  # Dice coefficient


@dataclass
class TextBlock:
    text: str
    conf: float
    x: int
    y: int
    w: int
    h: int


def classify_blocks(
    blocks: List[dict],
    img_w: int,
    img_h: int,
) -> Dict[str, str]:
    """
    Clasifica bloques OCR en campos de cédula según su posición.

    Layout conocido de la cédula colombiana (anverso):
    ┌──────────────────────────────────────────────┐
    │  REPUBLICA DE COLOMBIA                       │
    │  IDENTIFICACION PERSONAL                     │
    │                                              │
    │  ┌──────┐   CEDULA DE CIUDADANIA             │
    │  │ FOTO │   1.234.567.890    <- número        │
    │  │      │                                    │
    │  │      │   APELLIDOS                        │
    │  │      │   NOMBRES                          │
    │  │      │                                    │
    │  └──────┘   FECHA DE NACIMIENTO              │
    │             LUGAR DE NACIMIENTO              │
    │             ESTATURA    G.S.   SEXO          │
    │             FECHA DE EXPEDICION              │
    │                                              │
    │  ┌──────────────────────────────────┐        │
    │  │ FIRMA                            │        │
    │  └──────────────────────────────────┘        │
    │                                              │
    │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓     │
    │  ▓▓▓▓  PDF417 (código de barras)   ▓▓▓▓     │
    └──────────────────────────────────────────────┘

    La foto está a la izquierda (~0-35% del ancho).
    Los datos están a la derecha (~35-100% del ancho).
    """
    result = {
        "numero_cedula": "",
        "apellidos_nombres": "",
        "fecha_nacimiento": "",
        "lugar_nacimiento": "",
        "estatura": "",
        "grupo_sanguineo": "",
        "sexo": "",
        "fecha_expedicion": "",
    }

    if img_w <= 0 or img_h <= 0:
        return result

    # Filtrar bloques con confianza muy baja
    good_blocks = [b for b in blocks if b["conf"] >= 0.15]

    # Ordenar por posición Y (de arriba a abajo) para respetar orden natural
    good_blocks.sort(key=lambda b: (b["y"], b["x"]))

    # Separar bloques en lado izquierdo (foto/labels) y centro (datos reales)
    left_blocks = []
    center_blocks = []
    right_noise = []
    for b in good_blocks:
        x_pct = b["x"] / img_w
        if x_pct < 0.06:
            left_blocks.append(b)
        elif x_pct > 0.40:
            right_noise.append(b)
        else:
            center_blocks.append(b)

    # Clasificar: procesar primero los bloques del centro (los datos reales)
    for b in center_blocks + left_blocks:
        text = b["text"].strip()
        y_pct = b["y"] / img_h
        x_pct = b["x"] / img_w

        # Filtrar etiquetas conocidas (con fuzzy match para variantes OCR)
        text_upper = text.upper()
        if _is_label(text_upper):
            continue

        # Número de cédula: arriba, contiene muchos dígitos (>=5)
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count >= 5 and y_pct < 0.50:
            result["numero_cedula"] += " " + text

        # Apellidos y nombres: franja media-superior
        elif 0.10 <= y_pct <= 0.65 and x_pct < 0.40 and digit_count < 3:
            # Si ya hay apellidos, agregar a nombres
            if result["apellidos_nombres"]:
                result["apellidos_nombres"] += " " + text
            else:
                result["apellidos_nombres"] = text

        # Fecha de nacimiento: números en franja media
        elif 0.35 <= y_pct <= 0.55 and digit_count >= 2:
            result["fecha_nacimiento"] += " " + text

        # Lugar de nacimiento: franja media
        elif 0.45 <= y_pct <= 0.65 and x_pct >= 0.20 and digit_count < 3:
            result["lugar_nacimiento"] += " " + text

        # Estatura + G.S. + Sexo: franja media-baja
        elif 0.40 <= y_pct <= 0.65 and x_pct >= 0.40:
            if x_pct < 0.52:
                result["estatura"] += " " + text
            elif x_pct < 0.62:
                result["grupo_sanguineo"] += " " + text
            else:
                result["sexo"] += " " + text

        # Fecha de expedición: franja baja
        elif 0.55 <= y_pct <= 0.75 and x_pct >= 0.25 and digit_count >= 2:
            result["fecha_expedicion"] += " " + text

    # Post-procesar: correcciones OCR comunes
    _OCR_FIXES = {
        "HUIZ": "RUIZ",
        "COLOMBLA": "COLOMBIA",
        "PEPUBLICA": "REPUBLICA",
    }
    for key in result:
        if result[key]:
            for bad, good in _OCR_FIXES.items():
                result[key] = result[key].replace(bad, good)

    # Limpiar espacios extra
    for key in result:
        result[key] = " ".join(result[key].split())

    return result


# ── Labels específicas del reverso ──────────────────────────
# Formato: label -> campo. Se emparejan contra el texto OCR.
_REVERSO_LABEL_MAP = {
    "FECHA DE NACIMIENTO": "fecha_nacimiento",
    "FECHA NACIMIENTO": "fecha_nacimiento",
    "LUGAR DE NACIMIENTO": "lugar_nacimiento",
    "ESTATURA": "estatura",
    "G.S.": "grupo_sanguineo",
    "GS": "grupo_sanguineo",
    "GRUPO SANGUINEO": "grupo_sanguineo",
    "SEXO": "sexo",
    "FECHA DE EXPEDICION": "fecha_expedicion",
    "FECHA EXPEDICION": "fecha_expedicion",
}


def _match_reverso_label(text: str) -> Optional[str]:
    """Devuelve el campo al que pertenece un label del reverso, o None.
    Solo acepta coincidencias de alta confianza (ratio >= 0.85) para
    evitar falsos positivos con valores como 'Estatuaa' -> ESTATURA."""
    t = text.upper().strip()

    # Si el texto es corto (<6 chars), exigir coincidencia exacta
    if len(t) < 6:
        for label, campo in _REVERSO_LABEL_MAP.items():
            if t == label:
                return campo
        return None

    # Textos largos: fuzzy matching con umbral alto
    for label, campo in _REVERSO_LABEL_MAP.items():
        if t == label:
            return campo
        if len(label) >= 6 and _similarity_ratio(t, label) >= 0.85:
            return campo

    return None


def classify_blocks_reverso(
    blocks: List[dict],
    img_w: int,
    img_h: int,
) -> Dict[str, str]:
    """
    Clasifica bloques OCR del REVERSO de la cédula colombiana.

    Layout real (cc_reverso.jpg 1600x1200):
    ┌────────────────────────────────────────────────┐
    │  0-20%        35-55%          55-100%          │
    │              ┌──────────────┬─────────────────┤
    │              │ FECHA DE NAC │ 27-NOV-1983     │  y 18-28%
    │              │              │ CAÑASGORDAS     │  y 22-32%
    │              │              │ (ANTIOQUIA)     │  y 27-32%
    │              │ Lugar...     │                 │  y 30-35%
    │              ├──────────────┼──────┬──────────┤
    │              │              │      │          │
    │              │ 1.72         │ O+   │ M        │  y 35-42%
    │              │ Estatuaa     │ Gs   │ SEXO     │  y 42-46%
    │              ├──────────────┴──────┴──────────┤
    │              │ 29-NOV-2001 CANASGORDAS         │  y 46-52%
    │              │ Fecha   lugaade ExpediCion      │  y 52-56%
    │              ├─────────────────────────────────┤
    │              │ REGISTRADOR NACIONAL            │  y 56-60%
    │ indice Der.  │ ivan Duque Escodaa              │
    │              ├─────────────────────────────────┤
    │              │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
    │              │ P-0107600-14099842-...          │  y 82-90%
    │              │ 0559402012A 01   104059525      │
    └────────────────────────────────────────────────┘

    Estrategia: filtrar labels conocidos, usar solo valores reales,
    clasificar por franjas Y específicas del reverso.
    """
    result = {
        "numero_cedula": "",
        "apellidos_nombres": "",
        "fecha_nacimiento": "",
        "lugar_nacimiento": "",
        "estatura": "",
        "grupo_sanguineo": "",
        "sexo": "",
        "fecha_expedicion": "",
    }

    if img_w <= 0 or img_h <= 0:
        return result

    # Filtrar labels y headers para quedarnos solo con valores
    values = []
    for b in blocks:
        text = b["text"].strip()
        if b["conf"] < 0.05:
            continue
        if _is_reverso_junk(text):
            continue
        values.append({"text": text, "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"], "conf": b["conf"]})

    if not values:
        return result

    # ── Clasificar por franjas Y ──────────────────────────
    for b in values:
        text = b["text"]
        y_pct = b["y"] / img_h
        x_pct = b["x"] / img_w
        digit_count = sum(1 for c in text if c.isdigit())

        # ── Fecha de nacimiento: y 18-28%, con muchos dígitos ──
        if 0.18 <= y_pct <= 0.28 and digit_count >= 4:
            result["fecha_nacimiento"] += " " + text

        # ── Lugar de nacimiento: y 22-35%, texto sin números ──
        elif 0.22 <= y_pct <= 0.35 and digit_count == 0 and x_pct > 0.30:
            result["lugar_nacimiento"] += " " + text

        # ── Estatura: y 35-42%, patrón N.NN, izquierda ──
        elif 0.35 <= y_pct <= 0.42 and x_pct < 0.48:
            import re
            if re.match(r'^\d\.\d{2}$', text):
                result["estatura"] = text

        # ── Grupo sanguíneo: y 35-42%, centro ──
        elif 0.35 <= y_pct <= 0.42 and 0.48 <= x_pct <= 0.65:
            import re
            cleaned = text.upper().replace(" ", "").replace("0", "O")
            if re.match(r'^[ABO][+-]$', cleaned):
                result["grupo_sanguineo"] = cleaned

        # ── Sexo: y 35-42%, derecha ──
        elif 0.35 <= y_pct <= 0.42 and x_pct > 0.65:
            t = text.upper().strip()
            if t in ("M", "F"):
                result["sexo"] = t

        # ── Fecha de expedición: y 46-55%, con dígitos ──
        elif 0.46 <= y_pct <= 0.55 and digit_count >= 4:
            result["fecha_expedicion"] += " " + text

    # ── Limpiar ─────────────────────────────────────────
    for key in result:
        if result[key]:
            result[key] = " ".join(result[key].split()).strip()

    # ── Correcciones ────────────────────────────────────
    # Si fecha_expedicion incluye texto sin dígitos, limpiar
    if result["fecha_expedicion"]:
        import re
        m = re.search(r'\d{1,2}[\s-]*[A-Z]{3}[\s-]*\d{4}', result["fecha_expedicion"])
        if m:
            result["fecha_expedicion"] = m.group(0)

    # Si fecha_nacimiento tiene texto extra (ej: "CAÑASGORDAS"), mover a lugar
    if result["fecha_nacimiento"]:
        import re
        parts = result["fecha_nacimiento"].split()
        date_parts = []
        lugar_parts = []
        for p in parts:
            if re.match(r'\d{1,2}$', p) or re.match(r'^[A-Z]{3}$', p) or re.match(r'^\d{4}$', p) or re.match(r'\d{1,2}-[A-Z]{3}-\d{4}', p):
                date_parts.append(p)
            else:
                lugar_parts.append(p)
        if date_parts:
            result["fecha_nacimiento"] = " ".join(date_parts)
        if lugar_parts:
            result["lugar_nacimiento"] = " ".join(lugar_parts) + " " + result["lugar_nacimiento"]
            result["lugar_nacimiento"] = result["lugar_nacimiento"].strip()

    return result


def _is_reverso_junk(text: str) -> bool:
    """Detecta si un texto es una etiqueta/header del reverso (no es un dato)."""
    t = text.upper().strip()
    # Datos válidos de 1 caracter: M, F
    if t in ("M", "F"):
        return False
    if len(t) < 2:
        return True
    # Labels conocidos
    if t in ("FECHA DE NACIMIENTO", "FECHA NACIMIENTO", "LUGAR DE NACIMIENTO",
             "ESTATURA", "G.S.", "GS", "GRUPO SANGUINEO",
             "SEXO", "FECHA DE EXPEDICION", "FECHA EXPEDICION",
             "REGISTRADOR NACIONAL", "INDICE DERECHO", "IVAN DUQUE ESCODAA",
             "CEDULA DE CIUDADANIA"):
        return True
    # Fuzzy match para labels
    if _similarity_ratio(t, "REPUBLICA DE COLOMBIA") >= 0.70:
        return True
    if _similarity_ratio(t, "REGISTRADOR NACIONAL") >= 0.80:
        return True
    if _match_reverso_label(t):
        return True
    # Textos muy cortos que son ruido
    if len(t) <= 3 and not any(c.isdigit() for c in t) and t not in ("M", "F"):
        return True
    return False
