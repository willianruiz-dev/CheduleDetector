"""
Clasificador de bloques OCR por posición en la cédula colombiana.

Basado en zonas proporcionales exactas definidas desde el plano
de la cédula (anverso y reverso), con soporte para CC, CE y TI.

Cada zona se define como [ymin, xmin, ymax, xmax] en rango 0-1.
Un bloque se asigna a una zona si su centro cae dentro de ella.
"""
from typing import List, Dict, Optional
import re


# ═══════════════════════════════════════════════════════════════
# ZONAS DEL ANVERSO (frente)
# ═══════════════════════════════════════════════════════════════
# Layout real tras ROTATE_90_COUNTERCLOCKWISE → 1600x1200:
#
#  ┌──────────────────────────────────────────────────┐
#  │  REPUBLICA DE COLOMBIA                           │
#  │  IDENTIFICACION PERSONAL       CEDULA DE         │
#  │                                CIUDADANIA        │
#  │  NUMERO: [=== 70435855 ===]       ┌──────────┐   │
#  │  APELLIDOS: [= HUIZ ZAPATA =]     │          │   │
#  │  NOMBRES: [= WILLIAN DARIO =]     │  [FOTO]  │   │
#  │                                   │          │   │
#  │  ┌──────────────┐                 └──────────┘   │
#  │  │   [FIRMA]    │                                │
#  │  └──────────────┘                                │
#  └──────────────────────────────────────────────────┘
#
# Coordenadas en formato [ymin, xmin, ymax, xmax] (0-1)
# Valores medidos: cedula cy≈0.38, apellidos cy≈0.48, nombres cy≈0.60

ANVERSO_ZONES: Dict[str, list] = {
    # Número de cédula: arriba a la izquierda
    "numero_cedula":      [0.30, 0.05, 0.44, 0.42],

    # Apellidos: debajo del número (label + valor)
    "apellidos":          [0.42, 0.05, 0.56, 0.42],

    # Nombres: debajo de apellidos (label + valor)
    "nombres":            [0.55, 0.05, 0.70, 0.42],

    # Firma: abajo a la izquierda
    "firma":              [0.68, 0.05, 0.88, 0.45],
}


# ═══════════════════════════════════════════════════════════════
# ZONAS DEL REVERSO (atrás)
# ═══════════════════════════════════════════════════════════════
# Layout real (cc_reverso.jpg → 1600x1200):
#
#  ┌──────────────────────────────────────────────────┐
#  │  ┌──────────┐  FECHA DE NACIMIENTO  27-NOV-1983 │
#  │  │          │  CAÑASGORDAS                       │
#  │  │ [HUELLA] │  (ANTIOQUIA)                       │
#  │  │          │  Lugar De NaciMiEnto                │
#  │  │          │                                     │
#  │  │          │  1.72       O+         M            │
#  │  │          │  Estatuaa   Gs   aH    SEXO         │
#  │  └──────────┘                                     │
#  │              29-NOV-2001 CANASGORDAS              │
#  │              Fecha  lugaade ExpediCion            │
#  │              REGISTRADOR NACIONAL                 │
#  │              ivan Duque Escodaa                   │
#  │  ┌────────────────────────────────────────────┐  │
#  │  │ P-0107600-14099842-M-0070435855-20020112   │  │
#  │  │ 0559402012A 01              104059525      │  │
#  │  └────────────────────────────────────────────┘  │
#  └──────────────────────────────────────────────────┘

REVERSO_ZONES: Dict[str, list] = {
    # Fecha de nacimiento: arriba derecha (y 15-28%)
    "fecha_nacimiento":   [0.12, 0.42, 0.30, 0.92],

    # Lugar de nacimiento: debajo de fecha (y 22-40%)
    "lugar_nacimiento":   [0.22, 0.42, 0.40, 0.92],

    # Estatura: franja media, parte izquierda (y 35-50%, x 37-58%)
    "estatura":            [0.35, 0.37, 0.50, 0.58],

    # Grupo sanguíneo: franja media, centro (y 35-50%, x 55-73%)
    "grupo_sanguineo":     [0.35, 0.53, 0.50, 0.73],

    # Sexo: franja media, derecha (y 35-50%, x 70-92%)
    "sexo":                [0.35, 0.68, 0.50, 0.92],

    # Fecha y lugar de expedición: debajo de estatura/gs/sexo (y 48-65%)
    "fecha_expedicion":    [0.46, 0.42, 0.66, 0.92],

    # Código de barras PDF417: franja inferior (y 78-95%)
    "codigo_barras":       [0.78, 0.02, 0.96, 0.98],
}


# ═══════════════════════════════════════════════════════════════
# ZONAS DE LA CÉDULA DIGITAL NUEVA (formato horizontal/landscape)
# ═══════════════════════════════════════════════════════════════
# Las imágenes de la cédula digital nueva vienen en horizontal
# (w > h). No necesitan rotación.
#
# Layout del ANVERSO en landscape (~1600x1000):
#  ┌──────────────────────────────────────────────────────┐
#  │              REPÚBLICA DE COLOMBIA                    │
#  │                                                       │
#  │  ┌────────────────┐    ┌──────────────────────────┐  │
#  │  │                │    │  NUIP: 1.234.567.890     │  │
#  │  │    [FOTO]      │    │                          │  │
#  │  │                │    └──────────────────────────┘  │
#  │  └────────────────┘                                   │
#  │  APELLIDOS: Walteros                                 │
#  │  NOMBRES: Laura                SEXO: F               │
#  │  COL                                                  │
#  │  FECHA NAC: 15 ABR 2004                              │
#  │  LUGAR NAC: CARTAGENA (BOLIVAR)                      │
#  │  EXPEDICION: 19 ABR 2027 CARTAGENA                   │
#  └──────────────────────────────────────────────────────┘

ANVERSO_ZONES_NUEVA: Dict[str, list] = {
    # NUIP: arriba derecha (y=12-22%, x=65-95%)
    "nuip_number":         [0.10, 0.60, 0.24, 0.97],

    # Apellidos: izquierda, debajo de la foto (y=17-28%, x=35-60%)
    "last_names":          [0.15, 0.35, 0.28, 0.62],

    # Nombres: izquierda, debajo de apellidos (y=28-40%, x=35-60%)
    "given_names":         [0.26, 0.35, 0.42, 0.62],

    # Sexo: derecha central (y=38-52%, x=62-80%)
    "sexo_nuevo":          [0.38, 0.62, 0.52, 0.82],

    # País / Nacionalidad: izquierda central (y=43-52%, x=35-55%)
    "pais_emisor":         [0.42, 0.33, 0.52, 0.55],

    # Fecha de nacimiento: izquierda (y=48-62%, x=35-65%)
    "birth_date":          [0.48, 0.35, 0.62, 0.65],

    # Lugar de nacimiento: izquierda (y=58-75%, x=35-85%)
    "birth_place":         [0.56, 0.35, 0.75, 0.85],

    # Fecha y lugar de expedición: abajo (y=75-96%, x=35-85%)
    "issue_info":          [0.73, 0.35, 0.96, 0.88],
}

# Layout del REVERSO en landscape (~1600x1000):
#  ┌──────────────────────────────────────────────────────┐
#  │                                                       │
#  │                    ┌──────────┐                       │
#  │                    │ [QR CODE]│                       │
#  │                    └──────────┘                       │
#  │                                                       │
#  │  ════════════════════════════════════════════════════ │
#  │  ════════ [MRZ LÍNEA 1] ═══════════════════════════ │
#  │  ════════ [MRZ LÍNEA 2] ═══════════════════════════ │
#  │  ════════ [MRZ LÍNEA 3] ═══════════════════════════ │
#  └──────────────────────────────────────────────────────┘

REVERSO_ZONES_NUEVA: Dict[str, list] = {
    # Código QR: centro-derecha superior
    "qr_code":             [0.05, 0.55, 0.45, 0.95],

    # MRZ línea 1: franja inferior (y=66-73%)
    "mrz_line_1":          [0.64, 0.02, 0.74, 0.98],

    # MRZ línea 2: franja inferior (y=74-81%)
    "mrz_line_2":          [0.73, 0.02, 0.82, 0.98],

    # MRZ línea 3: franja inferior (y=82-90%)
    "mrz_line_3":          [0.82, 0.02, 0.92, 0.98],
}


# ═══════════════════════════════════════════════════════════════
# Filtros: textos que NO son datos del titular
# ═══════════════════════════════════════════════════════════════
_IGNORE_EXACT = {
    "M", "F",  # sexo se maneja aparte, pero no queremos que caiga en otras zonas
}

_IGNORE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^REPUBLICA\s*(DE\s*)?COLOMBI?A?$",
        r"^CEDULA\s*(DE\s*)?CIUDADANIA$",
        r"^CEDULA\s*(DE\s*)?EXTRANJERIA$",
        r"^TARJETA\s*(DE\s*)?IDENTIDAD$",
        r"^IDENTIFICACION\s*PERSONAL$",
        r"^REGISTRADOR\s*NACIONAL$",
        r"^REGISTRADURIA",
        r"^FIRMA(\s*DEL\s*TITULAR)?$",
        r"^HUELLA(\s*DACTILAR)?$",
        r"^INDICE\s*DERECHO$",
        r"^(FECHA|LUGAR)\s*(DE\s*)?(NACIMIENTO|EXPEDICION)$",
        r"^(FECHA|LUGAR)\s*(Y|DE)\s*(NACIMIENTO|EXPEDICION)$",
        r"^(ESTATU(A|RA)|ESTATUAA?)$",
        r"^(G\.?S\.?|GRUPO\s*SANGUINEO|RH|AH?)$",
        r"^SEXO$",
        r"^NACIONALIDAD$",
        r"^(NUMERO|NO\.?)\s*(DE\s*)?CEDULA$",
        r"^(NOMBRES?|APELLIDOS?)$",
        r"^(PRIMER|SEGUNDO)\s*APELLIDO$",
        r"^IVAN\s*DUQUE",
        r"^[A-Z]?\d{6,}[A-Z]\s*\d{2}$",  # código numérico auxiliar del PDF417 (ej: 0559402012A 01)
    ]
]


def _is_label(text: str) -> bool:
    """Detecta si un texto es una etiqueta del formato (no es un dato)."""
    t = text.strip()
    # M y F son sexo válido, no son labels
    if t.upper() in ("M", "F"):
        return False
    if len(t) <= 1:
        return True
    if len(t) <= 3 and not any(c.isdigit() for c in t):
        return True
    for pat in _IGNORE_PATTERNS:
        if pat.match(t):
            return True
    return False


def _point_in_zone(cx: float, cy: float, zone: list) -> bool:
    """Verifica si un punto (cx, cy en 0..1) cae dentro de una zona."""
    ymin, xmin, ymax, xmax = zone
    return xmin <= cx <= xmax and ymin <= cy <= ymax


def classify_blocks(
    blocks: List[dict],
    img_w: int,
    img_h: int,
) -> Dict[str, str]:
    """
    Clasifica bloques OCR del ANVERSO usando zonas proporcionales exactas.
    """
    result: Dict[str, str] = {
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

    # Filtrar ruido y labels
    good = [b for b in blocks if b["conf"] >= 0.10 and not _is_label(b["text"])]

    # Agrupar por zona
    zone_texts: Dict[str, list] = {k: [] for k in ANVERSO_ZONES}

    for b in good:
        cx = (b["x"] + b["w"] / 2) / img_w
        cy = (b["y"] + b["h"] / 2) / img_h

        for campo, zone in ANVERSO_ZONES.items():
            if _point_in_zone(cx, cy, zone):
                zone_texts[campo].append((b["text"], b["y"]))

    # ── Número de cédula ──
    if zone_texts["numero_cedula"]:
        # Preferir el texto con más dígitos
        texts = [t for t, _ in zone_texts["numero_cedula"]]
        best = max(texts, key=lambda t: sum(1 for c in t if c.isdigit()))
        result["numero_cedula"] = best

    # ── Apellidos + Nombres ──
    # Ordenar por Y (de arriba a abajo): apellidos primero, luego nombres
    apellidos_texts = [t for t, _ in sorted(zone_texts["apellidos"], key=lambda x: x[1])]
    nombres_texts = [t for t, _ in sorted(zone_texts["nombres"], key=lambda x: x[1])]
    combined = apellidos_texts + nombres_texts
    if combined:
        result["apellidos_nombres"] = " ".join(combined)

    return result


def classify_blocks_reverso(
    blocks: List[dict],
    img_w: int,
    img_h: int,
) -> Dict[str, str]:
    """
    Clasifica bloques OCR del REVERSO usando zonas proporcionales exactas.
    """
    result: Dict[str, str] = {
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

    # Filtrar ruido y labels
    good = [b for b in blocks if b["conf"] >= 0.05 and not _is_label(b["text"])]

    # Agrupar por zona
    zone_texts: Dict[str, list] = {k: [] for k in REVERSO_ZONES}

    for b in good:
        cx = (b["x"] + b["w"] / 2) / img_w
        cy = (b["y"] + b["h"] / 2) / img_h

        for campo, zone in REVERSO_ZONES.items():
            if _point_in_zone(cx, cy, zone):
                zone_texts[campo].append(b["text"])

    # ── Fecha de nacimiento ──
    for t in zone_texts["fecha_nacimiento"]:
        m = re.search(r"(\d{1,2}[-/][A-Z]{3}[-/]\d{4})", t, re.IGNORECASE)
        if m:
            result["fecha_nacimiento"] = m.group(1)
            break
    if not result["fecha_nacimiento"] and zone_texts["fecha_nacimiento"]:
        for t in zone_texts["fecha_nacimiento"]:
            if sum(1 for c in t if c.isdigit()) >= 4:
                result["fecha_nacimiento"] = t
                break

    # ── Lugar de nacimiento ──
    for t in zone_texts["lugar_nacimiento"]:
        cleaned = re.sub(r"[()]", "", t).strip()
        if sum(1 for c in cleaned if c.isdigit()) == 0 and len(cleaned) >= 3:
            if result["lugar_nacimiento"]:
                result["lugar_nacimiento"] += " " + t
            else:
                result["lugar_nacimiento"] = t

    # ── Estatura ──
    for t in zone_texts["estatura"]:
        m = re.search(r"\d\.\d{2}", t)
        if m:
            result["estatura"] = m.group(0)
            break

    # ── Grupo sanguíneo ──
    for t in zone_texts["grupo_sanguineo"]:
        cleaned = t.upper().replace(" ", "").replace("0", "O")
        m = re.search(r"[ABO][+-]", cleaned)
        if m:
            result["grupo_sanguineo"] = m.group(0)
            break

    # ── Sexo ──
    for t in zone_texts["sexo"]:
        tt = t.upper().strip()
        if tt in ("M", "F"):
            result["sexo"] = tt
            break

    # ── Fecha de expedición ──
    for t in zone_texts["fecha_expedicion"]:
        m = re.search(r"(\d{1,2}[-/][A-Z]{3}[-/]\d{4})", t, re.IGNORECASE)
        if m:
            result["fecha_expedicion"] = m.group(1)
            break
    if not result["fecha_expedicion"] and zone_texts["fecha_expedicion"]:
        for t in zone_texts["fecha_expedicion"]:
            if sum(1 for c in t if c.isdigit()) >= 4:
                result["fecha_expedicion"] = t
                break

    # ── Limpiar espacios ──
    for key in result:
        if result[key]:
            result[key] = " ".join(result[key].split())

    return result


# ═══════════════════════════════════════════════════════════════
# Clasificación para CÉDULA DIGITAL NUEVA (formato azul)
# ═══════════════════════════════════════════════════════════════

def classify_blocks_nueva(
    blocks: List[dict],
    img_w: int,
    img_h: int,
) -> Dict[str, str]:
    """
    Clasifica bloques OCR del ANVERSO de la cédula digital nueva.
    """
    result: Dict[str, str] = {
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

    good = [b for b in blocks if b["conf"] >= 0.10 and not _is_label(b["text"])]

    zone_texts: Dict[str, list] = {k: [] for k in ANVERSO_ZONES_NUEVA}

    for b in good:
        cx = (b["x"] + b["w"] / 2) / img_w
        cy = (b["y"] + b["h"] / 2) / img_h

        for campo, zone in ANVERSO_ZONES_NUEVA.items():
            if _point_in_zone(cx, cy, zone):
                zone_texts[campo].append((b["text"], b["y"]))

    # ── NUIP / Número de cédula ──
    if zone_texts["nuip_number"]:
        texts = [t for t, _ in zone_texts["nuip_number"]]
        best = max(texts, key=lambda t: sum(1 for c in t if c.isdigit()))
        # Limpiar: quitar prefijos no numéricos, conservar dígitos y puntos
        cleaned = re.sub(r"[^0-9.]", "", best)
        # Si queda algo como 1.234.567.890, eliminar puntos para dejar solo dígitos
        result["numero_cedula"] = cleaned.replace(".", "") if len(cleaned) >= 6 else cleaned

    # ── Apellidos + Nombres ──
    apellidos = [t for t, _ in sorted(zone_texts["last_names"], key=lambda x: x[1])]
    nombres = [t for t, _ in sorted(zone_texts["given_names"], key=lambda x: x[1])]
    combined = apellidos + nombres
    if combined:
        result["apellidos_nombres"] = " ".join(combined)

    # ── Fecha de nacimiento ──
    for t, _ in zone_texts["birth_date"]:
        m = re.search(r"(\d{1,2}[-/][A-Z]{3}[-/]\d{4})", t, re.IGNORECASE)
        if m:
            result["fecha_nacimiento"] = m.group(1)
            break
    if not result["fecha_nacimiento"] and zone_texts["birth_date"]:
        for t, _ in zone_texts["birth_date"]:
            # También acepta formato DD/MM/AAAA
            m = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", t)
            if m:
                result["fecha_nacimiento"] = m.group(1)
                break
    if not result["fecha_nacimiento"] and zone_texts["birth_date"]:
        for t, _ in zone_texts["birth_date"]:
            if sum(1 for c in t if c.isdigit()) >= 4:
                result["fecha_nacimiento"] = t
                break

    # ── Lugar de nacimiento ──
    for t, _ in zone_texts["birth_place"]:
        cleaned = re.sub(r"[()]", "", t).strip()
        if len(cleaned) >= 3:
            if result["lugar_nacimiento"]:
                result["lugar_nacimiento"] += " " + t
            else:
                result["lugar_nacimiento"] = t

    # ── Fecha de expedición ──
    for t, _ in zone_texts["issue_info"]:
        # Puede contener fecha + lugar
        m = re.search(r"(\d{1,2}[-/][A-Z]{3}[-/]\d{4})", t, re.IGNORECASE)
        if m:
            result["fecha_expedicion"] = m.group(1)
            break
        m = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{4})", t)
        if m:
            result["fecha_expedicion"] = m.group(1)
            break
    if not result["fecha_expedicion"] and zone_texts["issue_info"]:
        for t, _ in zone_texts["issue_info"]:
            if sum(1 for c in t if c.isdigit()) >= 4:
                result["fecha_expedicion"] = t
                break

    # ── Limpiar espacios ──
    for key in result:
        if result[key]:
            result[key] = " ".join(result[key].split())

    # ── Sexo (zona dedicada en el anverso nuevo) ──
    for t, _ in zone_texts.get("sexo_nuevo", []):
        tt = t.upper().strip().rstrip(".")
        if tt in ("M", "F", "MAS", "MASC", "MASCULINO", "FEM", "FEMENINO"):
            result["sexo"] = "M" if tt.startswith("M") else "F"
            break

    return result


def classify_blocks_reverso_nueva(
    blocks: List[dict],
    img_w: int,
    img_h: int,
) -> Dict[str, str]:
    """
    Clasifica bloques OCR del REVERSO de la cédula digital nueva.

    Extrae principalmente las líneas MRZ (Machine Readable Zone),
    que contienen todos los datos del titular codificados.
    """
    result: Dict[str, str] = {
        "numero_cedula": "",
        "apellidos_nombres": "",
        "fecha_nacimiento": "",
        "lugar_nacimiento": "",
        "estatura": "",
        "grupo_sanguineo": "",
        "sexo": "",
        "fecha_expedicion": "",
        "mrz_raw": "",
    }

    if img_w <= 0 or img_h <= 0:
        return result

    good = [b for b in blocks if b["conf"] >= 0.05 and not _is_label(b["text"])]

    zone_texts: Dict[str, list] = {k: [] for k in REVERSO_ZONES_NUEVA}

    for b in good:
        cx = (b["x"] + b["w"] / 2) / img_w
        cy = (b["y"] + b["h"] / 2) / img_h

        for campo, zone in REVERSO_ZONES_NUEVA.items():
            if _point_in_zone(cx, cy, zone):
                zone_texts[campo].append(b["text"])

    # ── Concatenar líneas MRZ ──
    mrz_lines = []
    for key in ("mrz_line_1", "mrz_line_2", "mrz_line_3"):
        for t in zone_texts[key]:
            # Limpiar: solo letras, dígitos y <
            cleaned = re.sub(r"[^A-Z0-9<]", "", t.upper())
            if len(cleaned) >= 10:
                mrz_lines.append(cleaned)
    if mrz_lines:
        result["mrz_raw"] = "\n".join(mrz_lines)

    return result


def detect_format(blocks: List[dict], img_w: int, img_h: int,
                  es_reverso: bool = False) -> str:
    """
    Detecta si la imagen corresponde a la cedula antigua o la nueva digital.

    Estrategia:
    1. MRZ: si hay 2+ lineas con '<' en el tercio inferior → 'nueva' (definitivo)
    2. Si es_reverso=True y NO hay MRZ → 'clasica' (reverso clasico no tiene MRZ)
    3. Votacion SOLO con zonas de anverso de ambos formatos + labels

    Returns:
        'nueva' o 'clasica'
    """
    if img_w <= 0 or img_h <= 0:
        return "clasica"

    # ── 1. MRZ (exclusivo de la digital nueva, 100% confiable) ──
    mrz_candidates = 0
    for b in blocks:
        t = b["text"].strip().upper()
        cy = (b["y"] + b["h"] / 2) / img_h
        bw = b["w"] / img_w
        if cy > 0.60 and bw > 0.60 and t.count("<") >= 5 and len(t) >= 20:
            mrz_candidates += 1
    if mrz_candidates >= 2:
        return "nueva"

    # ── 2. Reverso sin MRZ → clasica ──
    if es_reverso:
        return "clasica"

    # ── 3. Votacion solo con anversos (sin REVERSO_ZONES que solapan) ──
    score_nueva = 0.0
    score_clasica = 0.0

    for b in blocks:
        if b["conf"] < 0.10:
            continue
        cx = (b["x"] + b["w"] / 2) / img_w
        cy = (b["y"] + b["h"] / 2) / img_h

        for zone in ANVERSO_ZONES_NUEVA.values():
            if _point_in_zone(cx, cy, zone):
                score_nueva += 1.0
                break
        for zone in ANVERSO_ZONES.values():
            if _point_in_zone(cx, cy, zone):
                score_clasica += 2.0
                break

    # ── 4. Labels caracteristicos ──
    for b in blocks:
        t = b["text"].strip().upper()
        if t in ("NUIP", "NUIP:"):
            score_nueva += 5.0
        if "CEDULA DE CIUDADANIA" in t or "CEDULA DE EXTRANJERIA" in t:
            score_clasica += 5.0

    return "nueva" if score_nueva > score_clasica else "clasica"


# ── Compatibilidad con código que referencia estas funciones ──
def _is_reverso_junk(text: str) -> bool:
    return _is_label(text)


def _match_reverso_label(text: str) -> Optional[str]:
    return None
