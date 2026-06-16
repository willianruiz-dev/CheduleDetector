"""
Genera imágenes sintéticas de cédulas colombianas para testing.

Formato realista con DOS CARAS:
  - Anverso: foto, número, apellidos, nombres, firma
  - Reverso: fecha/lugar nacimiento, estatura, G.S., sexo,
             fecha/lugar expedición, huella, código PDF417
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# ─── Constantes de diseño ─────────────────────────────────────
W, H = 1280, 800
AMARILLO_CEDULA = np.array([0, 235, 248], dtype=np.uint8)  # HSV -> BGR
VERDE_HEADER = (0, 100, 50)
GRIS_LABEL = (70, 70, 70)
GRIS_TEXTO = (30, 30, 30)
GRIS_BORDE = (160, 160, 160)


def _hsv2bgr(h, s, v):
    """Convierte HSV a BGR uint8."""
    return tuple(int(c) for c in cv2.cvtColor(
        np.full((1, 1, 3), [h, s, v], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0])


def _fondo_cedula(img, tipo):
    """Aplica el color de fondo según tipo de cédula."""
    colores = {
        "CC": (0, 220, 240),    # Amarillo
        "CE": (100, 180, 240),  # Azul claro
        "TI": (10, 180, 240),   # Naranja suave
    }
    hsv = colores.get(tipo, (0, 220, 240))
    color_bgr = _hsv2bgr(*hsv)
    overlay = np.full_like(img, color_bgr, dtype=np.uint8)
    return cv2.addWeighted(img, 0.75, overlay, 0.25, 0)


def _texto(img, text, x, y, scale=0.5, color=GRIS_TEXTO, thickness=1, bold=False):
    """Helper para escribir texto."""
    font = cv2.FONT_HERSHEY_DUPLEX if bold else cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, scale, color, thickness)


def _marca_agua(img):
    """Añade patrón tenue de fondo como las cédulas reales."""
    h, w = img.shape[:2]
    overlay = img.copy()
    for i in range(0, w, 60):
        cv2.line(overlay, (i, 0), (i, h), (200, 200, 180), 1)
    for j in range(0, h, 60):
        cv2.line(overlay, (0, j), (w, j), (200, 200, 180), 1)
    return cv2.addWeighted(img, 0.95, overlay, 0.05, 0)


# ════════════════════════════════════════════════════════════════
#  ANVERSO (FRONTAL)
# ════════════════════════════════════════════════════════════════

def crear_anverso_cc(numero="1.032.951.572", apellidos="GARCIA MARQUEZ",
                      nombres="GABRIEL JOSE"):
    """Crea el ANVERSO de una cédula de ciudadanía."""
    img = np.ones((H, W, 3), dtype=np.uint8) * 248
    img = _fondo_cedula(img, "CC")
    img = _marca_agua(img)

    # ── Borde ─────────────────────────────────────────────────
    cv2.rectangle(img, (20, 20), (W - 20, H - 20), (100, 100, 100), 2)

    # ── Header ────────────────────────────────────────────────
    _texto(img, "REPUBLICA DE COLOMBIA", W // 2 - 170, 55, 0.8, VERDE_HEADER, 2, bold=True)
    _texto(img, "CEDULA DE CIUDADANIA", W // 2 - 155, 90, 0.7, (0, 80, 40), 2, bold=True)
    cv2.line(img, (40, 110), (W - 40, 110), (120, 120, 120), 1)

    # ── Foto (izquierda) ──────────────────────────────────────
    foto_x, foto_y = 50, 140
    foto_w, foto_h = 220, 270
    cv2.rectangle(img, (foto_x, foto_y), (foto_x + foto_w, foto_y + foto_h), GRIS_BORDE, 2)
    # Silueta básica
    cv2.circle(img, (foto_x + 110, foto_y + 90), 55, (170, 170, 170), 2)
    # Torso
    pts = np.array([
        [foto_x + 40, foto_y + 260],
        [foto_x + 180, foto_y + 260],
        [foto_x + 150, foto_y + 160],
        [foto_x + 70, foto_y + 160],
    ], np.int32)
    cv2.polylines(img, [pts], True, (170, 170, 170), 2)

    # ── Datos principales (derecha) ───────────────────────────
    x_dato = foto_x + foto_w + 60
    y0 = foto_y + 20
    dy = 38

    _texto(img, "NUMERO:", x_dato, y0, 0.45, GRIS_LABEL)
    _texto(img, numero, x_dato + 130, y0, 0.55, GRIS_TEXTO, bold=True)

    _texto(img, "APELLIDOS:", x_dato, y0 + dy, 0.45, GRIS_LABEL)
    _texto(img, apellidos, x_dato + 130, y0 + dy, 0.55, GRIS_TEXTO, bold=True)

    _texto(img, "NOMBRES:", x_dato, y0 + 2 * dy, 0.45, GRIS_LABEL)
    _texto(img, nombres, x_dato + 130, y0 + 2 * dy, 0.55, GRIS_TEXTO, bold=True)

    # ── Firma ──────────────────────────────────────────────────
    firma_y = y0 + 3 * dy + 40
    _texto(img, "FIRMA:", x_dato, firma_y, 0.45, GRIS_LABEL)
    # Simular firma manuscrita
    pts_firma = np.array([
        [x_dato + 80, firma_y + 10],
        [x_dato + 150, firma_y - 5],
        [x_dato + 200, firma_y + 5],
        [x_dato + 280, firma_y - 8],
        [x_dato + 350, firma_y + 8],
        [x_dato + 380, firma_y],
    ], np.int32)
    cv2.polylines(img, [pts_firma], False, (0, 0, 0), 2)

    # ── Pie ────────────────────────────────────────────────────
    cv2.line(img, (40, H - 50), (W - 40, H - 50), (120, 120, 120), 1)
    _texto(img, "REGISTRADURIA NACIONAL DEL ESTADO CIVIL",
           W // 2 - 230, H - 25, 0.55, (100, 100, 100), bold=True)

    return img


# ════════════════════════════════════════════════════════════════
#  REVERSO (TRASERO)
# ════════════════════════════════════════════════════════════════

def crear_reverso_cc(fecha_nac="06-MAR-1927", lugar_nac="ARACATACA (MAGDALENA)",
                      estatura="1.68", gs="O+", sexo="M",
                      fecha_exp="15-ABR-2020", lugar_exp="BOGOTA D.C."):
    """Crea el REVERSO de una cédula de ciudadanía colombiana."""
    img = np.ones((H, W, 3), dtype=np.uint8) * 250
    img = _fondo_cedula(img, "CC")
    img = _marca_agua(img)

    # ── Borde ─────────────────────────────────────────────────
    cv2.rectangle(img, (20, 20), (W - 20, H - 20), (100, 100, 100), 2)

    # ── Columna izquierda: datos ──────────────────────────────
    x_label = 60
    x_val = 300
    y0 = 80
    dy = 50

    campos = [
        ("FECHA DE NACIMIENTO:", fecha_nac),
        ("LUGAR DE NACIMIENTO:", lugar_nac),
        ("ESTATURA:", f"{estatura} m"),
        ("G.S. (RH):", gs),
        ("SEXO:", sexo),
        ("FECHA DE EXPEDICION:", fecha_exp),
        ("LUGAR DE EXPEDICION:", lugar_exp),
    ]

    for i, (label, valor) in enumerate(campos):
        y = y0 + i * dy
        _texto(img, label, x_label, y, 0.48, GRIS_LABEL)
        _texto(img, valor, x_val, y, 0.52, GRIS_TEXTO, bold=True)

    # ── Huella dactilar (derecha) ─────────────────────────────
    huella_x = W - 260
    huella_y = 100
    huella_w, huella_h = 200, 280
    cv2.rectangle(img, (huella_x, huella_y),
                  (huella_x + huella_w, huella_y + huella_h), GRIS_BORDE, 2)
    _texto(img, "HUELLA", huella_x + 55, huella_y + 150, 0.5, (150, 150, 150))
    # Dibujar huella simulada (círculos concéntricos tipo huella)
    cx, cy = huella_x + huella_w // 2, huella_y + huella_h // 2 - 20
    for r in range(30, 90, 12):
        cv2.ellipse(img, (cx, cy), (r, r + 10), -15, 0, 270, (170, 170, 170), 1)
    for r in range(20, 70, 10):
        cv2.ellipse(img, (cx - 5, cy - 15), (r - 5, r), 10, 0, 270, (170, 170, 170), 1)

    # ── Código de barras PDF417 ────────────────────────────────
    barcode_x, barcode_y = 40, H - 200
    barcode_w, barcode_h = W - 80, 110

    # Generar PDF417 sintético visualmente
    bw = np.ones((barcode_h, barcode_w, 3), dtype=np.uint8) * 255
    np.random.seed(99)
    col = 0
    while col < barcode_w:
        w_bar = np.random.choice([2, 3, 4, 5, 6, 7], p=[0.08, 0.12, 0.18, 0.25, 0.22, 0.15])
        if col + w_bar > barcode_w:
            break
        is_black = np.random.random() > 0.42
        cv2.line(bw, (col, 0), (col, barcode_h), (0, 0, 0) if is_black else (255, 255, 255), w_bar)
        col += w_bar
    # Barras de inicio/fin más gruesas
    cv2.line(bw, (0, 0), (0, barcode_h), (0, 0, 0), 4)
    cv2.line(bw, (barcode_w - 1, 0), (barcode_w - 1, barcode_h), (0, 0, 0), 4)

    img[barcode_y:barcode_y + barcode_h, barcode_x:barcode_x + barcode_w] = bw

    _texto(img, "PDF417", barcode_x + 10, barcode_y + barcode_h + 20, 0.4, (120, 120, 120))

    # ── Pie ────────────────────────────────────────────────────
    cv2.line(img, (40, H - 40), (W - 40, H - 40), (120, 120, 120), 1)
    _texto(img, "REGISTRADURIA NACIONAL DEL ESTADO CIVIL",
           W // 2 - 230, H - 15, 0.55, (100, 100, 100), bold=True)

    return img


# ════════════════════════════════════════════════════════════════
#  CÉDULA DE EXTRANJERÍA (CE)
# ════════════════════════════════════════════════════════════════

def crear_anverso_ce(numero="E-987654", apellidos="SMITH",
                      nombres="JOHN MICHAEL"):
    """Anverso de cédula de extranjería."""
    img = np.ones((H, W, 3), dtype=np.uint8) * 248
    img = _fondo_cedula(img, "CE")

    cv2.rectangle(img, (20, 20), (W - 20, H - 20), (100, 100, 100), 2)
    _texto(img, "REPUBLICA DE COLOMBIA", W // 2 - 170, 55, 0.8, VERDE_HEADER, 2, bold=True)
    _texto(img, "CEDULA DE EXTRANJERIA", W // 2 - 170, 90, 0.7, (0, 80, 40), 2, bold=True)
    cv2.line(img, (40, 110), (W - 40, 110), (120, 120, 120), 1)

    foto_x, foto_y = 50, 140
    foto_w, foto_h = 220, 270
    cv2.rectangle(img, (foto_x, foto_y), (foto_x + foto_w, foto_y + foto_h), GRIS_BORDE, 2)
    cv2.circle(img, (foto_x + 110, foto_y + 80), 45, (170, 170, 170), 2)

    x_dato = foto_x + foto_w + 60
    y0 = foto_y + 20
    dy = 38

    _texto(img, "NUMERO:", x_dato, y0, 0.45, GRIS_LABEL)
    _texto(img, numero, x_dato + 130, y0, 0.55, GRIS_TEXTO, bold=True)
    _texto(img, "APELLIDOS:", x_dato, y0 + dy, 0.45, GRIS_LABEL)
    _texto(img, apellidos, x_dato + 130, y0 + dy, 0.55, GRIS_TEXTO, bold=True)
    _texto(img, "NOMBRES:", x_dato, y0 + 2 * dy, 0.45, GRIS_LABEL)
    _texto(img, nombres, x_dato + 130, y0 + 2 * dy, 0.55, GRIS_TEXTO, bold=True)
    _texto(img, "NACIONALIDAD:", x_dato, y0 + 3 * dy, 0.45, GRIS_LABEL)
    _texto(img, "ESTADOUNIDENSE", x_dato + 130, y0 + 3 * dy, 0.55, GRIS_TEXTO, bold=True)

    firma_y = y0 + 4 * dy + 30
    _texto(img, "FIRMA:", x_dato, firma_y, 0.45, GRIS_LABEL)
    cv2.line(img, (x_dato + 80, firma_y + 10), (x_dato + 320, firma_y - 5), (0, 0, 0), 2)

    cv2.line(img, (40, H - 50), (W - 40, H - 50), (120, 120, 120), 1)
    _texto(img, "REGISTRADURIA NACIONAL DEL ESTADO CIVIL",
           W // 2 - 230, H - 25, 0.55, (100, 100, 100), bold=True)

    return img


def crear_reverso_ce(fecha_nac="15-JUL-1985", lugar_nac="MIAMI (EE.UU.)",
                      estatura="1.78", gs="A+", sexo="M",
                      fecha_exp="20-ENE-2024", lugar_exp="BOGOTA D.C."):
    """Reverso de cédula de extranjería."""
    return crear_reverso_cc(fecha_nac, lugar_nac, estatura, gs, sexo,
                             fecha_exp, lugar_exp)


# ════════════════════════════════════════════════════════════════
#  UTILIDADES
# ════════════════════════════════════════════════════════════════

def anadir_ruido_realista(img, intensidad=0.015):
    """Añade ruido Gaussiano + ligera rotación."""
    noise = np.random.randn(*img.shape) * intensidad * 255
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    angle = np.random.uniform(-1.5, 1.5)
    h, w = noisy.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(noisy, M, (w, h), borderValue=(248, 248, 248))


def guardar(img, path, calidad=92):
    """Guarda imagen JPEG."""
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, calidad])
    print(f"  [+] {path.name}  ({img.shape[1]}x{img.shape[0]})")


# ════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Genera imágenes sintéticas realistas de cédulas colombianas"
    )
    parser.add_argument("--tipo", choices=["CC", "CE", "TI", "todas"], default="todas")
    parser.add_argument("--output", "-o", type=str, default="samples")
    parser.add_argument("--ruido", type=float, default=0.008)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("  GENERADOR DE CÉDULAS COLOMBIANAS SINTÉTICAS")
    print("=" * 55)

    if args.tipo in ("CC", "todas"):
        print("\n📋 Cédula de Ciudadanía (CC):")
        anverso = crear_anverso_cc()
        reverso = crear_reverso_cc()
        if args.ruido > 0:
            anverso = anadir_ruido_realista(anverso, args.ruido)
            reverso = anadir_ruido_realista(reverso, args.ruido)
        guardar(anverso, output_dir / "cc_anverso.jpg")
        guardar(reverso, output_dir / "cc_reverso.jpg")

    if args.tipo in ("CE", "todas"):
        print("\n📋 Cédula de Extranjería (CE):")
        anverso = crear_anverso_ce()
        reverso = crear_reverso_ce()
        if args.ruido > 0:
            anverso = anadir_ruido_realista(anverso, args.ruido)
            reverso = anadir_ruido_realista(reverso, args.ruido)
        guardar(anverso, output_dir / "ce_anverso.jpg")
        guardar(reverso, output_dir / "ce_reverso.jpg")

    print("\n✅ Listo. Las imágenes están en:", output_dir.resolve())
    print("   Formato realista: ANVERSO (foto + datos) | REVERSO (huella + PDF417)\n")


if __name__ == "__main__":
    main()
