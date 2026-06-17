"""Diagnóstico del reverso de cédula."""
import cv2
import numpy as np
import sys
sys.path.insert(0, ".")

from src.ocr.ocr_adapter import TextRecognizer, OCRBackend

img = cv2.imread('samples/capturas/cc_reverso.jpg')
print(f'Shape: {img.shape} | Vertical: {img.shape[0] > img.shape[1]}')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f'Gray mean: {gray.mean():.0f} | min: {gray.min()} | max: {gray.max()}')

dark_pixels = (gray < 80).sum()
light_pixels = (gray > 180).sum()
total = gray.size
print(f'Dark%: {dark_pixels/total*100:.1f} | Light%: {light_pixels/total*100:.1f}')

# OCR crudo sin preprocesamiento
tr = TextRecognizer(backend=OCRBackend.EASYOCR, use_gpu=False)
blocks = tr.recognize_full_page(img)
print(f'\nOCR blocks (raw): {len(blocks)}')
for b in blocks[:15]:
    print(f'  ({b["x"]},{b["y"]}) {b["w"]}x{b["h"]} conf={b["conf"]:.2f} | "{b["text"]}"')

# Probar con imagen invertida (si es fondo oscuro)
if gray.mean() < 128:
    inverted = cv2.bitwise_not(img)
    blocks_inv = tr.recognize_full_page(inverted)
    print(f'\nOCR blocks (inverted): {len(blocks_inv)}')
    for b in blocks_inv[:15]:
        print(f'  ({b["x"]},{b["y"]}) {b["w"]}x{b["h"]} conf={b["conf"]:.2f} | "{b["text"]}"')
