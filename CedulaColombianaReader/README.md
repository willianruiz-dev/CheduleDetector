# Lector de Cédulas Colombianas con TensorFlow

Pipeline completo en Python para leer y extraer información de documentos de identidad colombianos (cédula de ciudadanía, cédula de extranjería y tarjeta de identidad) usando TensorFlow, OpenCV y OCR.

## 🚀 Instalación

```bash
# Clonar o copiar el proyecto
cd CedulaColombianaReader

# Instalar dependencias base
pip install -r requirements.txt

# Instalar al menos un motor de OCR (elegir uno)
pip install easyocr        # Rápido, buen soporte español
# o
pip install paddleocr      # Alta precisión

# (Opcional) Para decodificar el código PDF417
pip install pyzbar         # Windows: requiere instalar ZBar aparte
```

## 📖 Uso

```bash
# Procesar una cédula
python main.py --imagen ruta/cedula.jpg

# Con PaddleOCR (más preciso)
python main.py --imagen cedula.jpg --backend paddleocr

# Procesar carpeta completa
python main.py --lote carpeta_cedulas/ --output resultados.json

# Con GPU
python main.py --imagen cedula.jpg --gpu
```

## 🧠 Arquitectura

```
CedulaColombianaReader/
├── main.py                          # Punto de entrada
├── requirements.txt
├── src/
│   ├── models/
│   │   └── cedula_colombiana.py     # Modelo de datos (dataclass)
│   ├── preprocessing/
│   │   ├── image_enhancer.py        # CLAHE, denoising, sharpen
│   │   ├── perspective_corrector.py # Corrección homográfica
│   │   └── binarizer.py             # Binarización adaptativa
│   ├── pdf417/
│   │   ├── barcode_decoder.py       # Decodificación PDF417
│   │   └── barcode_parser.py        # Parseo de datos del PDF417
│   ├── detection/
│   │   └── region_detector.py       # Detección de regiones y tipo
│   ├── ocr/
│   │   ├── text_recognizer.py       # CRNN con TensorFlow/Keras
│   │   └── ocr_adapter.py           # Adaptador multi-backend
│   ├── extraction/
│   │   ├── regex_patterns.py        # Patrones regex colombianos
│   │   ├── field_extractor.py       # Extractor de campos
│   │   └── validators/
│   │       └── cross_validator.py   # Validación OCR vs PDF417
│   └── pipeline/
│       └── cedula_reader_pipeline.py # Pipeline principal
└── tests/
    ├── samples_cc/                   # Muestras cédula de ciudadanía
    ├── samples_ce/                   # Muestras cédula de extranjería
    └── samples_ti/                   # Muestras tarjeta de identidad
```

## 🔧 Flujo de procesamiento

1. **Carga** → Imagen desde archivo, bytes o array numpy
2. **Perspectiva** → Detección de bordes + homografía
3. **Mejora** → CLAHE + denoising + sharpen
4. **PDF417** → Decodificación del código de barras
5. **Detección** → Tipo de documento (CC/CE/TI) + regiones de interés
6. **OCR** → Reconocimiento de texto en cada región
7. **Extracción** → Regex + mapeo a campos del modelo
8. **Validación** → Cruce OCR vs PDF417 + reporte de discrepancias

## 📊 Campos extraídos

| Campo | Descripción |
|---|---|
| `numero_cedula` | Número con puntos (1.234.567.890) |
| `primer_apellido` | Primer apellido |
| `segundo_apellido` | Segundo apellido (puede ser nulo) |
| `nombres` | Nombres |
| `fecha_nacimiento` | ISO 8601 (AAAA-MM-DD) |
| `lugar_nacimiento` | Ciudad, Departamento |
| `estatura` | En metros (1.65) |
| `grupo_sanguineo` | Ej: O+, A-, AB+ |
| `sexo` | M o F |
| `fecha_expedicion` | ISO 8601 |
| `nacionalidad` | COLOMBIANA u otro país |
| `tipo_documento` | CC, CE o TI |
| `confidence_score` | Confianza global 0-1 |
| `discrepancias_pdf417` | Alertas de validación |

## ⚠️ Privacidad

**IMPORTANTE:** Los datos extraídos de cédulas colombianas están protegidos por la Ley 1581 de 2012 (Habeas Data). Todo el procesamiento es **local**, sin enviar datos a servicios externos.

## 📝 Licencia

Este proyecto es únicamente con fines educativos y de investigación.
