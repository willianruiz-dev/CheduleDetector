# Prompt para Agente de IA: Lector de Documentos de Identidad Colombianos con TensorFlow

```
Eres un agente de desarrollo de software especializado en visión por computadora,
aprendizaje automático y procesamiento de imágenes. Tu perfil de razonamiento es
"deep_reasoning_math": desglosas problemas complejos paso a paso, fundamentas cada
decisión técnica con lógica clara y anticipas casos límite antes de escribir código.

---

## 🎯 OBJETIVO

Diseñar y proporcionar el código base completo para una solución en **C#**
(preferiblemente, usando .NET 8+) o **Python** (como alternativa) que permita la
**lectura y extracción automatizada de información clave** desde imágenes de
documentos de identidad **exclusivamente colombianos**:
- **Cédula de Ciudadanía** (documento amarillo/formato horizontal).
- **Cédula de Extranjería** (documento para residentes extranjeros en Colombia).
- **Tarjeta de Identidad** (documento para menores de edad, formato azul).

El proyecto debe utilizar **TensorFlow** como motor principal de reconocimiento
óptico de caracteres (OCR) y detección de regiones de interés.

---

## 🧠 RAZONAMIENTO PREVIO (antes de codificar)

Antes de generar cualquier línea de código, quiero que razones sobre:

1. **¿Qué hace único al documento de identidad colombiano?**
   - Layout horizontal característico con foto a la izquierda y datos a la derecha.
   - Código de barras PDF417 en la parte inferior con datos legibles por máquina.
   - Campos específicos: primer apellido, segundo apellido, nombres, número de
     cédula (con puntos como separadores de miles, ej. 1.234.567.890), fecha y
     lugar de nacimiento, estatura, grupo sanguíneo y factor RH, sexo, firma e
     índice dactilar derecho.
   - Elementos de seguridad: hologramas, microtextos, tinta reactiva, patrones
     guilloché y fondo de seguridad con la bandera de Colombia.
   - Variantes: cédula amarilla (formato clásico) y cédula azul/digital (nuevo
     formato con chip).
   - La cédula de extranjería tiene formato similar pero con campos adicionales
     como nacionalidad de origen y tipo de visa.

2. **¿Por qué TensorFlow y no solo Tesseract?**
   - Tesseract es rápido pero frágil ante las distorsiones de fotos tomadas con
     cámara de celular (muy comunes en Colombia).
   - La tipografía de las cédulas colombianas tiene espaciado y peso específicos
     que un modelo custom puede aprender mejor.
   - TensorFlow permite un pipeline unificado: detección de zonas (CNN) + OCR
     (CRNN/Attention) + decodificación del código de barras PDF417.

3. **¿Qué enfoque de arquitectura es más robusto?**
   - Pipeline modular híbrido: Preprocesamiento → Decodificación PDF417 (datos
     de respaldo) → Detección de regiones → OCR → Cruce y validación cruzada
     entre datos del PDF417 y texto OCR.
   - El código PDF417 ofrece una fuente de verdad complementaria que permite
     validar y corregir errores del OCR.

---

## 📋 REQUISITOS FUNCIONALES

### 1. Carga y preprocesamiento de imagen
- Soportar formatos: JPG, PNG, BMP, TIFF, PDF (escaneos de cédula).
- Redimensionamiento inteligente manteniendo relación de aspecto (la cédula
  colombiana tiene proporción horizontal ~1.6:1).
- Corrección de perspectiva (transformación homográfica) mediante detección de
  los 4 vértices del documento (usando Canny + contornos o un detector entrenado).
- Mejora de contraste (CLAHE - Contrast Limited Adaptive Histogram Equalization)
  para resaltar texto sobre el fondo amarillo de la cédula.
- Binarización adaptativa (Otsu, Sauvola) optimizada para tinta negra sobre
  fondo claro.
- Eliminación de ruido y atenuación de elementos de seguridad (hologramas,
  patrones guilloché) que interfieren con el OCR.

### 2. Decodificación del código de barras PDF417
- **Paso crítico:** la cédula colombiana contiene un código PDF417 en la parte
  inferior que codifica los datos del ciudadano en formato legible por máquina.
- Implementar decodificación PDF417 usando bibliotecas como ZXing (Java/C#)
  o pdf417decoder (Python).
- Los datos del PDF417 sirven como **ground truth parcial** para validar y
  corregir los resultados del OCR.
- Manejar casos donde el código de barras está dañado, borroso o parcialmente
  cubierto (fallback al OCR puro).

### 3. Detección de regiones de interés (ROI) específicas de la cédula colombiana
- Segmentar el documento en zonas específicas del formato colombiano:
  - **Foto** (lado izquierdo, ~35% del ancho).
  - **Número de cédula** (esquina superior derecha, tipografía grande).
  - **Apellidos y Nombres** (bloque central derecho, dos líneas).
  - **Fecha de nacimiento** (derecha, formato DD-MMM-AAAA o DD/MM/AAAA).
  - **Lugar de nacimiento** (debajo de fecha de nacimiento).
  - **Estatura** (campo numérico con unidad en metros, ej. 1.70).
  - **Grupo Sanguíneo y RH** (ej. O+, A-, AB+).
  - **Sexo** (M o F, generalmente abreviado).
  - **Fecha de expedición** (parte inferior del bloque de datos).
  - **Firma del titular** (zona inferior derecha).
  - **Huella dactilar / índice derecho** (esquina inferior izquierda).
  - **Código de barras PDF417** (franja inferior horizontal).
- Implementar con TensorFlow Object Detection API (SSD MobileNet o EfficientDet)
  o YOLO convertido a TensorFlow.
- Guía para entrenar con un dataset etiquetado de cédulas colombianas reales
  (anonimizadas).

### 4. Reconocimiento Óptico de Caracteres (OCR)
- Implementar una arquitectura **CRNN** (CNN + RNN + CTC Loss) o un modelo
  basado en **Attention** (Transformer encoder-decoder) con TensorFlow/Keras.
- Considerar el uso de modelos pre-entrenados como TrOCR, EasyOCR o PaddleOCR
  y cómo integrarlos vía ONNX o TensorFlow Serving.
- Manejar caracteres especiales del español colombiano: letras con tilde (á, é,
  í, ó, ú), eñe (ñ), diéresis (ü).
- Las fechas en cédulas colombianas suelen usar abreviaturas de meses en español
  (ENE, FEB, MAR, etc.) → el modelo debe reconocer estos tokens.
- Manejar múltiples líneas de texto en cada región (ej. apellidos en una línea
  y nombres en otra).

### 5. Extracción estructurada de datos
- Parsear el texto reconocido de cada región y mapearlo a un modelo de datos
  estructurado (clase/POCO en C#, dataclass en Python) específico para cédula
  colombiana.
- Implementar expresiones regulares adaptadas a formatos colombianos:
  - Número de cédula: `\d{1,3}(\.\d{3})*` (con puntos separadores de miles).
  - Fechas en español: `\d{1,2}-[A-Z]{3}-\d{4}` o `\d{2}/\d{2}/\d{4}`.
  - Estatura: `\d\.\d{2}` (metros con dos decimales).
  - Grupo sanguíneo: `[ABO]{1,2}[+-]`.
- **Validación cruzada:** comparar cada campo extraído por OCR contra los datos
  decodificados del PDF417 y reportar discrepancias.
- Normalización de valores (ej. convertir "UN MILLON" a "1.000.000" para números
  de cédula altos).

### 6. Manejo de errores y robustez
- Puntuación de confianza por campo extraído (OCR confidence + concordancia con
  PDF417).
- Rechazo de documentos con calidad insuficiente (score global < umbral
  configurable).
- Logging detallado en español para depuración.
- Soporte para procesamiento por lotes (batch processing) de múltiples cédulas.
- Detección automática del tipo de documento (cédula de ciudadanía vs. cédula
  de extranjería vs. tarjeta de identidad) basada en características visuales
  (color predominante, layout).

---

## 📦 ENTREGABLES

### A. Estructura del proyecto

Proporciona la estructura de carpetas recomendada, por ejemplo:

```
CedulaColombianaReader/
├── src/
│   ├── Preprocessing/
│   │   ├── ImageEnhancer.cs / .py
│   │   ├── PerspectiveCorrector.cs / .py
│   │   └── Binarizer.cs / .py
│   ├── PDF417/
│   │   ├── BarcodeDecoder.cs / .py
│   │   └── BarcodeParser.cs / .py       # parseo de datos del PDF417
│   ├── Detection/
│   │   ├── RegionDetector.cs / .py
│   │   └── models/                       # modelos .pb / .h5
│   ├── OCR/
│   │   ├── TextRecognizer.cs / .py
│   │   └── models/
│   ├── Extraction/
│   │   ├── FieldExtractor.cs / .py
│   │   ├── RegexPatterns.cs / .py        # patrones regex colombianos
│   │   └── Validators/
│   │       └── CrossValidator.cs / .py   # validación OCR vs PDF417
│   ├── Models/
│   │   ├── CedulaColombiana.cs / .py
│   │   ├── TipoDocumento.cs / .py        # enum: CC, CE, TI
│   │   └── ResultadoValidacion.cs / .py
│   └── Pipeline/
│       └── CedulaReaderPipeline.cs / .py
├── tests/
│   ├── samples_cc/                        # muestras cédula de ciudadanía
│   ├── samples_ce/                        # muestras cédula de extranjería
│   └── samples_ti/                        # muestras tarjeta de identidad
├── notebooks/                             # (Python) experimentación
├── requirements.txt / *.csproj
└── README.md
```

### B. Clase/Módulo principal

Proporciona el código fuente en C# (con ML.NET + TensorFlow.NET o mediante
interop con Python vía gRPC/API REST) o en Python puro con TensorFlow/Keras.

La clase principal debe llamarse `CedulaReaderPipeline` y debe exponer
un método como:

```csharp
// C#
public CedulaColombiana ProcesarCedula(byte[] imageBytes);
// o
public async Task<CedulaColombiana> ProcesarCedulaAsync(Stream imageStream);
```

```python
# Python
def procesar_cedula(imagen_path: str) -> CedulaColombiana:
    ...
```

El pipeline debe:
1. Detectar automáticamente el tipo de documento (CC, CE o TI).
2. Decodificar el PDF417 si está presente.
3. Ejecutar OCR sobre las regiones de interés.
4. Cruzar y validar datos.
5. Devolver el objeto `CedulaColombiana` poblado con confianza por campo.

### C. Modelo de datos `CedulaColombiana`

Define una clase/DTO con los siguientes campos específicos para la cédula
colombiana (todos `string` o `Nullable<T>`):

| Campo                      | Ejemplo                    | Notas                                    |
|----------------------------|----------------------------|------------------------------------------|
| NumeroCedula               | "1.234.567.890"            | Con puntos separadores de miles          |
| PrimerApellido             | "GARCÍA"                   |                                          |
| SegundoApellido            | "LÓPEZ"                    | Puede ser nulo (cédulas antiguas)        |
| Nombres                    | "MARÍA FERNANDA"           |                                          |
| FechaNacimiento            | "1990-05-15"               | Normalizado a ISO 8601                   |
| LugarNacimiento            | "BOGOTÁ D.C."              | Departamento o ciudad                    |
| Estatura                   | "1.65"                     | En metros, formato X.XX                  |
| GrupoSanguineo             | "O+"                       | Ej. A+, B-, AB+, O-                      |
| Sexo                       | "F"                        | M o F                                    |
| FechaExpedicion            | "2020-03-10"               | Normalizado a ISO 8601                   |
| Nacionalidad               | "COLOMBIANA"               | "COLOMBIANA" o país de origen en CE      |
| TipoDocumento              | "CC"                       | CC, CE o TI                              |
| DatosPDF417                | "{...}"                    | JSON/String con datos del código PDF417  |
| RawText                    | texto completo OCR         |                                          |
| ImagenRostro               | byte[]                     | Recorte de la foto (opcional)            |
| ImagenFirma                | byte[]                     | Recorte de la firma (opcional)           |
| ConfidenceScore            | 0.95                       | Float 0.0 - 1.0                          |
| DiscrepanciasPDF417        | ["Estatura difiere", ...]  | Alertas de validación cruzada            |
| ProcessingTimeMs           | 340                        | Int                                      |

### D. Dependencias

Lista las dependencias exactas y sus versiones:

**Python:**
- tensorflow >= 2.15
- opencv-python >= 4.9
- numpy >= 1.26
- Pillow >= 10.0
- scikit-image >= 0.22
- pdf417decoder >= 0.1  (decodificación de PDF417 colombiano)
- pyzbar >= 0.1.9        (alternativa para PDF417 + QR)
- (opcional) easyocr, paddleocr, transformers (HuggingFace)

**C#/.NET:**
- TensorFlow.NET >= 0.120
- OpenCvSharp4 >= 4.9
- SixLabors.ImageSharp >= 3.1
- ZXing.Net >= 0.16.9     (decodificación de PDF417)
- (opcional) ML.NET para NER/post-procesamiento

### E. Ejemplo de uso

Proporciona un `Program.cs` / `main.py` funcional con:

1. Carga de una imagen de ejemplo (cédula colombiana real anonimizada o sintética).
2. Detección automática del tipo de documento (CC, CE o TI).
3. Decodificación del PDF417 y extracción de datos.
4. Ejecución del pipeline completo de OCR.
5. Validación cruzada PDF417 vs OCR con reporte de discrepancias.
6. Impresión en consola de los datos extraídos en formato JSON.
7. Medición del tiempo de procesamiento.

Ejemplo de salida esperada en consola:

```json
{
  "tipo_documento": "CC",
  "numero_cedula": "1.234.567.890",
  "primer_apellido": "GARCÍA",
  "segundo_apellido": "LÓPEZ",
  "nombres": "MARÍA FERNANDA",
  "fecha_nacimiento": "1990-05-15",
  "lugar_nacimiento": "BOGOTÁ D.C.",
  "estatura": "1.65",
  "grupo_sanguineo": "O+",
  "sexo": "F",
  "fecha_expedicion": "2020-03-10",
  "nacionalidad": "COLOMBIANA",
  "confidence_score": 0.94,
  "discrepancias_pdf417": [],
  "tiempo_procesamiento_ms": 340
}
```

### F. Guía de entrenamiento (si se requiere modelo custom para cédulas colombianas)

Si se propone entrenar un modelo propio, incluye:
- Estructura esperada del dataset de cédulas colombianas (anonimizadas):
  - Carpetas por tipo: `cc_amarilla/`, `cc_azul/`, `ce/`, `ti/`.
  - Anotaciones en formato COCO o YOLO con las regiones de interés.
- Script de entrenamiento con TensorFlow/Keras.
- Métricas de evaluación:
  - CER (Character Error Rate) y WER (Word Error Rate) en texto extraído.
  - Tasa de acierto en tipo de documento (CC/CE/TI).
  - Tasa de decodificación exitosa del PDF417.
- Estrategia de data augmentation adaptada a cédulas colombianas:
  - Rotación leve (±5°) simulando fotos torcidas.
  - Variación de brillo y contraste (simulando diferentes condiciones de luz).
  - Desenfoque gaussiano (simulando fotos movidas/baja resolución).
  - Ruido de impresión (simulando cédulas deterioradas o fotocopias).
  - Distorsión de perspectiva simulada.
  - Superposición de sombras parciales (simulando fotos con sombra del celular).

---

## ⚠️ DESAFÍOS Y RECOMENDACIONES

Menciona en tu respuesta:

1. **Variedad de formatos dentro de Colombia:**
   - Cédula de ciudadanía amarilla (formato clásico, vigente desde ~2000): fondo
     amarillo, tipografía con serifas, PDF417.
   - Cédula azul/digital (nuevo formato desde ~2020): fondo azul con degradado,
     chip NFC, diseño actualizado, PDF417 en nueva ubicación.
   - Tarjeta de identidad (menores de 14-18 años): formato azul con rosa, campos
     similares pero sin PDF417 en versiones antiguas.
   - Cédula de extranjería: formato similar al CC pero con campos adicionales
     (nacionalidad de origen, tipo de visa, fecha de vencimiento).
   - Estrategia: sistema de clasificación previa del tipo de documento +
     configuración de coordenadas de ROI por tipo.

2. **Calidad de imagen en contexto colombiano:**
   - Fotos tomadas con celulares de gama baja-media, a menudo con mala iluminación,
     flash que quema el holograma, o ángulo oblicuo.
   - Cédulas deterioradas por uso diario (raspaduras, dobleces, desgaste del
     holograma).
   - Fotos de fotocopias, no del documento original (común en trámites informales).
   - Estrategia: umbral de calidad configurable + rechazo temprano con mensaje
     claro al usuario ("La imagen está muy oscura, por favor use mejor iluminación").

3. **Desafíos específicos del PDF417 colombiano:**
   - No todas las cédulas tienen PDF417 legible (versiones muy antiguas).
   - El PDF417 puede estar parcialmente cubierto por sellos o firmas.
   - La decodificación puede fallar en imágenes de baja resolución.
   - Estrategia: usar PDF417 como fuente complementaria, no única. El OCR debe
     funcionar de forma independiente.

4. **Rendimiento en dispositivos locales:**
   - Tiempo de inferencia esperado en CPU de laptop típica colombiana (Core i5,
     8 GB RAM): < 2 segundos por cédula.
   - Estrategias de optimización: cuantización INT8, TensorFlow Lite, ONNX Runtime.
   - Posibilidad de usar GPU NVIDIA si está disponible (común en equipos de
     entidades públicas).

5. **Privacidad y seguridad de datos:**
   - **CRÍTICO:** los datos de cédulas colombianas son datos personales protegidos
     por la Ley 1581 de 2012 (Habeas Data) y la Ley 1266 de 2008.
   - Todo el procesamiento debe ser **local** (on-premise/on-device), sin enviar
     imágenes ni datos a servicios cloud.
   - No almacenar imágenes sin anonimizar. Si se requiere guardar para auditoría,
     encriptar en reposo.
   - La foto del rostro extraída del documento también es dato sensible
     biométrico — requiere protección adicional.

6. **Particularidades del español colombiano:**
   - Los nombres de ciudades/departamentos pueden ser largos (ej.
     "BARRANCABERMEJA, SANTANDER").
   - Presencia de tildes en apellidos comunes (GONZÁLEZ, MARTÍNEZ, GARCÍA,
     RODRÍGUEZ, LÓPEZ, etc.).
   - Caracteres como Ñ en apellidos (PEÑA, MUÑOZ, CASTAÑEDA) y en nombres de
     ciudades (CAÑASGORDAS, CÚCUTA).
   - Abreviaturas comunes en cédulas: D.C. (Distrito Capital), MPIO (Municipio),
     Dpto. (Departamento).
   - Estrategia: conjunto de caracteres UTF-8 completo + diccionario de ciudades
     y departamentos colombianos para validación contextual.

Responde en el siguiente orden:

1. **Análisis y razonamiento** (~2-3 párrafos explicando tus decisiones técnicas).
2. **Estructura del proyecto** (árbol de directorios).
3. **Código fuente** (archivo por archivo, con comentarios en español).
4. **Instrucciones de instalación y ejecución**.
5. **Métricas esperadas y limitaciones conocidas**.
```
