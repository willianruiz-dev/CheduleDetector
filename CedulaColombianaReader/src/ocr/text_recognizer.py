"""
Modelo CRNN (CNN + RNN + CTC) para OCR de cédulas colombianas.

Implementa una red neuronal convolucional recurrente con CTC Loss
para reconocimiento de texto en imágenes de documentos de identidad.
"""

import numpy as np
from typing import Optional, List, Tuple

# ── Constantes del modelo ──────────────────────────────────────
# Caracteres usados en cédulas colombianas
# Incluye mayúsculas, dígitos, tildes, ñ y símbolos comunes
ALPHABET = (
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "ÁÉÍÓÚÜÑ"
    " .,-/:()"
    " "  # espacio
)

# Carácter en blanco para CTC
BLANK_INDEX = len(ALPHABET)

# Mapeo carácter <-> índice
CHAR_TO_IDX = {char: idx for idx, char in enumerate(ALPHABET)}
IDX_TO_CHAR = {idx: char for idx, char in enumerate(ALPHABET)}


class CRNNModel:
    """
    Modelo CRNN para OCR de cédulas colombianas.

    Arquitectura:
    ┌───────────────────────────────────────────┐
    │  Input (H, W, 1)                          │
    │    ↓                                      │
    │  Conv2D 64@3x3 + BatchNorm + ReLU         │
    │  MaxPool 2x2                              │
    │    ↓                                      │
    │  Conv2D 128@3x3 + BatchNorm + ReLU        │
    │  MaxPool 2x2                              │
    │    ↓                                      │
    │  Conv2D 256@3x3 + BatchNorm + ReLU        │
    │  Conv2D 256@3x3 + BatchNorm + ReLU        │
    │  MaxPool 2x1 (solo reducir altura)        │
    │    ↓                                      │
    │  Conv2D 512@3x3 + BatchNorm + ReLU        │
    │  Conv2D 512@3x3 + BatchNorm + ReLU        │
    │  MaxPool 2x1                              │
    │    ↓                                      │
    │  Reshape → (W/4, 512)                     │
    │    ↓                                      │
    │  BiLSTM 256 (2 capas)                     │
    │    ↓                                      │
    │  Dense (num_clases) + Softmax             │
    │    ↓                                      │
    │  CTC Decoder                              │
    │    ↓                                      │
    │  Output: texto reconocido                 │
    └───────────────────────────────────────────┘
    """

    def __init__(
        self,
        input_height: int = 32,
        input_width: int = 256,
        num_channels: int = 1,
        num_classes: int = None,
        model_path: Optional[str] = None,
    ):
        """
        Args:
            input_height: Altura de entrada (debe ser múltiplo de 4).
            input_width: Ancho máximo de entrada.
            num_channels: Canales (1 = escala de grises).
            num_classes: Número de clases de salida (len(alphabet) + 1 para CTC).
            model_path: Ruta a un modelo pre-entrenado (.h5 o SavedModel).
        """
        self.input_height = input_height
        self.input_width = input_width
        self.num_channels = num_channels
        self.num_classes = num_classes or len(ALPHABET) + 1

        self.model = None
        self._built = False

        if model_path:
            self.load_model(model_path)
        else:
            self._build_model()

    def _build_model(self) -> None:
        """
        Construye la arquitectura CRNN con TensorFlow/Keras.
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow es requerido. Instálelo con: pip install tensorflow"
            )

        input_layer = tf.keras.layers.Input(
            shape=(self.input_height, self.input_width, self.num_channels),
            name="image_input",
        )

        x = input_layer

        # ── Bloque convolucional 1 ──
        x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", name="conv1")(x)
        x = tf.keras.layers.BatchNormalization(name="bn1")(x)
        x = tf.keras.layers.Activation("relu", name="relu1")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

        # ── Bloque convolucional 2 ──
        x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", name="conv2")(x)
        x = tf.keras.layers.BatchNormalization(name="bn2")(x)
        x = tf.keras.layers.Activation("relu", name="relu2")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

        # ── Bloque convolucional 3 ──
        x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", name="conv3")(x)
        x = tf.keras.layers.BatchNormalization(name="bn3")(x)
        x = tf.keras.layers.Activation("relu", name="relu3")(x)
        x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
        x = tf.keras.layers.BatchNormalization(name="bn4")(x)
        x = tf.keras.layers.Activation("relu", name="relu4")(x)
        x = tf.keras.layers.MaxPooling2D((2, 1), name="pool3")(x)

        # ── Bloque convolucional 4 ──
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", name="conv5")(x)
        x = tf.keras.layers.BatchNormalization(name="bn5")(x)
        x = tf.keras.layers.Activation("relu", name="relu5")(x)
        x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", name="conv6")(x)
        x = tf.keras.layers.BatchNormalization(name="bn6")(x)
        x = tf.keras.layers.Activation("relu", name="relu6")(x)
        x = tf.keras.layers.MaxPooling2D((2, 1), name="pool4")(x)

        # ── Reshape para la parte recurrente ──
        # Eliminar dimensión de altura (debe ser 1 después de los pools)
        x = tf.keras.layers.Reshape((-1, 512), name="reshape")(x)

        # ── Capas recurrentes BiLSTM ──
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.25),
            name="bilstm1",
        )(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.25),
            name="bilstm2",
        )(x)

        # ── Capa de salida ──
        x = tf.keras.layers.Dense(self.num_classes, name="dense_out")(x)
        output = tf.keras.layers.Activation("softmax", name="softmax_out")(x)

        self.model = tf.keras.Model(inputs=input_layer, outputs=output, name="CRNN")
        self._built = True

    def predict(
        self, image: np.ndarray
    ) -> Tuple[str, float]:
        """
        Reconoce texto en una imagen.

        Args:
            image: Imagen en escala de grises (H, W) o (H, W, 1).

        Returns:
            Tuple con el texto reconocido y la confianza promedio (0-1).
        """
        if self.model is None:
            raise RuntimeError("El modelo no está cargado.")

        import tensorflow as tf

        # Preprocesar
        processed = self._preprocess_image(image)

        # Inferencia
        predictions = self.model.predict(processed, verbose=0)

        # Decodificar con CTC greedy decoder
        text, confidence = self._ctc_decode(predictions[0])

        return text, confidence

    def predict_batch(
        self, images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """
        Reconoce texto en un lote de imágenes.

        Args:
            images: Lista de imágenes en escala de grises.

        Returns:
            Lista de (texto, confianza) para cada imagen.
        """
        import tensorflow as tf

        batch = []
        for img in images:
            processed = self._preprocess_image(img)
            batch.append(processed[0])

        batch = np.array(batch)

        predictions = self.model.predict(batch, verbose=0)

        results = []
        for pred in predictions:
            text, confidence = self._ctc_decode(pred)
            results.append((text, confidence))

        return results

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen para el modelo CRNN.

        - Asegurar que sea escala de grises.
        - Redimensionar manteniendo relación de aspecto.
        - Normalizar a [0, 1].
        """
        # Asegurar escala de grises
        if len(image.shape) == 3 and image.shape[2] == 3:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        # Redimensionar altura al input_height, ajustar ancho proporcionalmente
        h, w = image.shape[:2]
        ratio = self.input_height / h
        new_w = int(w * ratio)
        new_w = max(new_w, 4)  # mínimo 4 píxeles de ancho

        import tensorflow as tf
        image = tf.image.resize(image, (self.input_height, new_w))
        image = tf.image.pad_to_bounding_box(
            image, 0, 0, self.input_height, self.input_width
        )

        # Normalizar
        image = tf.cast(image, tf.float32) / 255.0

        # Añadir batch dimension
        image = tf.expand_dims(image, axis=0)

        return image.numpy()

    def _ctc_decode(
        self, predictions: np.ndarray
    ) -> Tuple[str, float]:
        """
        Decodifica la salida del modelo CRNN usando CTC greedy decoder.

        Args:
            predictions: Array (time_steps, num_classes) con probabilidades.

        Returns:
            Tuple (texto, confianza_promedio).
        """
        # CTC greedy: tomar el índice de mayor probabilidad en cada paso
        best_indices = np.argmax(predictions, axis=-1)

        # Colapsar repetidos y eliminar blanks
        decoded = []
        confidence_sum = 0.0
        count = 0

        prev_idx = -1
        for t, idx in enumerate(best_indices):
            if idx != BLANK_INDEX and idx != prev_idx:
                if idx in IDX_TO_CHAR:
                    decoded.append(IDX_TO_CHAR[idx])
                confidence_sum += predictions[t, idx]
                count += 1
            prev_idx = idx

        text = "".join(decoded).strip()
        confidence = confidence_sum / max(count, 1)

        return text, float(confidence)

    def load_model(self, model_path: str) -> None:
        """Carga un modelo pre-entrenado desde disco."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(model_path)
            self._built = True
            print(f"[CRNN] Modelo cargado: {model_path}")
        except Exception as e:
            print(f"[CRNN] Error cargando modelo: {e}")
            self._build_model()

    def save_model(self, model_path: str) -> None:
        """Guarda el modelo a disco."""
        if self.model is None:
            raise RuntimeError("No hay modelo para guardar.")
        self.model.save(model_path)
        print(f"[CRNN] Modelo guardado: {model_path}")

    def summary(self) -> None:
        """Imprime el resumen de la arquitectura del modelo."""
        if self.model:
            self.model.summary()


# ── Función de pérdida CTC para entrenamiento ──────────────────

def ctc_loss_lambda_func(args):
    """
    Función de pérdida CTC para Keras.

    Uso:
        loss = Lambda(ctc_loss_lambda_func, output_shape=(1,),
                      name='ctc')([output, labels, input_length, label_length])
    """
    import tensorflow as tf
    y_pred, labels, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(
        labels, y_pred, input_length, label_length
    )
