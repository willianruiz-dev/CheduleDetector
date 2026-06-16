"""
Script de entrenamiento del modelo CRNN para cédulas colombianas.

Entrena el modelo de OCR personalizado con TensorFlow/Keras
usando un dataset de imágenes de texto de cédulas colombianas.

Uso:
    python train_crnn.py --dataset ./dataset/ --epochs 50 --batch_size 32
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.ocr.text_recognizer import ALPHABET, BLANK_INDEX, CRNNModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenar modelo CRNN para OCR de cédulas colombianas"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        required=True,
        help="Ruta a la carpeta del dataset.",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Número de épocas (default: 50).",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=32,
        help="Tamaño del batch (default: 32).",
    )
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje (default: 0.001).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="modelo_crnn_cedulas.h5",
        help="Ruta para guardar el modelo entrenado.",
    )
    parser.add_argument(
        "--validation_split", "-v",
        type=float,
        default=0.2,
        help="Fracción del dataset para validación (default: 0.2).",
    )
    return parser.parse_args()


def load_dataset(dataset_path: str):
    """
    Carga el dataset de entrenamiento.

    Estructura esperada:
        dataset/
        ├── train/
        │   ├── images/          # imágenes de regiones de texto
        │   └── labels.txt       # imagen.jpg|texto_referencia
        └── val/  (opcional)
            ├── images/
            └── labels.txt

    labels.txt formato:
        imagen_001.jpg|GARCIA
        imagen_002.jpg|1.234.567.890
        ...
    """
    import cv2

    dataset_path = Path(dataset_path)
    images = []
    labels = []

    for split in ["train", "val"]:
        split_path = dataset_path / split
        if not split_path.exists():
            continue

        labels_file = split_path / "labels.txt"
        images_dir = split_path / "images"

        if not labels_file.exists() or not images_dir.exists():
            print(f"[Dataset] Saltando {split}: no encontrado.")
            continue

        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue

                img_name, text = line.split("|", 1)
                img_path = images_dir / img_name

                if not img_path.exists():
                    continue

                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                images.append(img)
                labels.append(text.strip().upper())

    print(f"[Dataset] Cargadas {len(images)} imágenes con etiquetas.")
    return images, labels


def encode_labels(labels: list):
    """Codifica etiquetas de texto a secuencias de índices para CTC."""
    max_len = max(len(label) for label in labels)

    encoded = []
    for label in labels:
        indices = []
        for char in label:
            idx = 0  # default: blank
            if char in ALPHABET:
                idx = ALPHABET.index(char)
            indices.append(idx)

        # Padding al máximo largo
        indices += [BLANK_INDEX] * (max_len - len(label))
        encoded.append(indices)

    return np.array(encoded), max_len


def data_generator(images, labels_encoded, input_length, label_length, batch_size):
    """
    Generador de datos para entrenamiento.

    Yields:
        (batch_images, batch_labels, batch_input_length, batch_label_length)
    """
    import tensorflow as tf

    num_samples = len(images)
    indices = np.arange(num_samples)

    while True:
        np.random.shuffle(indices)

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]

            batch_images = []
            batch_labels = []

            for idx in batch_indices:
                img = images[idx]
                label = labels_encoded[idx]

                # Preprocesar imagen (redimensionar + normalizar)
                h, w = img.shape
                ratio = 32 / h
                new_w = max(int(w * ratio), 4)
                img_resized = tf.image.resize(
                    img[..., np.newaxis], (32, new_w)
                )
                img_resized = tf.image.pad_to_bounding_box(
                    img_resized, 0, 0, 32, 256
                )
                img_resized = tf.cast(img_resized, tf.float32) / 255.0

                batch_images.append(img_resized)
                batch_labels.append(label)

            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)

            # Calcular input_lengths y label_lengths para CTC
            batch_input_length = np.full(
                (len(batch_indices),), input_length, dtype=np.int32
            )
            batch_label_length = np.full(
                (len(batch_indices),), label_length, dtype=np.int32
            )

            yield (
                {"image_input": batch_images},
                {
                    "softmax_out": batch_labels,
                    "ctc": np.zeros((len(batch_indices),), dtype=np.float32),
                },
                batch_input_length,
                batch_label_length,
            )


def main():
    args = parse_args()

    print("=" * 60)
    print("  ENTRENAMIENTO CRNN - CÉDULAS COLOMBIANAS")
    print("=" * 60)

    # ── Cargar dataset ─────────────────────────────────────────
    print(f"\n📂 Cargando dataset: {args.dataset}")
    images, labels = load_dataset(args.dataset)

    if not images:
        print("❌ No se encontraron datos de entrenamiento.")
        print("\nEstructura esperada del dataset:")
        print("  dataset/train/images/*.jpg")
        print("  dataset/train/labels.txt (formato: imagen.jpg|texto)")
        sys.exit(1)

    # ── Codificar etiquetas ────────────────────────────────────
    print("[Codificación] Convirtiendo etiquetas a secuencias...")
    labels_encoded, max_label_len = encode_labels(labels)
    print(f"[Codificación] Longitud máxima de etiqueta: {max_label_len}")

    # ── Construir modelo ───────────────────────────────────────
    print("\n🧠 Construyendo modelo CRNN...")
    model_wrapper = CRNNModel()

    # ── Compilar con CTC Loss ──────────────────────────────────
    import tensorflow as tf

    # Añadir capa CTC loss al modelo
    labels_input = tf.keras.layers.Input(
        shape=(max_label_len,), dtype="int32", name="labels"
    )
    input_length = tf.keras.layers.Input(shape=(1,), dtype="int32", name="input_length")
    label_length = tf.keras.layers.Input(
        shape=(1,), dtype="int32", name="label_length"
    )

    ctc_loss = tf.keras.layers.Lambda(
        function=lambda args: tf.keras.backend.ctc_batch_cost(
            args[0], args[1], args[2], args[3]
        ),
        output_shape=(1,),
        name="ctc",
    )([labels_input, model_wrapper.model.output, input_length, label_length])

    train_model = tf.keras.Model(
        inputs=[
            model_wrapper.model.input,
            labels_input,
            input_length,
            label_length,
        ],
        outputs=ctc_loss,
    )

    # Compilar
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
    train_model.compile(optimizer=optimizer)

    print("[Modelo] Arquitectura:")
    model_wrapper.summary()

    # ── Entrenar ────────────────────────────────────────────────
    print(f"\n🚀 Iniciando entrenamiento ({args.epochs} épocas)...")

    # Calcular input_length para CTC
    # Después de 4 MaxPool (2,2), (2,2), (2,1), (2,1):
    # input_height=32 -> 32/2/2/2/2 = 2
    # input_width=256 -> 256/2/2/1/1 = 64 timesteps
    input_len = 64

    num_samples = len(images)
    steps_per_epoch = max(1, num_samples // args.batch_size)
    val_steps = max(1, int(steps_per_epoch * args.validation_split))

    # Separar train/val
    split_idx = int(num_samples * (1 - args.validation_split))
    train_images = images[:split_idx]
    train_labels = labels_encoded[:split_idx]
    val_images = images[split_idx:]
    val_labels = labels_encoded[split_idx:]

    # Preparar datos para el formato que espera el modelo
    def prepare_data(img_list, lab_list):
        import cv2
        batch_imgs = []
        for img in img_list:
            h, w = img.shape
            ratio = 32 / h
            new_w = max(int(w * ratio), 4)
            resized = tf.image.resize(img[..., np.newaxis], (32, new_w))
            resized = tf.image.pad_to_bounding_box(resized, 0, 0, 32, 256)
            batch_imgs.append(resized.numpy())
        return np.array(batch_imgs) / 255.0

    X_train = prepare_data(train_images, None)
    X_val = prepare_data(val_images, None) if val_images else None

    train_labels_padded = train_labels
    val_labels_padded = val_labels if val_images.size > 0 else None

    train_input_len = np.full((len(X_train), 1), input_len, dtype=np.float32)
    train_label_len = np.full((len(X_train), 1), max_label_len, dtype=np.float32)

    # Entrenamiento
    history = train_model.fit(
        x=[X_train, train_labels_padded, train_input_len, train_label_len],
        y=np.zeros(len(X_train)),
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_data=(
            [X_val, val_labels_padded,
             np.full((len(X_val), 1), input_len, dtype=np.float32),
             np.full((len(X_val), 1), max_label_len, dtype=np.float32)],
            np.zeros(len(X_val)),
        ) if X_val is not None and len(X_val) > 0 else None,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None and len(X_val) > 0 else "loss",
                patience=10,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None and len(X_val) > 0 else "loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
            ),
        ],
    )

    # ── Guardar modelo ─────────────────────────────────────────
    print(f"\n💾 Guardando modelo: {args.output}")
    model_wrapper.model.save(args.output)

    print("\n✅ Entrenamiento completado.")
    print(f"   Modelo guardado en: {args.output}")
    print(f"   Úsalo con: python main.py --imagen cedula.jpg")


if __name__ == "__main__":
    main()
