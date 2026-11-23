from __future__ import annotations

import numpy as np
import tensorflow as tf

from src.preprocess_img import preprocess_image


def generate_gradcam(
    model,
    tensor_input: np.ndarray,
    layer_name: str | None = None,
) -> np.ndarray:
    """Genera un mapa de calor Grad-CAM para la clase predicha por el modelo.

    Calcula los gradientes de la clase con mayor probabilidad respecto a las
    activaciones de la última capa convolucional (o de ``layer_name`` si se
    indica) mediante ``tf.GradientTape``. El mapa resultante se pondera con los
    gradientes globales promediados, se normaliza a [0.0, 1.0] y se redimensiona
    a 512x512 píxeles. También acepta el orden heredado ``(array, model)``.

    Args:
        model: Modelo Keras construido con entradas y salidas simbólicas.
        tensor_input: Tensor de entrada con forma (lote, alto, ancho, canales).
        layer_name: Nombre opcional de la capa convolucional a utilizar.

    Returns:
        Mapa de calor float32 con forma (512, 512) y valores en [0.0, 1.0].
    """
    legacy_order = False
    if (isinstance(model, np.ndarray) or hasattr(model, "numpy")) and hasattr(
        tensor_input, "get_layer"
    ):
        model, tensor_input = tensor_input, model
        legacy_order = True

    array = np.asarray(tensor_input)
    if legacy_order and array.dtype == np.uint8 and array.ndim in (2, 3):
        array = preprocess_image(array)

    inputs = tf.convert_to_tensor(array, dtype=tf.float32)
    if inputs.shape.rank == 2:
        inputs = inputs[..., tf.newaxis, tf.newaxis]
    elif inputs.shape.rank == 3:
        inputs = inputs[tf.newaxis, ...]
    elif inputs.shape.rank != 4:
        raise ValueError(
            f"La entrada debe tener 2, 3 o 4 dimensiones, no {inputs.shape.rank}."
        )

    if layer_name is not None:
        conv_layer = model.get_layer(layer_name)
    else:
        conv_types = tuple(
            layer_type
            for layer_type in (
                getattr(tf.keras.layers, name, None)
                for name in (
                    "Conv2D",
                    "Conv2DTranspose",
                    "SeparableConv2D",
                    "DepthwiseConv2D",
                )
            )
            if layer_type is not None
        )
        conv_layer = None
        fallback_layer = None
        for layer in reversed(model.layers):
            output_shape = getattr(getattr(layer, "output", None), "shape", None)
            rank = getattr(output_shape, "rank", None)
            if rank is None and output_shape is not None:
                rank = len(output_shape)
            if fallback_layer is None and rank == 4:
                fallback_layer = layer
            if isinstance(layer, conv_types):
                conv_layer = layer
                break
        if conv_layer is None:
            conv_layer = fallback_layer
        if conv_layer is None:
            raise ValueError("El modelo no contiene una capa convolucional 2D.")

    cam_model = tf.keras.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        tape.watch(inputs)
        feature_maps, predictions = cam_model(inputs, training=False)
        predicted_index = tf.argmax(predictions[0], output_type=tf.int32)
        class_score = tf.gather(predictions, predicted_index, axis=1)
        if class_score.shape.rank == 2:
            class_score = tf.squeeze(class_score, axis=1)

    gradients = tape.gradient(class_score, feature_maps)
    if gradients is None:
        raise RuntimeError("No se pudieron calcular los gradientes Grad-CAM.")

    weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(feature_maps[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, tf.zeros_like(heatmap)).numpy()

    if heatmap.ndim == 3:
        heatmap = heatmap[..., 0]

    heatmap = tf.image.resize(
        heatmap[..., np.newaxis], (512, 512), method="bilinear"
    ).numpy()[..., 0]
    maximum = float(np.max(heatmap))
    if maximum > 0.0:
        heatmap = heatmap / maximum

    return np.clip(heatmap, 0.0, 1.0).astype(np.float32)
