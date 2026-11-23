from __future__ import annotations

import cv2
import numpy as np


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """Preprocesa una imagen para alimentar el modelo de clasificación.

    Convierte la imagen a escala de grises cuando tiene tres canales, la
    redimensiona a 512x512 píxeles, aplica ecualización adaptativa de
    histograma (CLAHE) y normaliza sus valores a float32 en el rango
    [0.0, 1.0].

    Args:
        image: Imagen de entrada en formato uint8 con dos o tres dimensiones.

    Returns:
        Tensor float32 con forma (1, 512, 512, 1) listo para el modelo.

    Raises:
        ValueError: Si la imagen es nula, está vacía o tiene una forma no
            compatible con el preprocesamiento.
    """
    if image is None:
        raise ValueError("La imagen de entrada no puede ser nula.")

    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("La imagen de entrada está vacía.")

    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            array = array[:, :, 0]
        elif channels == 3:
            array = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        elif channels == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2GRAY)
        else:
            raise ValueError(
                f"La imagen tiene una cantidad de canales no soportada: {channels}"
            )
    elif array.ndim != 2:
        raise ValueError(f"La imagen debe tener 2 o 3 dimensiones, no {array.ndim}.")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    array = cv2.resize(array, (512, 512))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    array = clahe.apply(array)
    array = array.astype(np.float32) / np.float32(255.0)

    return array[np.newaxis, :, :, np.newaxis]
