"""Orquestador del flujo de inferencia de la herramienta de neumonía.

Este módulo es el único punto que conoce el orden de ejecución de los
módulos de lectura, preprocesamiento, carga del modelo y Grad-CAM. La
interfaz gráfica consume únicamente las funciones públicas ``load_image`` y
``predict``.

Contrato de los módulos integrados:

- ``read_img.load_image(path)`` devuelve un ``np.ndarray`` uint8.
- ``preprocess_img.preprocess_image(array)`` devuelve un batch para el modelo.
- ``load_model.get_model()`` devuelve la red neuronal entrenada.
- ``grad_cam.generate_gradcam(batch, model)`` devuelve el mapa de calor 2D.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src import grad_cam, load_model, preprocess_img, read_img

LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}


def load_image(path):
    """Carga una imagen DICOM o convencional para mostrarla y procesarla.

    Args:
        path: Ruta del archivo de imagen seleccionado por el usuario.

    Returns:
        Una tupla ``(array, imagen)`` con el arreglo NumPy de la imagen y una
        imagen PIL lista para visualizarse en la interfaz gráfica.
    """
    array = read_img.load_image(path)
    return array, Image.fromarray(array)


def predict(array):
    """Ejecuta el flujo completo de predicción y explicación Grad-CAM.

    Args:
        array: Imagen de entrada en el formato devuelto por ``load_image``.

    Returns:
        Una tupla ``(etiqueta, probabilidad, heatmap)`` donde ``etiqueta`` es
        la clase predicha, ``probabilidad`` está expresada en porcentaje y
        ``heatmap`` es el mapa de calor 2D ``float32`` en el rango [0.0, 1.0].
    """
    batch = preprocess_img.preprocess_image(array)
    model = load_model.get_model()
    predictions = model.predict(batch)
    index = int(np.argmax(predictions[0]))
    probability = float(np.max(predictions[0])) * 100
    label = LABELS.get(index, "desconocida")
    heatmap = grad_cam.generate_gradcam(batch, model)
    return label, probability, heatmap
