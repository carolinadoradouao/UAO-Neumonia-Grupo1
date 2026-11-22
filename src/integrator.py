"""Orquestador del flujo de inferencia de la herramienta de neumonía.

Este módulo es el único punto que conoce el orden de ejecución de los
módulos de lectura, preprocesamiento, carga del modelo y Grad-CAM. La
interfaz gráfica consume únicamente las funciones públicas ``load_image`` y
``predict``.

Contrato esperado de los módulos integrados:

- ``read_img.load_image(path)`` devuelve ``(array, imagen_pil)``.
- ``preprocess_img.preprocess_image(array)`` devuelve un batch para el modelo.
- ``load_model.get_model()`` devuelve la red neuronal entrenada.
- ``grad_cam.generate_gradcam(array, model)`` devuelve la imagen con el mapa
  de calor superpuesto.
"""

from __future__ import annotations

import numpy as np

from src import grad_cam, load_model, preprocess_img, read_img

LABELS = {0: "bacteriana", 1: "normal", 2: "viral"}


def load_image(path):
    """Carga una imagen DICOM o convencional para mostrarla y procesarla.

    Args:
        path: Ruta del archivo de imagen seleccionado por el usuario.

    Returns:
        Una tupla ``(array, imagen)`` con el arreglo de la imagen y la imagen
        PIL lista para visualizarse en la interfaz gráfica.
    """
    return read_img.load_image(path)


def predict(array):
    """Ejecuta el flujo completo de predicción y explicación Grad-CAM.

    Args:
        array: Imagen de entrada en el formato devuelto por ``load_image``.

    Returns:
        Una tupla ``(etiqueta, probabilidad, heatmap)`` donde ``etiqueta`` es
        la clase predicha, ``probabilidad`` está expresada en porcentaje y
        ``heatmap`` es la imagen con el mapa de calor superpuesto.
    """
    batch = preprocess_img.preprocess_image(array)
    model = load_model.get_model()
    predictions = model.predict(batch)
    index = int(np.argmax(predictions[0]))
    probability = float(np.max(predictions[0])) * 100
    label = LABELS.get(index, "desconocida")
    heatmap = grad_cam.generate_gradcam(array, model)
    return label, probability, heatmap
