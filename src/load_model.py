from __future__ import annotations

from pathlib import Path

import tensorflow as tf

_model = None
_MODEL = None
_model_path = None


def get_model(model_path: str = "model/conv_MLP_84.h5"):
    """Obtiene la red neuronal Keras cargada una única vez en memoria.

    La primera llamada valida que ``model_path`` exista en disco y carga el
    modelo con TensorFlow/Keras; las llamadas posteriores con la misma ruta
    devuelven la instancia ya creada. Si el archivo no existe, se lanza
    ``FileNotFoundError`` con la ruta solicitada.

    Args:
        model_path: Ruta al archivo ``.h5`` del modelo entrenado.

    Returns:
        El modelo Keras compartido por todas las llamadas a esta función.
    """
    global _model, _MODEL, _model_path

    if _model is not None and _MODEL is _model and _model_path == model_path:
        return _model

    _model = None
    _MODEL = None
    _model_path = None

    if not Path(model_path).is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo del modelo; no existe en disco: {model_path}"
        )

    _model = tf.keras.models.load_model(model_path, compile=False)
    _MODEL = _model
    _model_path = model_path
    return _model
