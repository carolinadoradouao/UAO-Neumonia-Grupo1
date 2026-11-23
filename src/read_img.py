from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pydicom


def load_image(filepath: str) -> np.ndarray:
    """Carga una imagen DICOM o una imagen convencional desde disco.

    Los archivos DICOM (``.dcm``) se leen con pydicom y su ``pixel_array`` se
    normaliza al rango uint8 [0, 255]. Los archivos ``.jpeg``, ``.jpg`` y
    ``.png`` se leen con OpenCV y se convierten de BGR a RGB.

    Args:
        filepath: Ruta del archivo de imagen que se desea cargar.

    Returns:
        Arreglo NumPy en formato uint8 con la imagen lista para preprocesarse.

    Raises:
        ValueError: Si el archivo no existe, su extensión no está soportada,
            su contenido es inválido o no se puede decodificar.
    """
    if not filepath:
        raise ValueError("La ruta de la imagen no puede estar vacía.")

    path = Path(filepath)
    if not path.is_file():
        raise ValueError(f"El archivo no existe o no es accesible: {filepath}")

    extension = path.suffix.lower()

    if extension == ".dcm":
        try:
            dataset = pydicom.dcmread(str(path))
            pixels = np.asarray(dataset.pixel_array, dtype=np.float64)
        except Exception as error:
            raise ValueError(f"No se pudo leer el archivo DICOM: {filepath}") from error

        if pixels.size == 0:
            raise ValueError(f"El archivo DICOM no contiene píxeles: {filepath}")

        minimum = float(pixels.min())
        maximum = float(pixels.max())
        if maximum > minimum:
            pixels = (pixels - minimum) / (maximum - minimum) * 255.0
        else:
            pixels = np.zeros_like(pixels)

        return np.clip(pixels, 0, 255).astype(np.uint8)

    if extension in {".jpeg", ".jpg", ".png"}:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"No se pudo leer la imagen: {filepath}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    raise ValueError(f"Formato de imagen no soportado: {path.suffix or filepath}")
