# UAO Neumonía Grupo 1

[![CI](https://github.com/carolinadoradouao/UAO-Neumonia-Grupo1/actions/workflows/ci.yml/badge.svg)](https://github.com/carolinadoradouao/UAO-Neumonia-Grupo1/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/managed%20by-uv-black.svg)](https://docs.astral.sh/uv/)
[![Ruff](https://img.shields.io/badge/linted%20with-Ruff-black.svg)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

> Detección rápida de neumonía en radiografías de tórax (DICOM) con Deep Learning y Grad-CAM.

## Tabla de contenidos

- [Descripción](#descripción)
- [Uso de la herramienta](#uso-de-la-herramienta)
- [Estructura del proyecto](#estructura-del-proyecto)
- [Arquitectura MVC](#arquitectura-mvc-6-módulos)
- [Acerca del modelo](#acerca-del-modelo)
- [Acerca de Grad-CAM](#acerca-de-grad-cam)
- [Contribución](#contribución)
- [Licencia](#licencia)
- [Créditos](#proyecto-original-realizado-por)

## Descripción

Bienvenido a la herramienta para la detección rápida de neumonía.

Deep Learning aplicado en el procesamiento de imágenes radiográficas de tórax en formato DICOM con el fin de clasificarlas en 3 categorías diferentes:

1. Neumonía Bacteriana

2. Neumonía Viral

3. Sin Neumonía

Aplicación de una técnica de explicación llamada Grad-CAM para resaltar con un mapa de calor las regiones relevantes de la imagen de entrada.

---

## Uso de la herramienta:

### Requisitos

- [uv](https://docs.astral.sh/uv/) (gestor de entornos y dependencias)
- `make` (opcional, para usar los atajos del Makefile)
- Python 3.13 (uv lo instala automáticamente si no está disponible)

La versión de Python está fijada en `.python-version` y todas las dependencias se declaran en `pyproject.toml`.

### Instalación reproducible

```bash
# Instalar Python 3.13 y crear el entorno con todas las dependencias
uv sync

# Alternativa usando Make
make install
```

### Ejecución

```bash
# Abrir la interfaz gráfica
uv run python detector_neumonia.py

# Alternativa usando Make
make run
```

### Pruebas y calidad de código

```bash
# Ejecutar la suite de pruebas
uv run pytest

# Ejecutar el linter y verificar el formato
uv run ruff check .
uv run ruff format --check .

# Instalar los hooks de pre-commit
uv run pre-commit install

# Alternativa usando Make
make dev      # uv sync + hooks de pre-commit
make test
make check    # lint + formato + pruebas
```

### Uso de la Interfaz Gráfica

- Ingrese la cédula del paciente en la caja de texto
- Presione el botón 'Cargar Imagen', seleccione la imagen del explorador de archivos del computador (Imagenes de prueba en https://drive.google.com/drive/folders/1WOuL0wdVC6aojy8IfssHcqZ4Up14dy0g?usp=drive_link)
- Presione el botón 'Predecir' y espere unos segundos hasta que observe los resultados
- Presione el botón 'Guardar' para almacenar la información del paciente en un archivo excel con extensión .csv
- Presione el botón 'PDF' para descargar un archivo PDF con la información desplegada en la interfaz
- Presión el botón 'Borrar' si desea cargar una nueva imagen

---

## Estructura del proyecto

```
UAO-Neumonia-Grupo1/
├── detector_neumonia.py     # Punto de entrada de la aplicación (Tkinter)
├── src/                     # Módulos de la aplicación (MVC, namespace `src.*`)
│   ├── load_model.py
│   ├── read_img.py
│   ├── preprocess_img.py
│   ├── grad_cam.py
│   ├── integrator.py
│   └── ui.py
├── model/                   # Pesos entrenados (no versionados)
├── tests/                   # Pruebas automatizadas
├── docs/                    # Documentación, ADRs e investigación
├── .github/                 # PR/issue templates, CI, labels y Dependabot
├── .opencode/rules/         # Reglas para agentes (tgrep)
├── AGENTS.md                # Contexto para asistentes LLM
├── CONSTITUTION.md          # Fuente única de verdad
├── CONTRIBUTING.md          # Guía de contribución
├── CODE_OF_CONDUCT.md       # Código de conducta
├── LICENSE                  # Licencia MIT
├── .python-version          # Python 3.13 fijado con uv
├── .pre-commit-config.yaml  # Hooks de calidad
├── pyproject.toml           # Dependencias y configuración de Ruff/pytest
├── uv.lock                  # Lockfile determinista (versionado)
└── Makefile                 # Automatización de tareas
```

> El modelo `conv_MLP_84.h5` no se versiona: colócalo en `model/` antes de ejecutar la
> interfaz.

## Arquitectura MVC (6 módulos)

```
                 +-------------------------+
                 |      ui.py (Vista)      |
                 +------------+------------+
                              |
                              v
                 +-------------------------+
                 | integrator.py (Control) |
                 +--+-------+-------+----+-+
                    |       |       |    |
                    v       v       v    v
              read_img  preprocess grad_cam load_model
              (Modelo)   (Modelo)  (Modelo) (Modelo)
```

- **Vista:** `ui.py` muestra la imagen, el heatmap y los resultados.
- **Controlador:** `integrator.py` coordina los módulos y retorna solo lo necesario para la interfaz.
- **Modelo:** `read_img.py`, `preprocess_img.py`, `grad_cam.py` y `load_model.py` contienen la lógica de lectura, preprocesamiento, explicación y carga de la red neuronal.

---

## Arquitectura de archivos propuesta.

## detector_neumonia.py

Contiene el diseño de la interfaz gráfica utilizando Tkinter.

Los botones llaman métodos contenidos en otros scripts.

## src/integrator.py

Es un módulo que integra los demás scripts y retorna solamente lo necesario para ser visualizado en la interfaz gráfica.
Retorna la clase, la probabilidad y una imagen el mapa de calor generado por Grad-CAM.

## src/read_img.py

Script que lee la imagen en formato DICOM para visualizarla en la interfaz gráfica. Además, la convierte a arreglo para su preprocesamiento.

## src/preprocess_img.py

Script que recibe el arreglo proveniento de read_img.py, realiza las siguientes modificaciones:

- resize a 512x512
- conversión a escala de grises
- ecualización del histograma con CLAHE
- normalización de la imagen entre 0 y 1
- conversión del arreglo de imagen a formato de batch (tensor)

## src/load_model.py

Script que lee el archivo binario del modelo de red neuronal convolucional previamente entrenado llamado 'conv_MLP_84.h5'.

## src/grad_cam.py

Script que recibe la imagen y la procesa, carga el modelo, obtiene la predicción y la capa convolucional de interés para obtener las características relevantes de la imagen.

## src/ui.py

Módulo de la vista encargado de construir y actualizar la interfaz gráfica.

---

## Acerca del Modelo

La red neuronal convolucional implementada (CNN) es basada en el modelo implementado por F. Pasa, V.Golkov, F. Pfeifer, D. Cremers & D. Pfeifer
en su artículo Efcient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization.

Está compuesta por 5 bloques convolucionales, cada uno contiene 3 convoluciones; dos secuenciales y una conexión 'skip' que evita el desvanecimiento del gradiente a medida que se avanza en profundidad.
Con 16, 32, 48, 64 y 80 filtros de 3x3 para cada bloque respectivamente.

Después de cada bloque convolucional se encuentra una capa de max pooling y después de la última una capa de Average Pooling seguida por tres capas fully-connected (Dense) de 1024, 1024 y 3 neuronas respectivamente.

Para regularizar el modelo utilizamos 3 capas de Dropout al 20%; dos en los bloques 4 y 5 conv y otra después de la 1ra capa Dense.

## Acerca de Grad-CAM

Es una técnica utilizada para resaltar las regiones de una imagen que son importantes para la clasificación. Un mapeo de activaciones de clase para una categoría en particular indica las regiones de imagen relevantes utilizadas por la CNN para identificar esa categoría.

Grad-CAM realiza el cálculo del gradiente de la salida correspondiente a la clase a visualizar con respecto a las neuronas de una cierta capa de la CNN. Esto permite tener información de la importancia de cada neurona en el proceso de decisión de esa clase en particular. Una vez obtenidos estos pesos, se realiza una combinación lineal entre el mapa de activaciones de la capa y los pesos, de esta manera, se captura la importancia del mapa de activaciones para la clase en particular y se ve reflejado en la imagen de entrada como un mapa de calor con intensidades más altas en aquellas regiones relevantes para la red con las que clasificó la imagen en cierta categoría.

## Proyecto original realizado por:

Isabella Torres Revelo - https://github.com/isa-tr
Nicolas Diaz Salazar - https://github.com/nicolasdiazsalazar

---

## Contribución

Lee [`CONTRIBUTING.md`](./CONTRIBUTING.md) y el [`Código de conducta`](./CODE_OF_CONDUCT.md)
antes de abrir un Pull Request. La documentación técnica central está en
[`CONSTITUTION.md`](./CONSTITUTION.md).

```bash
git clone git@github.com:carolinadoradouao/UAO-Neumonia-Grupo1.git
cd UAO-Neumonia-Grupo1
make dev      # uv sync + hooks de pre-commit
make check    # lint + formato + pruebas
```

## Licencia

Este proyecto se distribuye bajo la [Licencia MIT](./LICENSE).
