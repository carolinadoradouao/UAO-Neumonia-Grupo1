# CONSTITUTION — UAO Neumonía Grupo 1

> **Verdad central del proyecto.** Este documento define el propósito, el stack, la
> estructura, las convenciones y el flujo de trabajo de `UAO Neumonía Grupo 1`. Si algo aquí
> contradice al código, el código manda **y este archivo se actualiza**.

> **Directiva de mantenimiento:** cada vez que se realicen cambios significativos (nuevas
> funcionalidades, cambios arquitectónicos, adición de clases/módulos o ajustes de
> tecnologías), **se debe actualizar este archivo en el mismo commit** para mantener la
> verdad central del proyecto bien documentada.

---

## 0. Identidad del proyecto

| Campo | Valor |
|-------|-------|
| **Nombre** | UAO Neumonía Grupo 1 |
| **Descripción** | Detección rápida de neumonía con Deep Learning y Grad-CAM |
| **Repositorio** | https://github.com/carolinadoradouao/UAO-Neumonia-Grupo1 |
| **Autoría** | UAO Neumonía Grupo 1 — roberto_ale.baez@uaovirtual.edu.co |
| **Licencia** | MIT |
| **Rama principal** | `main` |
| **Estado** | En construcción |

## 1. Propósito y no-propósito

### Propósito

Clasificar radiografías de tórax en formato DICOM en tres categorías (neumonía bacteriana,
neumonía viral o sin neumonía) y explicar la decisión del modelo con mapas de calor Grad-CAM
a través de una interfaz de escritorio.

### No-propósito

No es un dispositivo médico ni sustituye el diagnóstico de un profesional. No entrena ni
redistribuye modelos ni datos clínicos reales dentro del repositorio.

## 2. Alcance

### Dentro del alcance

- Lectura y preprocesamiento de imágenes DICOM.
- Inferencia con la CNN preentrenada (`conv_MLP_84.h5`).
- Generación de mapas Grad-CAM para explicar la predicción.
- Interfaz gráfica (Tkinter) para cargar imágenes, predecir, guardar CSV y exportar PDF.

### Fuera del alcance (por ahora)

- Entrenamiento o ajuste fino del modelo.
- Servicio web o API remota.
- Almacenamiento de historias clínicas reales.
- Despliegue clínico o validación regulatoria.

## 3. Stack técnico

| Categoría | Tecnología |
|-----------|------------|
| **Lenguaje** | Python 3.13+ |
| **Gestor de paquetes y entorno** | [`uv`](https://docs.astral.sh/uv/) |
| **Modo de proyecto** | Aplicación (módulos planos en `src/`, sin paquete publicable) |
| **Interfaz** | Tkinter |
| **Deep Learning** | TensorFlow / Keras |
| **Visión por computador** | OpenCV, Pillow, pydicom |
| **Testing** | [`pytest`](https://docs.pytest.org/) |
| **Linting / Formato** | [Ruff](https://docs.astral.sh/ruff/) |
| **Pre-commit** | [pre-commit](https://pre-commit.com) |
| **CI/CD** | [GitHub Actions](.github/workflows/ci.yml) |
| **Dependencias automáticas** | [Dependabot](.github/dependabot.yml) |

> **Regla dura:** toda gestión de dependencias y entornos pasa por `uv`. Está
> **prohibido** `pip` y `requirements.txt`.

## 4. Estructura del repositorio

```text
UAO-Neumonia-Grupo1/
├── detector_neumonia.py     # punto de entrada (Tkinter)
├── src/                     # módulos de la aplicación (namespace package `src.*`)
│   ├── load_model.py
│   ├── read_img.py
│   ├── preprocess_img.py
│   ├── grad_cam.py
│   ├── integrator.py
│   └── ui.py
├── model/                   # pesos entrenados (no versionados)
├── tests/                   # pruebas (pytest)
├── docs/                    # documentación, ADRs e investigación
├── .github/                 # gobernanza (PR/issue templates, CI, Dependabot, labels)
├── .opencode/rules/         # reglas para agentes (tgrep)
├── AGENTS.md                # contexto para agentes LLM
├── CONSTITUTION.md          # este archivo — verdad central
├── README.md                # puerta de entrada pública
├── CONTRIBUTING.md          # guía de contribución
├── CODE_OF_CONDUCT.md       # código de conducta
├── Makefile                 # automatización de tareas
├── pyproject.toml           # metadatos y dependencias (uv)
├── uv.lock                  # lockfile determinista (versionado)
├── .python-version          # versión de Python fijada
└── .pre-commit-config.yaml  # hooks de calidad
```

> **Nota sobre `src/`:** los módulos se importan como `src.<modulo>` (namespace package) y
> así los consumen los tests. Por restricción del proyecto, el contenido de `src/` **no se
> reorganiza** en `src/<paquete>/`; se declara explícitamente el modo aplicación.

### Contratos entre módulos

| Módulo | Entrada | Salida |
|--------|---------|--------|
| `read_img.load_image(path)` | Ruta de archivo | `np.ndarray` uint8 |
| `preprocess_img.preprocess_image(array)` | `np.ndarray` | batch `(1, 512, 512, 1)` float32 |
| `load_model.get_model()` | — | modelo Keras (singleton) |
| `grad_cam.generate_gradcam(batch, model)` | batch + modelo | mapa de calor `(512, 512)` float32 en [0,1] |
| `integrator.load_image(path)` | Ruta de archivo | `(np.ndarray, PIL.Image)` |
| `integrator.predict(array)` | `np.ndarray` | `(etiqueta, probabilidad, heatmap 2D)` |
| `ui.App` | — | compone el heatmap sobre la imagen y lo muestra |

> `detector_neumonia.py` es un punto de entrada fino que delega en `src.ui.main`.

## 5. Convenciones de código

- **Estilo:** [PEP 8](https://peps.python.org/pep-0008/) y [PEP 257](https://peps.python.org/pep-0257/),
  impuesto por **Ruff** (comillas dobles).
- **Longitud de línea:** se mantiene el valor por defecto de Ruff (88) para no alterar el
  código existente de `src/`. La migración a 100 queda como tarea pendiente (ver §10).
- **Tipado:** anotaciones en funciones y firmas públicas.
- **Nomenclatura:** `snake_case` (funciones/variables), `PascalCase` (clases),
  `UPPER_CASE` (constantes).
- **Docstrings:** en módulos, clases y funciones públicas.
- **Pruebas:** `pytest` con el patrón **AAA** (*Arrange · Act · Assert*).
- **Idioma:** UI, comentarios y commits en español; identificadores de código en inglés.
- **Reglas propias:**
  - `src/` es de solo lectura para tareas de estandarización: se documenta, no se reorganiza.
  - El punto de entrada es `detector_neumonia.py`; la lógica reutilizable vive en `src/`.

## 6. Calidad y automatización

| Herramienta | Rol | Comando |
|-------------|-----|---------|
| `uv` | Entornos y dependencias | `uv sync` |
| Ruff | Lint + formato | `make lint` / `make format` |
| `pytest` | Pruebas | `make test` |
| `pre-commit` | Hooks antes de cada commit | `make pre-commit` |
| GitHub Actions | CI (pre-commit + pruebas) | `.github/workflows/ci.yml` |

- Ningún commit se salta los hooks (`--no-verify` está prohibido salvo causa justificada y documentada).
- La **CI debe estar en verde** antes de fusionar un Pull Request.
- Toda dependencia nueva entra con `uv add` / `uv add --dev` (nunca editando `uv.lock` a mano).

## 7. Repositorio y Git

- **Remoto (SSH):** `git@github.com:carolinadoradouao/UAO-Neumonia-Grupo1.git`
- **Remoto (HTTPS):** `https://github.com/carolinadoradouao/UAO-Neumonia-Grupo1.git`
- **Rama principal:** `main`
- **Estrategia de ramas:** `feat/...`, `fix/...`, `docs/...`, `refactor/...`, `test/...`,
  `chore/...`; integración vía Pull Request.
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/) —
  `feat:`, `fix:`, `docs:`, `refactor:`, `style:`, `test:`, `chore:`.
- **Identidad git:** configuración local de la máquina (ver §11).

## 8. Flujo diario (commit + push)

```bash
# 1. Ver cambios
git status

# 2. Verificación completa (lint + formato + pruebas)
make check

# 3. Agregar y commitear (los hooks se ejecutan solos)
git add -A
git commit -m "tipo: mensaje descriptivo"

# 4. Subir a GitHub
git push
```

## 9. Historial de decisiones técnicas

| Fecha | Decisión | Motivo |
|-------|----------|--------|
| 2026-11-24 | Adoptar `uv` como gestor único de entornos y dependencias | Reproducibilidad; se elimina `requirements.txt` |
| 2026-11-24 | Adoptar Ruff (lint + formato) y pre-commit | Calidad automatizada antes de cada commit |
| 2026-11-24 | Adoptar Conventional Commits | Historial legible y automatizable |
| 2026-11-24 | Adoptar licencia MIT y `CONSTITUTION.md` como fuente única de verdad | Claridad legal y operativa |
| 2026-11-24 | Declarar el proyecto como aplicación (módulos planos en `src/`) | No se puede modificar `src/`; los tests importan `src.*` |
| 2026-11-24 | Mantener `line-length = 88` en Ruff | Evitar reformatear código existente de `src/` |
| 2026-11-24 | Alinear contratos entre módulos | `integrator.load_image` devuelve `(array, PIL)` y `integrator.predict` entrega el heatmap 2D a la UI, que compone la superposición |
| 2026-11-24 | `detector_neumonia.py` delega en `src.ui.main` | Eliminar la duplicación legacy y respetar la responsabilidad única |
| 2026-11-24 | Dockerfile migrado a `uv` (`uv sync --frozen --no-dev`) | Cumplir la prohibición de `pip` y `requirements.txt` |
| 2026-11-24 | `load_model.get_model` carga con `compile=False` | El `.h5` legado usa `reduction=auto`, rechazado por Keras 3 al recompilar; la inferencia no necesita optimizador |
| 2026-11-24 | Ruta por defecto `model/conv_MLP_84.h5` (minúsculas) | Compatibilidad con sistemas de archivos sensibles a mayúsculas (Linux/CI/Docker) |

## 10. Notas y tareas pendientes

- [ ] Migrar `src/` a un paquete `src/uao_neumonia_grupo1/` cuando se permita modificar su contenido.
- [ ] Subir `line-length` de Ruff a 100 y reformatear todo el proyecto.
- [ ] Incorporar el modelo `conv_MLP_84.h5` fuera de Git y documentar su descarga.
- [ ] Añadir pruebas de integración de la interfaz gráfica en CI (requiere entorno con display).
- [ ] Configurar la protección de la rama `main` en GitHub.
- [ ] Definir el roadmap de nuevas funcionalidades.

## 11. Configuración local (no trackear)

> Configuración específica de la máquina local. **NO incluir en commits.**

Dentro de la carpeta de proyectos, `~/.gitconfig` puede aplicar un perfil con `includeIf`:

```ini
[includeIf "gitdir:~/Repos/"]
    path = ~/.gitconfig-personal
```

`~/.gitconfig-personal`:

```ini
[user]
    name = <Tu Nombre>
    email = <tu-correo>
[core]
    sshCommand = ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes
```

> Ajusta la carpeta y tus datos. Guía completa (macOS/Windows) en el skill
> `jan-project-standard` → `references/git-identity.md`.
