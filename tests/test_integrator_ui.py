import ast
import tkinter as tk
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src import integrator, ui


def _model_with_probabilities(probabilities):
    model = MagicMock()
    model.predict.return_value = np.array([probabilities])
    return model


def _ui_imports():
    source = Path(ui.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


@pytest.fixture(autouse=True)
def _stub_missing_dependencies():
    targets = [
        "src.integrator.read_img.load_image",
        "src.integrator.preprocess_img.preprocess_image",
        "src.integrator.load_model.get_model",
        "src.integrator.grad_cam.generate_gradcam",
    ]
    with ExitStack() as stack:
        for target in targets:
            stack.enter_context(patch(target, create=True))
        yield


@pytest.fixture(scope="module")
def tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def app(tk_root):
    window = tk.Toplevel(tk_root)
    window.withdraw()
    instance = ui.App(window)
    yield instance
    window.destroy()


def test_load_image_delegates_to_read_img():
    # Arrange
    expected = ("array", "image")

    # Act
    with patch(
        "src.integrator.read_img.load_image", return_value=expected
    ) as load_image:
        result = integrator.load_image("scan.dcm")

    # Assert
    assert result == expected
    load_image.assert_called_once_with("scan.dcm")


def test_load_image_returns_array_and_image_tuple():
    # Arrange
    array = np.zeros((2, 2, 3), dtype=np.uint8)
    image = Image.new("RGB", (2, 2))

    # Act
    with patch("src.integrator.read_img.load_image", return_value=(array, image)):
        result = integrator.load_image("scan.dcm")

    # Assert
    assert result == (array, image)


def test_load_image_propagates_read_error():
    # Arrange
    error = FileNotFoundError("missing file")

    # Act
    with (
        patch("src.integrator.read_img.load_image", side_effect=error),
        pytest.raises(FileNotFoundError, match="missing file"),
    ):
        integrator.load_image("missing.dcm")

    # Assert
    assert True


def test_predict_returns_bacteriana_class():
    # Arrange
    model = _model_with_probabilities([0.8, 0.1, 0.1])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        label, probability, heatmap = integrator.predict("array")

    # Assert
    assert label == "bacteriana"
    assert probability == pytest.approx(80.0)
    assert heatmap == "heatmap"


def test_predict_returns_viral_class():
    # Arrange
    model = _model_with_probabilities([0.1, 0.1, 0.8])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        label, probability, heatmap = integrator.predict("array")

    # Assert
    assert label == "viral"
    assert probability == pytest.approx(80.0)
    assert heatmap == "heatmap"


def test_predict_returns_normal_class():
    # Arrange
    model = _model_with_probabilities([0.1, 0.8, 0.1])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        label, probability, heatmap = integrator.predict("array")

    # Assert
    assert label == "normal"
    assert probability == pytest.approx(80.0)
    assert heatmap == "heatmap"


def test_predict_returns_unknown_class_label():
    # Arrange
    model = _model_with_probabilities([0.1, 0.1, 0.1, 0.7])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        label, probability, heatmap = integrator.predict("array")

    # Assert
    assert label == "desconocida"
    assert probability == pytest.approx(70.0)
    assert heatmap == "heatmap"


def test_predict_probability_is_percentage():
    # Arrange
    model = _model_with_probabilities([0.25, 0.5, 0.25])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        _, probability, _ = integrator.predict("array")

    # Assert
    assert probability == pytest.approx(50.0)


def test_predict_calls_preprocess_image_with_array():
    # Arrange
    model = _model_with_probabilities([0.8, 0.1, 0.1])

    # Act
    with (
        patch(
            "src.integrator.preprocess_img.preprocess_image", return_value="batch"
        ) as preprocess,
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        integrator.predict("array")

    # Assert
    preprocess.assert_called_once_with("array")


def test_predict_calls_get_model_once():
    # Arrange
    model = _model_with_probabilities([0.8, 0.1, 0.1])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model) as get_model,
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        integrator.predict("array")

    # Assert
    get_model.assert_called_once_with()


def test_predict_calls_model_predict_with_batch():
    # Arrange
    model = _model_with_probabilities([0.8, 0.1, 0.1])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        integrator.predict("array")

    # Assert
    model.predict.assert_called_once_with("batch")


def test_predict_calls_generate_gradcam_with_array_and_model():
    # Arrange
    model = _model_with_probabilities([0.8, 0.1, 0.1])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch(
            "src.integrator.grad_cam.generate_gradcam", return_value="heatmap"
        ) as generate_gradcam,
    ):
        integrator.predict("array")

    # Assert
    generate_gradcam.assert_called_once_with("array", model)


def test_predict_returns_gradcam_heatmap():
    # Arrange
    model = _model_with_probabilities([0.8, 0.1, 0.1])
    heatmap = MagicMock()

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value=heatmap),
    ):
        result = integrator.predict("array")

    # Assert
    assert result[2] is heatmap


def test_predict_calls_dependencies_in_order():
    # Arrange
    calls = []
    model = MagicMock()
    model.predict.side_effect = lambda batch: (
        calls.append("predict") or np.array([[0.8, 0.1, 0.1]])
    )

    # Act
    with (
        patch(
            "src.integrator.preprocess_img.preprocess_image",
            side_effect=lambda array: calls.append("preprocess") or "batch",
        ),
        patch(
            "src.integrator.load_model.get_model",
            side_effect=lambda: calls.append("model") or model,
        ),
        patch(
            "src.integrator.grad_cam.generate_gradcam",
            side_effect=lambda array, model: calls.append("gradcam") or "heatmap",
        ),
    ):
        integrator.predict("array")

    # Assert
    assert calls == ["preprocess", "model", "predict", "gradcam"]


def test_predict_propagates_preprocess_error():
    # Arrange
    model = _model_with_probabilities([0.8, 0.1, 0.1])

    # Act
    with (
        patch(
            "src.integrator.preprocess_img.preprocess_image",
            side_effect=ValueError("bad preprocess"),
        ),
        patch("src.integrator.load_model.get_model", return_value=model) as get_model,
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
        pytest.raises(ValueError, match="bad preprocess"),
    ):
        integrator.predict("array")

    # Assert
    get_model.assert_not_called()


def test_predict_propagates_model_error():
    # Arrange
    error = RuntimeError("model failure")

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", side_effect=error),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
        pytest.raises(RuntimeError, match="model failure"),
    ):
        integrator.predict("array")

    # Assert
    assert True


def test_predict_propagates_prediction_error():
    # Arrange
    model = MagicMock()
    model.predict.side_effect = RuntimeError("prediction failure")

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
        pytest.raises(RuntimeError, match="prediction failure"),
    ):
        integrator.predict("array")

    # Assert
    assert True


def test_predict_propagates_gradcam_error():
    # Arrange
    model = _model_with_probabilities([0.8, 0.1, 0.1])
    error = RuntimeError("gradcam failure")

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", side_effect=error),
        pytest.raises(RuntimeError, match="gradcam failure"),
    ):
        integrator.predict("array")

    # Assert
    assert True


def test_predict_uses_first_batch_row():
    # Arrange
    model = MagicMock()
    model.predict.return_value = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        label, probability, _ = integrator.predict("array")

    # Assert
    assert label == "viral"
    assert probability == pytest.approx(70.0)


def test_predict_handles_zero_probability():
    # Arrange
    model = _model_with_probabilities([0.0, 0.0, 0.0])

    # Act
    with (
        patch("src.integrator.preprocess_img.preprocess_image", return_value="batch"),
        patch("src.integrator.load_model.get_model", return_value=model),
        patch("src.integrator.grad_cam.generate_gradcam", return_value="heatmap"),
    ):
        label, probability, _ = integrator.predict("array")

    # Assert
    assert label == "bacteriana"
    assert probability == 0.0


def test_ui_ast_does_not_import_tensorflow():
    # Arrange
    modules = _ui_imports()

    # Act
    has_tensorflow = "tensorflow" in modules

    # Assert
    assert has_tensorflow is False


def test_ui_ast_does_not_import_tensorflow_submodule():
    # Arrange
    modules = _ui_imports()

    # Act
    has_tensorflow_submodule = any(
        module.startswith("tensorflow.") for module in modules
    )

    # Assert
    assert has_tensorflow_submodule is False


def test_ui_ast_does_not_import_keras():
    # Arrange
    modules = _ui_imports()

    # Act
    has_keras = any(
        module == "keras" or module.startswith("keras.") for module in modules
    )

    # Assert
    assert has_keras is False


def test_ui_ast_does_not_import_load_model():
    # Arrange
    modules = _ui_imports()

    # Act
    has_load_model = any(module.endswith("load_model") for module in modules)

    # Assert
    assert has_load_model is False


def test_ui_ast_imports_integrator():
    # Arrange
    modules = _ui_imports()

    # Act
    imports_integrator = "src.integrator" in modules

    # Assert
    assert imports_integrator is True


def test_ui_ast_only_src_import_is_integrator():
    # Arrange
    modules = _ui_imports()

    # Act
    src_imports = [module for module in modules if module.startswith("src.")]

    # Assert
    assert src_imports == ["src.integrator"]


def test_ui_app_initial_state(app):
    # Arrange
    expected_button_state = "disabled"

    # Act
    state = str(app.button1["state"])

    # Assert
    assert app.array is None
    assert app.label is None
    assert app.proba is None
    assert app.heatmap is None
    assert state == expected_button_state


def test_ui_load_img_file_calls_integrator(app):
    # Arrange
    image = Image.new("RGB", (10, 10))

    # Act
    with (
        patch("src.ui.filedialog.askopenfilename", return_value="scan.dcm"),
        patch(
            "src.ui.integrator.load_image", return_value=("array", image)
        ) as load_image,
    ):
        app.load_img_file()

    # Assert
    load_image.assert_called_once_with("scan.dcm")
    assert app.array == "array"


def test_ui_load_img_file_without_selection_does_nothing(app):
    # Arrange
    with (
        patch("src.ui.filedialog.askopenfilename", return_value=""),
        patch("src.ui.integrator.load_image") as load_image,
    ):
        # Act
        app.load_img_file()

    # Assert
    load_image.assert_not_called()
    assert app.array is None


def test_ui_load_img_file_enables_predict_button(app):
    # Arrange
    image = Image.new("RGB", (10, 10))

    # Act
    with (
        patch("src.ui.filedialog.askopenfilename", return_value="scan.dcm"),
        patch("src.ui.integrator.load_image", return_value=("array", image)),
    ):
        app.load_img_file()

    # Assert
    assert str(app.button1["state"]) == "enabled"


def test_ui_run_model_without_image_shows_info(app):
    # Arrange
    with (
        patch("src.ui.integrator.predict") as predict,
        patch("src.ui.showinfo") as showinfo,
    ):
        # Act
        app.run_model()

    # Assert
    predict.assert_not_called()
    showinfo.assert_called_once()


def test_ui_run_model_calls_integrator_predict(app):
    # Arrange
    image = Image.new("RGB", (10, 10))

    # Act
    with patch(
        "src.ui.integrator.predict", return_value=("normal", 90.0, image)
    ) as predict:
        app.array = "array"
        app.run_model()

    # Assert
    predict.assert_called_once_with("array")


def test_ui_run_model_updates_result_widgets(app):
    # Arrange
    image = Image.new("RGB", (10, 10))

    # Act
    with patch("src.ui.integrator.predict", return_value=("viral", 77.5, image)):
        app.array = "array"
        app.run_model()

    # Assert
    assert app.label == "viral"
    assert app.proba == 77.5
    assert app.heatmap is image
    assert app.text2.get("1.0", "end-1c") == "viral"
    assert app.text3.get("1.0", "end-1c") == "77.50%"


def test_ui_save_results_csv_without_prediction_shows_info(app):
    # Arrange
    with patch("src.ui.showinfo") as showinfo:
        # Act
        app.save_results_csv()

    # Assert
    showinfo.assert_called_once()


def test_ui_save_results_csv_writes_expected_row(app, tmp_path, monkeypatch):
    # Arrange
    app.label = "normal"
    app.proba = 88.0
    app.text1.insert(0, "123")
    monkeypatch.chdir(tmp_path)

    # Act
    with patch("src.ui.showinfo"):
        app.save_results_csv()

    # Assert
    content = (tmp_path / "historial.csv").read_text(encoding="utf-8")
    assert "123-normal-88.00%" in content


def test_ui_delete_cancelled_keeps_state(app):
    # Arrange
    app.array = "array"
    app.label = "normal"
    app.proba = 50.0

    # Act
    with (
        patch("src.ui.askokcancel", return_value=False),
        patch("src.ui.showinfo") as showinfo,
    ):
        app.delete()

    # Assert
    assert app.array == "array"
    assert app.label == "normal"
    assert app.proba == 50.0
    showinfo.assert_not_called()


def test_ui_delete_confirmed_clears_state(app):
    # Arrange
    app.array = "array"
    app.label = "normal"
    app.proba = 50.0
    app.heatmap = "heatmap"
    app.text1.insert(0, "123")

    # Act
    with (
        patch("src.ui.askokcancel", return_value=True),
        patch("src.ui.showinfo"),
    ):
        app.delete()

    # Assert
    assert app.array is None
    assert app.label is None
    assert app.proba is None
    assert app.heatmap is None
    assert app.text1.get() == ""


def test_ui_delete_confirmed_disables_predict_button(app):
    # Arrange
    app.button1["state"] = "enabled"

    # Act
    with (
        patch("src.ui.askokcancel", return_value=True),
        patch("src.ui.showinfo"),
    ):
        app.delete()

    # Assert
    assert str(app.button1["state"]) == "disabled"


def test_ui_create_pdf_saves_report(app, tmp_path, monkeypatch):
    # Arrange
    image = Image.new("RGB", (10, 10))
    monkeypatch.chdir(tmp_path)

    # Act
    with (
        patch("src.ui.ImageGrab.grab", return_value=image) as grab,
        patch("src.ui.showinfo"),
    ):
        app.create_pdf()

    # Assert
    assert (tmp_path / "Reporte0.pdf").exists()
    assert app.report_id == 1
    grab.assert_called_once()


def test_ui_photo_image_converts_array(app):
    # Arrange
    array = np.zeros((20, 30, 3), dtype=np.uint8)

    # Act
    with patch("src.ui.ImageTk.PhotoImage") as photo_image:
        result = app._photo_image(array, size=(50, 60))

    # Assert
    image = photo_image.call_args.args[0]
    assert result is photo_image.return_value
    assert isinstance(image, Image.Image)
    assert image.size == (50, 60)
