from unittest.mock import call, patch

import numpy as np
import pytest
import tensorflow as tf

from src import load_model
from src.grad_cam import generate_gradcam


def _clear_cache():
    load_model._model = None
    load_model._MODEL = None
    load_model._model_path = None


@pytest.fixture(autouse=True)
def _reset_model_cache():
    _clear_cache()
    yield
    _clear_cache()


def _build_model(
    target_class=0,
    num_classes=3,
    pooling_after_conv=False,
    zero_last_conv=False,
):
    inputs = tf.keras.Input(shape=(None, None, 1))
    x = tf.keras.layers.Conv2D(
        4, 3, padding="same", activation="relu", name="conv_first"
    )(inputs)
    x = tf.keras.layers.Conv2D(
        5, 3, padding="same", activation="relu", name="conv_last"
    )(x)
    if pooling_after_conv:
        x = tf.keras.layers.MaxPooling2D(2, padding="same", name="pool")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = tf.keras.layers.Dense(num_classes, name="dense")(x)
    model = tf.keras.Model(inputs, outputs)

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            kernel, bias = layer.get_weights()
            value = 0.0 if zero_last_conv and layer.name == "conv_last" else 0.2
            layer.set_weights(
                [np.full(kernel.shape, value, dtype=np.float32), np.zeros_like(bias)]
            )
        elif isinstance(layer, tf.keras.layers.Dense):
            kernel, bias = layer.get_weights()
            target_bias = np.full(bias.shape, -5.0, dtype=np.float32)
            target_bias[target_class] = 5.0
            layer.set_weights(
                [np.full(kernel.shape, 0.1, dtype=np.float32), target_bias]
            )

    return model


def _tensor(height=16, width=16):
    values = np.linspace(0.0, 1.0, height * width, dtype=np.float32)
    return values.reshape(1, height, width, 1)


def test_get_model_raises_file_not_found_for_missing_default_path():
    # Arrange
    missing = "Model/conv_MLP_84.h5"

    # Act
    with pytest.raises(FileNotFoundError) as error:
        load_model.get_model(missing)

    # Assert
    assert missing in str(error.value)


def test_get_model_raises_file_not_found_for_missing_custom_path(tmp_path):
    # Arrange
    missing = str(tmp_path / "weights.h5")

    # Act
    with pytest.raises(FileNotFoundError) as error:
        load_model.get_model(missing)

    # Assert
    assert missing in str(error.value)


def test_file_not_found_message_includes_requested_path(tmp_path):
    # Arrange
    missing = str(tmp_path / "modelo_ausente.h5")

    # Act
    with pytest.raises(FileNotFoundError) as error:
        load_model.get_model(missing)

    # Assert
    assert missing in str(error.value)


def test_file_not_found_message_mentions_model(tmp_path):
    # Arrange
    missing = str(tmp_path / "sin_modelo.h5")

    # Act
    with pytest.raises(FileNotFoundError) as error:
        load_model.get_model(missing)

    # Assert
    assert "modelo" in str(error.value).lower()


def test_get_model_does_not_call_keras_loader_when_file_missing(tmp_path):
    # Arrange
    missing = str(tmp_path / "missing.h5")
    loader_target = load_model.tf.keras.models

    # Act
    with (
        patch.object(loader_target, "load_model") as loader,
        pytest.raises(FileNotFoundError),
    ):
        load_model.get_model(missing)

    # Assert
    loader.assert_not_called()


def test_get_model_loads_model_when_file_exists(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    expected = object()
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(loader_target, "load_model", return_value=expected) as loader:
        result = load_model.get_model(str(path))

    # Assert
    assert result is expected
    loader.assert_called_once_with(str(path))


def test_get_model_is_singleton(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(loader_target, "load_model", return_value=object()) as loader:
        first = load_model.get_model(str(path))
        second = load_model.get_model(str(path))

    # Assert
    assert first is second
    loader.assert_called_once_with(str(path))


def test_singleton_returns_same_object_on_third_call(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(loader_target, "load_model", return_value=object()) as loader:
        models = [
            load_model.get_model(str(path)),
            load_model.get_model(str(path)),
            load_model.get_model(str(path)),
        ]

    # Assert
    assert models[0] is models[1] is models[2]
    loader.assert_called_once_with(str(path))


def test_singleton_skips_existence_check_after_first_load(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(loader_target, "load_model", return_value=object()):
        first = load_model.get_model(str(path))
    path.unlink()
    with patch.object(loader_target, "load_model") as second_loader:
        second = load_model.get_model(str(path))

    # Assert
    assert second is first
    second_loader.assert_not_called()


def test_get_model_uses_custom_path(tmp_path):
    # Arrange
    path = tmp_path / "conv_MLP_84.h5"
    path.write_bytes(b"dummy")
    expected = object()
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(loader_target, "load_model", return_value=expected) as loader:
        result = load_model.get_model(str(path))

    # Assert
    assert result is expected
    loader.assert_called_once_with(str(path))


def test_get_model_reloads_when_path_changes(tmp_path):
    # Arrange
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")
    first_model = object()
    second_model = object()
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(
        loader_target, "load_model", side_effect=[first_model, second_model]
    ) as loader:
        first = load_model.get_model(str(first_path))
        second = load_model.get_model(str(second_path))

    # Assert
    assert first is first_model
    assert second is second_model
    assert loader.call_args_list == [call(str(first_path)), call(str(second_path))]


def test_get_model_stores_cache_variables(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    expected = object()
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(loader_target, "load_model", return_value=expected):
        load_model.get_model(str(path))

    # Assert
    assert load_model._model is expected
    assert load_model._MODEL is expected
    assert load_model._model_path == str(path)


def test_reset_cache_forces_new_load(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    first_model = object()
    second_model = object()
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(
        loader_target, "load_model", side_effect=[first_model, second_model]
    ) as loader:
        first = load_model.get_model(str(path))
        second = load_model.get_model(str(path))
        _clear_cache()
        third = load_model.get_model(str(path))

    # Assert
    assert first is second
    assert third is second_model
    assert loader.call_count == 2


def test_reset_cache_clears_internal_references(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(loader_target, "load_model", return_value=object()):
        load_model.get_model(str(path))
    _clear_cache()

    # Assert
    assert load_model._model is None
    assert load_model._MODEL is None
    assert load_model._model_path is None


def test_get_model_passes_path_to_keras_loader_unchanged(tmp_path):
    # Arrange
    directory = tmp_path / "sub directorio"
    directory.mkdir()
    path = directory / "modelo con espacio.h5"
    path.write_bytes(b"dummy")
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(loader_target, "load_model", return_value=object()) as loader:
        load_model.get_model(str(path))

    # Assert
    loader.assert_called_once_with(str(path))


def test_get_model_propagates_loader_error(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    error = RuntimeError("fallo al cargar")
    loader_target = load_model.tf.keras.models

    # Act
    with (
        patch.object(loader_target, "load_model", side_effect=error),
        pytest.raises(RuntimeError, match="fallo al cargar"),
    ):
        load_model.get_model(str(path))

    # Assert
    assert load_model._model is None


def test_get_model_does_not_cache_failed_load(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    error = RuntimeError("fallo temporal")
    expected = object()
    loader_target = load_model.tf.keras.models

    # Act
    with patch.object(
        loader_target, "load_model", side_effect=[error, expected]
    ) as loader:
        with pytest.raises(RuntimeError, match="fallo temporal"):
            load_model.get_model(str(path))
        result = load_model.get_model(str(path))

    # Assert
    assert result is expected
    assert loader.call_count == 2


def test_get_model_checks_is_file_before_loading(tmp_path):
    # Arrange
    path = tmp_path / "model.h5"
    path.write_bytes(b"dummy")
    loader_target = load_model.tf.keras.models

    # Act
    with (
        patch.object(load_model.Path, "is_file", return_value=False),
        patch.object(loader_target, "load_model") as loader,
        pytest.raises(FileNotFoundError),
    ):
        load_model.get_model(str(path))

    # Assert
    loader.assert_not_called()


def test_generate_gradcam_returns_numpy_array():
    # Arrange
    model = _build_model()
    tensor = _tensor()

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert isinstance(result, np.ndarray)


def test_generate_gradcam_output_shape_is_512_by_512():
    # Arrange
    model = _build_model()
    tensor = _tensor()

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert result.shape == (512, 512)


def test_generate_gradcam_output_is_two_dimensional():
    # Arrange
    model = _build_model()
    tensor = _tensor()

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert result.ndim == 2


def test_generate_gradcam_values_are_within_unit_range():
    # Arrange
    model = _build_model()
    tensor = _tensor()

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


def test_generate_gradcam_maximum_value_is_one():
    # Arrange
    model = _build_model()
    tensor = _tensor()

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert float(result.max()) == pytest.approx(1.0)


def test_generate_gradcam_output_dtype_is_float32():
    # Arrange
    model = _build_model()
    tensor = _tensor()

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert result.dtype == np.float32


def test_generate_gradcam_uses_gradient_tape():
    # Arrange
    model = _build_model()
    tensor = _tensor()
    original_tape = tf.GradientTape

    # Act
    with patch("src.grad_cam.tf.GradientTape", wraps=original_tape) as tape_class:
        generate_gradcam(model, tensor)

    # Assert
    tape_class.assert_called_once_with()


def test_generate_gradcam_uses_explicit_layer_name():
    # Arrange
    model = _build_model()
    tensor = _tensor()
    original_model_class = tf.keras.Model

    # Act
    with patch(
        "src.grad_cam.tf.keras.Model", wraps=original_model_class
    ) as model_class:
        generate_gradcam(model, tensor, layer_name="conv_first")
    outputs = model_class.call_args.kwargs["outputs"]

    # Assert
    assert outputs[0] is model.get_layer("conv_first").output


def test_generate_gradcam_calls_get_layer_when_name_provided():
    # Arrange
    model = _build_model()
    tensor = _tensor()

    # Act
    with patch.object(model, "get_layer", wraps=model.get_layer) as get_layer:
        generate_gradcam(model, tensor, layer_name="conv_last")

    # Assert
    get_layer.assert_called_once_with("conv_last")


def test_generate_gradcam_raises_for_unknown_layer_name():
    # Arrange
    model = _build_model()
    tensor = _tensor()

    # Act
    with pytest.raises(ValueError):
        generate_gradcam(model, tensor, layer_name="no_existe")

    # Assert
    assert True


def test_generate_gradcam_auto_detects_last_convolutional_layer():
    # Arrange
    model = _build_model()
    tensor = _tensor()
    original_model_class = tf.keras.Model

    # Act
    with patch(
        "src.grad_cam.tf.keras.Model", wraps=original_model_class
    ) as model_class:
        generate_gradcam(model, tensor)
    outputs = model_class.call_args.kwargs["outputs"]

    # Assert
    assert outputs[0] is model.get_layer("conv_last").output


def test_generate_gradcam_auto_detection_ignores_later_pooling_layer():
    # Arrange
    model = _build_model(pooling_after_conv=True)
    tensor = _tensor()
    original_model_class = tf.keras.Model

    # Act
    with patch(
        "src.grad_cam.tf.keras.Model", wraps=original_model_class
    ) as model_class:
        generate_gradcam(model, tensor)
    outputs = model_class.call_args.kwargs["outputs"]

    # Assert
    assert outputs[0] is model.get_layer("conv_last").output
    assert outputs[0] is not model.get_layer("pool").output


def test_generate_gradcam_raises_value_error_without_convolutional_layers():
    # Arrange
    inputs = tf.keras.Input(shape=(8,))
    outputs = tf.keras.layers.Dense(3)(inputs)
    model = tf.keras.Model(inputs, outputs)
    tensor = np.ones((1, 8), dtype=np.float32)

    # Act
    with pytest.raises(ValueError, match="convolucional"):
        generate_gradcam(model, tensor)

    # Assert
    assert True


def test_generate_gradcam_value_error_message_mentions_convolucional():
    # Arrange
    inputs = tf.keras.Input(shape=(8,))
    outputs = tf.keras.layers.Dense(3)(inputs)
    model = tf.keras.Model(inputs, outputs)
    tensor = np.ones((1, 8), dtype=np.float32)

    # Act
    with pytest.raises(ValueError) as error:
        generate_gradcam(model, tensor)

    # Assert
    assert str(error.value) == "El modelo no contiene una capa convolucional 2D."


def test_generate_gradcam_handles_zero_gradients_without_nan():
    # Arrange
    model = _build_model(zero_last_conv=True)
    tensor = _tensor()

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert np.all(np.isfinite(result))
    assert not np.any(np.isnan(result))


def test_generate_gradcam_handles_zero_feature_maps_without_division_error():
    # Arrange
    model = _build_model(zero_last_conv=True)
    tensor = _tensor()

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert np.all(result == 0.0)
    assert float(np.max(result)) == 0.0


def test_generate_gradcam_predicted_class_zero():
    # Arrange
    model = _build_model(target_class=0)
    tensor = _tensor()

    # Act
    predictions = model(tensor, training=False).numpy()
    result = generate_gradcam(model, tensor)

    # Assert
    assert int(np.argmax(predictions[0])) == 0
    assert result.shape == (512, 512)
    assert float(result.max()) > 0.0


def test_generate_gradcam_predicted_class_one():
    # Arrange
    model = _build_model(target_class=1)
    tensor = _tensor()

    # Act
    predictions = model(tensor, training=False).numpy()
    result = generate_gradcam(model, tensor)

    # Assert
    assert int(np.argmax(predictions[0])) == 1
    assert result.shape == (512, 512)
    assert float(result.max()) > 0.0


def test_generate_gradcam_predicted_class_two():
    # Arrange
    model = _build_model(target_class=2)
    tensor = _tensor()

    # Act
    predictions = model(tensor, training=False).numpy()
    result = generate_gradcam(model, tensor)

    # Assert
    assert int(np.argmax(predictions[0])) == 2
    assert result.shape == (512, 512)
    assert float(result.max()) > 0.0


def test_generate_gradcam_accepts_three_dimensional_input():
    # Arrange
    model = _build_model()
    tensor = _tensor()[0]

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert result.shape == (512, 512)


def test_generate_gradcam_accepts_two_dimensional_input():
    # Arrange
    model = _build_model()
    tensor = _tensor()[0, :, :, 0]

    # Act
    result = generate_gradcam(model, tensor)

    # Assert
    assert result.shape == (512, 512)


def test_generate_gradcam_raises_for_invalid_input_rank():
    # Arrange
    model = _build_model()
    tensor = np.zeros((2, 2, 2, 2, 2), dtype=np.float32)

    # Act
    with pytest.raises(ValueError, match="dimensiones"):
        generate_gradcam(model, tensor)

    # Assert
    assert True
