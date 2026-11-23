import cv2
import numpy as np
import pytest
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import (
    ExplicitVRLittleEndian,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from src.preprocess_img import preprocess_image
from src.read_img import load_image


def _write_image(path, array, image_format):
    Image.fromarray(array).save(path, format=image_format)


def _write_dicom(path, pixels):
    metadata = FileMetaDataset()
    metadata.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    metadata.MediaStorageSOPInstanceUID = generate_uid()
    metadata.TransferSyntaxUID = ExplicitVRLittleEndian

    dataset = FileDataset(
        str(path), {}, file_meta=metadata, preamble=b"\0" * 128
    )
    dataset.SOPClassUID = SecondaryCaptureImageStorage
    dataset.SOPInstanceUID = metadata.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.Modality = "OT"
    dataset.Rows = pixels.shape[0]
    dataset.Columns = pixels.shape[1]
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = np.asarray(pixels, dtype=np.uint16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


def _rgb_image(height, width, color):
    return np.full((height, width, 3), color, dtype=np.uint8)


def test_load_image_reads_png_with_expected_shape_and_dtype(tmp_path):
    # Arrange
    path = tmp_path / "image.png"
    _write_image(path, _rgb_image(64, 48, (120, 80, 40)), "PNG")

    # Act
    image = load_image(str(path))

    # Assert
    assert image.shape == (64, 48, 3)
    assert image.dtype == np.uint8


def test_load_image_reads_jpg_file(tmp_path):
    # Arrange
    path = tmp_path / "image.jpg"
    _write_image(path, _rgb_image(50, 70, (200, 50, 30)), "JPEG")

    # Act
    image = load_image(str(path))

    # Assert
    assert image.shape == (50, 70, 3)
    assert image.dtype == np.uint8


def test_load_image_reads_jpeg_file(tmp_path):
    # Arrange
    path = tmp_path / "image.jpeg"
    _write_image(path, _rgb_image(80, 60, (20, 90, 180)), "JPEG")

    # Act
    image = load_image(str(path))

    # Assert
    assert image.shape == (80, 60, 3)
    assert image.dtype == np.uint8


def test_load_image_returns_rgb_channel_order_for_png(tmp_path):
    # Arrange
    path = tmp_path / "red.png"
    _write_image(path, _rgb_image(20, 20, (255, 0, 0)), "PNG")

    # Act
    image = load_image(str(path))

    # Assert
    assert tuple(image[10, 10]) == (255, 0, 0)


def test_load_image_reads_simulated_dicom(tmp_path):
    # Arrange
    path = tmp_path / "image.dcm"
    pixels = np.arange(32 * 36, dtype=np.uint16).reshape(32, 36) + 100
    _write_dicom(path, pixels)

    # Act
    image = load_image(str(path))

    # Assert
    assert image.shape == (32, 36)
    assert image.dtype == np.uint8


def test_load_image_normalizes_dicom_to_full_uint8_range(tmp_path):
    # Arrange
    path = tmp_path / "image.dcm"
    pixels = np.linspace(500, 2500, 24 * 24, dtype=np.uint16).reshape(24, 24)
    _write_dicom(path, pixels)

    # Act
    image = load_image(str(path))

    # Assert
    assert image.min() == 0
    assert image.max() == 255


def test_load_image_normalizes_constant_dicom_to_uniform_array(tmp_path):
    # Arrange
    path = tmp_path / "constant.dcm"
    _write_dicom(path, np.full((16, 16), 2048, dtype=np.uint16))

    # Act
    image = load_image(str(path))

    # Assert
    assert np.all(image == 0)
    assert image.shape == (16, 16)


def test_load_image_keeps_dicom_as_two_dimensions(tmp_path):
    # Arrange
    path = tmp_path / "grayscale.dcm"
    _write_dicom(path, np.arange(20 * 30, dtype=np.uint16).reshape(20, 30))

    # Act
    image = load_image(str(path))

    # Assert
    assert image.ndim == 2
    assert image.shape == (20, 30)


def test_load_image_raises_for_missing_png(tmp_path):
    # Arrange
    path = tmp_path / "missing.png"

    # Act
    with pytest.raises(ValueError, match="no existe"):
        load_image(str(path))

    # Assert
    assert not path.exists()


def test_load_image_raises_for_missing_dicom(tmp_path):
    # Arrange
    path = tmp_path / "missing.dcm"

    # Act
    with pytest.raises(ValueError, match="no existe"):
        load_image(str(path))

    # Assert
    assert not path.exists()


def test_load_image_raises_for_unsupported_extension(tmp_path):
    # Arrange
    path = tmp_path / "image.bmp"
    Image.fromarray(_rgb_image(10, 10, (10, 20, 30))).save(path, format="BMP")

    # Act
    with pytest.raises(ValueError, match="no soportado"):
        load_image(str(path))

    # Assert
    assert path.exists()


def test_load_image_raises_for_corrupt_png(tmp_path):
    # Arrange
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"this is not a png image")

    # Act
    with pytest.raises(ValueError, match="No se pudo leer"):
        load_image(str(path))

    # Assert
    assert path.stat().st_size > 0


def test_load_image_raises_for_empty_png_file(tmp_path):
    # Arrange
    path = tmp_path / "empty.png"
    path.write_bytes(b"")

    # Act
    with pytest.raises(ValueError, match="No se pudo leer"):
        load_image(str(path))

    # Assert
    assert path.stat().st_size == 0


def test_load_image_raises_for_empty_path():
    # Arrange
    path = ""

    # Act
    with pytest.raises(ValueError, match="vacía"):
        load_image(path)

    # Assert
    assert path == ""


def test_load_image_raises_for_invalid_dicom(tmp_path):
    # Arrange
    path = tmp_path / "invalid.dcm"
    path.write_bytes(b"invalid dicom content")

    # Act
    with pytest.raises(ValueError, match="DICOM"):
        load_image(str(path))

    # Assert
    assert path.stat().st_size > 0


def test_load_image_accepts_uppercase_png_extension(tmp_path):
    # Arrange
    path = tmp_path / "IMAGE.PNG"
    _write_image(path, _rgb_image(30, 40, (60, 120, 180)), "PNG")

    # Act
    image = load_image(str(path))

    # Assert
    assert image.shape == (30, 40, 3)
    assert image.dtype == np.uint8


def test_load_image_preserves_non_square_dimensions(tmp_path):
    # Arrange
    path = tmp_path / "wide.png"
    _write_image(path, _rgb_image(41, 67, (90, 45, 15)), "PNG")

    # Act
    image = load_image(str(path))

    # Assert
    assert image.shape == (41, 67, 3)


def test_preprocess_returns_exact_output_shape():
    # Arrange
    image = _rgb_image(128, 128, (100, 150, 200))

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_returns_float32():
    # Arrange
    image = _rgb_image(64, 64, (25, 50, 75))

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.dtype == np.float32


def test_preprocess_returns_values_within_unit_range():
    # Arrange
    image = np.arange(96 * 96 * 3, dtype=np.uint8).reshape(96, 96, 3)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_rgb_matches_equivalent_grayscale():
    # Arrange
    rgb = np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Act
    rgb_tensor = preprocess_image(rgb)
    grayscale_tensor = preprocess_image(grayscale)

    # Assert
    assert np.array_equal(rgb_tensor, grayscale_tensor)


def test_preprocess_rgba_matches_equivalent_grayscale():
    # Arrange
    rgb = np.arange(70 * 90 * 3, dtype=np.uint8).reshape(70, 90, 3)
    alpha = np.full((70, 90, 1), 255, dtype=np.uint8)
    rgba = np.concatenate((rgb, alpha), axis=2)
    grayscale = cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)

    # Act
    rgba_tensor = preprocess_image(rgba)
    grayscale_tensor = preprocess_image(grayscale)

    # Assert
    assert np.array_equal(rgba_tensor, grayscale_tensor)


def test_preprocess_accepts_grayscale_image():
    # Arrange
    image = np.linspace(0, 255, 120 * 90, dtype=np.uint8).reshape(120, 90)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)
    assert tensor.dtype == np.float32


def test_preprocess_accepts_single_channel_image():
    # Arrange
    image = np.full((90, 120, 1), 128, dtype=np.uint8)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_resizes_square_image():
    # Arrange
    image = _rgb_image(300, 300, (10, 20, 30))

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_resizes_wide_rectangular_image():
    # Arrange
    image = _rgb_image(200, 600, (30, 60, 90))

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_resizes_tall_rectangular_image():
    # Arrange
    image = _rgb_image(600, 200, (90, 60, 30))

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_resizes_small_image():
    # Arrange
    image = np.full((8, 8), 64, dtype=np.uint8)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_resizes_large_image():
    # Arrange
    image = _rgb_image(1200, 1600, (40, 80, 120))

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_black_image_produces_uniform_output():
    # Arrange
    image = np.zeros((256, 256), dtype=np.uint8)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert float(np.ptp(tensor)) == 0.0
    assert 0.0 <= float(tensor.min()) <= 1.0


def test_preprocess_white_image_produces_ones():
    # Arrange
    image = np.full((256, 256), 255, dtype=np.uint8)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert np.allclose(tensor, 1.0)


def test_preprocess_random_noise_stays_in_range():
    # Arrange
    rng = np.random.default_rng(42)
    image = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0


def test_preprocess_random_noise_returns_expected_shape():
    # Arrange
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(150, 100), dtype=np.uint8)

    # Act
    tensor = preprocess_image(image)

    # Assert
    assert tensor.shape == (1, 512, 512, 1)
    assert tensor.dtype == np.float32


def test_preprocess_uses_configured_clahe_parameters(monkeypatch):
    # Arrange
    calls = {}

    class FakeClahe:
        def apply(self, array):
            calls["input"] = array
            return array

    def fake_create_clahe(clipLimit, tileGridSize):
        calls["clipLimit"] = clipLimit
        calls["tileGridSize"] = tileGridSize
        return FakeClahe()

    monkeypatch.setattr(cv2, "createCLAHE", fake_create_clahe)

    # Act
    tensor = preprocess_image(np.full((32, 32), 128, dtype=np.uint8))

    # Assert
    assert calls["clipLimit"] == 2.0
    assert calls["tileGridSize"] == (8, 8)
    assert tensor.shape == (1, 512, 512, 1)


def test_preprocess_raises_for_none_image():
    # Arrange
    image = None

    # Act
    with pytest.raises(ValueError, match="nula"):
        preprocess_image(image)

    # Assert
    assert image is None


def test_preprocess_raises_for_empty_array():
    # Arrange
    image = np.array([], dtype=np.uint8)

    # Act
    with pytest.raises(ValueError, match="vacía"):
        preprocess_image(image)

    # Assert
    assert image.size == 0


def test_preprocess_raises_for_two_channel_image():
    # Arrange
    image = np.zeros((32, 32, 2), dtype=np.uint8)

    # Act
    with pytest.raises(ValueError, match="canales"):
        preprocess_image(image)

    # Assert
    assert image.shape[2] == 2


def test_preprocess_raises_for_one_dimensional_array():
    # Arrange
    image = np.zeros(100, dtype=np.uint8)

    # Act
    with pytest.raises(ValueError, match="dimensiones"):
        preprocess_image(image)

    # Assert
    assert image.ndim == 1


def test_pipeline_png_produces_model_tensor(tmp_path):
    # Arrange
    path = tmp_path / "pipeline.png"
    _write_image(path, _rgb_image(128, 96, (70, 140, 210)), "PNG")

    # Act
    tensor = preprocess_image(load_image(str(path)))

    # Assert
    assert tensor.shape == (1, 512, 512, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_pipeline_dicom_produces_model_tensor(tmp_path):
    # Arrange
    path = tmp_path / "pipeline.dcm"
    pixels = np.linspace(200, 1800, 64 * 64, dtype=np.uint16).reshape(64, 64)
    _write_dicom(path, pixels)

    # Act
    tensor = preprocess_image(load_image(str(path)))

    # Assert
    assert tensor.shape == (1, 512, 512, 1)
    assert tensor.dtype == np.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
