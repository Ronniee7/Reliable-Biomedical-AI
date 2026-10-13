"""Tests for image dataset loading and dataset structure validation."""

from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision import transforms

from robustmed.data.dataset import MedicalImageDataset
from robustmed.data.validation import validate_dataset


def create_image(path: Path, color: tuple[int, int, int] = (120, 40, 10)) -> None:
    """Create a small RGB image at ``path`` for use in tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def test_dataset_loads_images_and_assigns_sorted_class_labels(
    tmp_path: Path,
) -> None:
    """Images load as RGB tensors and labels follow sorted class names."""
    create_image(tmp_path / "class_1" / "image.jpeg")
    create_image(tmp_path / "class_0" / "image.PNG")
    create_image(tmp_path / "class_0" / "second.jpg")
    (tmp_path / "class_0" / "notes.txt").write_text("not an image")

    dataset = MedicalImageDataset(
        tmp_path,
        transform=transforms.Compose(
            [transforms.Resize((4, 4)), transforms.ToTensor()]
        ),
    )

    assert len(dataset) == 3
    assert dataset.classes == ["class_0", "class_1"]
    assert dataset.class_to_idx == {"class_0": 0, "class_1": 1}

    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 4, 4)
    assert image.dtype == torch.float32
    assert label == 0
    assert [dataset[index][1] for index in range(len(dataset))] == [0, 0, 1]


def test_dataset_converts_images_to_rgb_tensor_without_transform(
    tmp_path: Path,
) -> None:
    """Images become three-channel tensors when no transform is provided."""
    gray_image = Image.new("L", (6, 5), color=128)
    class_dir = tmp_path / "class_0"
    class_dir.mkdir()
    gray_image.save(class_dir / "gray.png")

    image, label = MedicalImageDataset(tmp_path)[0]

    assert image.shape == (3, 5, 6)
    assert label == 0


def test_dataset_raises_for_missing_directory(tmp_path: Path) -> None:
    """A missing dataset path produces a clear file-not-found error."""
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Dataset directory does not exist"):
        MedicalImageDataset(missing_dir)

    result = validate_dataset(missing_dir)
    assert not result.is_valid
    assert "Dataset directory does not exist" in result.errors[0]


def test_dataset_raises_for_empty_dataset_directory(tmp_path: Path) -> None:
    """A dataset with no class folders is rejected."""
    with pytest.raises(ValueError, match="No class folders found"):
        MedicalImageDataset(tmp_path)


def test_dataset_raises_when_class_folders_contain_no_supported_images(
    tmp_path: Path,
) -> None:
    """Empty class folders cannot produce a dataset with zero samples."""
    (tmp_path / "class_0").mkdir()

    with pytest.raises(ValueError, match="No supported images found"):
        MedicalImageDataset(tmp_path)


def test_validation_reports_counts_and_empty_classes(tmp_path: Path) -> None:
    """Validation counts image files and identifies classes needing images."""
    create_image(tmp_path / "class_0" / "one.png")
    create_image(tmp_path / "class_0" / "two.jpg")
    (tmp_path / "class_1").mkdir()
    (tmp_path / "class_2").mkdir()
    (tmp_path / "class_2" / "notes.txt").write_text("not an image")

    result = validate_dataset(tmp_path)

    assert not result.is_valid
    assert result.class_counts == {"class_0": 2, "class_1": 0, "class_2": 0}
    assert len(result.errors) == 2
    assert "class_1" in result.errors[0]
    assert "class_2" in result.errors[1]


def test_validation_accepts_populated_class_folders(tmp_path: Path) -> None:
    """A dataset with supported images in each class passes validation."""
    create_image(tmp_path / "class_0" / "one.png")
    create_image(tmp_path / "class_1" / "two.jpeg")

    result = validate_dataset(tmp_path)

    assert result.is_valid
    assert result.class_counts == {"class_0": 1, "class_1": 1}
    assert result.errors == []


def test_validation_reports_dataset_without_class_folders(tmp_path: Path) -> None:
    """An existing but empty directory receives an actionable validation error."""
    result = validate_dataset(tmp_path)

    assert not result.is_valid
    assert result.class_counts == {}
    assert "No class folders found" in result.errors[0]
