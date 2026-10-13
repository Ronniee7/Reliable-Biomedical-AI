"""Validate the folder structure of an image dataset."""

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class DatasetValidationResult:
    """Summary of supported image counts and dataset structure problems."""

    is_valid: bool
    class_counts: dict[str, int]
    errors: list[str]


def validate_dataset(dataset_dir: str | Path) -> DatasetValidationResult:
    """Count supported images in each class folder and report any problems.

    Empty class folders are reported as errors so that they can be populated
    or removed before the dataset is used.
    """
    root_dir = Path(dataset_dir)
    if not root_dir.exists():
        return DatasetValidationResult(
            is_valid=False,
            class_counts={},
            errors=[
                f"Dataset directory does not exist: {root_dir}. "
                "Create the directory before validating it."
            ],
        )
    if not root_dir.is_dir():
        return DatasetValidationResult(
            is_valid=False,
            class_counts={},
            errors=[
                f"Dataset path is not a directory: {root_dir}. "
                "Pass the root folder containing the class folders."
            ],
        )

    class_directories = sorted(
        (path for path in root_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    if not class_directories:
        return DatasetValidationResult(
            is_valid=False,
            class_counts={},
            errors=[
                f"No class folders found in {root_dir}. "
                "Create one folder per class (for example, class_0/)."
            ],
        )

    class_counts: dict[str, int] = {}
    errors: list[str] = []
    for class_directory in class_directories:
        image_count = sum(
            1
            for path in class_directory.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
        class_counts[class_directory.name] = image_count
        if image_count == 0:
            errors.append(
                f"Class folder '{class_directory.name}' contains no supported "
                "images. Add PNG, JPG, or JPEG files to it."
            )

    if not any(class_counts.values()):
        errors.append(
            f"No supported images found in {root_dir}. "
            "Add PNG, JPG, or JPEG images inside class folders."
        )

    return DatasetValidationResult(
        is_valid=not errors,
        class_counts=class_counts,
        errors=errors,
    )