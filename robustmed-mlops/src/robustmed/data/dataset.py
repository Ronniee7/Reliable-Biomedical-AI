"""PyTorch dataset for images organized into class folders."""

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class MedicalImageDataset(Dataset[tuple[Tensor, int]]):
    """Load RGB images from a directory with one subdirectory per class.

    Class labels are assigned in alphabetical order of the class directory
    names, making the mapping stable regardless of directory creation order.
    """

    def __init__(
        self,
        root_dir: str | Path,
        transform: Callable[[Image.Image], object] | None = None,
    ) -> None:
        """Find supported images beneath ``root_dir`` and assign class labels."""
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root_dir}"
            )
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"Dataset path is not a directory: {self.root_dir}"
            )

        class_directories = sorted(
            (path for path in self.root_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
        if not class_directories:
            raise ValueError(
                f"No class folders found in {self.root_dir}. "
                "Create one folder per class (for example, class_0/)."
            )

        self.classes = [path.name for path in class_directories]
        self.class_to_idx = {
            class_name: index for index, class_name in enumerate(self.classes)
        }
        self.samples: list[tuple[Path, int]] = []
        for class_directory in class_directories:
            label = self.class_to_idx[class_directory.name]
            image_paths = sorted(
                (
                    path
                    for path in class_directory.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                ),
                key=lambda path: path.name,
            )
            self.samples.extend((path, label) for path in image_paths)

        if not self.samples:
            raise ValueError(
                f"No supported images found in {self.root_dir}. "
                "Add PNG, JPG, or JPEG images inside class folders."
            )

        self.transform = transform
        self._to_tensor = ToTensor()

    def __len__(self) -> int:
        """Return the number of supported image files."""
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load one image as an RGB tensor and return its integer class label."""
        image_path, label = self.samples[index]
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                transformed = (
                    self.transform(rgb_image)
                    if self.transform is not None
                    else rgb_image
                )
        except (OSError, ValueError) as error:
            raise OSError(f"Could not load image {image_path}: {error}") from error

        if isinstance(transformed, Tensor):
            image_tensor = transformed
        elif isinstance(transformed, Image.Image):
            image_tensor = self._to_tensor(transformed)
        else:
            raise TypeError(
                "The image transform must return a PyTorch tensor or a Pillow image."
            )

        return image_tensor, label
