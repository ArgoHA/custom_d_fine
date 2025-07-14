import multiprocessing as mp
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from tqdm import tqdm


def convert_image_to_jpg(filepath: Path) -> None:
    """
    Convert a single image file to .jpg format
    """
    if filepath.suffix.lower() in [".tif", ".jpeg", ".png", ".tiff", ".heic"]:
        try:
            image = Image.open(filepath).convert("RGB")
            image = ImageOps.exif_transpose(image)  # fix rotation
        except OSError:
            print("Can't open, deleting:", filepath.name)
            # filepath.unlink()
            return
        image.save(filepath.with_suffix(".jpg"))
        filepath.unlink()

    elif filepath.suffix.lower() != ".jpg":
        print("NOT converted:", filepath.stem)
        # filepath.unlink()


def convert_images_to_jpg(dir_path: Path, num_threads: int) -> None:
    """
    Convert all images in a directory to .jpg format
    """
    all_files = [f.stem for f in dir_path.iterdir() if not f.name.startswith(".")]

    with mp.Pool(processes=num_threads) as pool:
        filepaths = [
            filepath for filepath in dir_path.glob("*") if not filepath.name.startswith(".")
        ]

        for _ in tqdm(pool.imap_unordered(convert_image_to_jpg, filepaths)):
            pass

    jpg_files = [f.stem for f in dir_path.iterdir() if f.suffix.lower() == ".jpg"]
    lost_files = set(all_files) - set(jpg_files)

    if not lost_files:
        logger.info(
            f"In {dir_path}, All files were converted to .jpg, total amount: {len(jpg_files)}"
        )
    else:
        logger.warning(
            f"In {dir_path}, Not converted to .jpg, amount: {len(lost_files)}, files: {lost_files}"
        )


def remove_empty_labels(dir_path: Path) -> None:
    label_dir = dir_path.parent / "labels"
    no_label_dir = label_dir / "noimages"
    no_label_dir.mkdir(exist_ok=True)  # 确保目标目录存在

    labels = [f.name for f in label_dir.iterdir() if not f.name.startswith(".")]
    counter = 0
    for label in labels:
        label_file = label_dir / label
        if not label_file.stat().st_size:
            target_path = no_label_dir / label_file.name
            label_file.rename(target_path)
            counter += 1

    logger.info(f"Moved {counter} labels without images to {no_label_dir} floder")


def remove_images_without_labels(dir_path: Path) -> None:
    """
    将没有对应标签文件的图像移动到 ./nolabels 目录。
    :param dir_path: 图像目录路径（如 images/train 或 images/val）
    """
    label_dir = dir_path.parent / "labels"
    no_label_dir = dir_path / "nolabels"
    no_label_dir.mkdir(exist_ok=True)  # 确保目标目录存在

    image_files = [f for f in dir_path.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    counter = 0
    for image_file in image_files:
        label_file = label_dir / f"{image_file.stem}.txt"
        if not label_file.exists() or label_file.stat().st_size == 0:
            target_path = no_label_dir / image_file.name
            image_file.rename(target_path)
            counter += 1
    
    logger.info(f"Moved {counter} images without labels to {no_label_dir} floder")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    paths = {"root_path": Path(cfg.train.data_path) / "images"}
    if test_path := cfg.train.path_to_test_data:
        paths["test_path"] = Path(test_path)
    for _, data_path in paths.items():
        if (data_path).exists():
            # 将图片转换为jpg
            convert_images_to_jpg(data_path, cfg.train.num_workers)
            if (data_path.parent / "labels").exists():
                remove_empty_labels(data_path)
                remove_images_without_labels(data_path)


if __name__ == "__main__":
    register_heif_opener()
    main()
