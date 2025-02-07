from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


def fix_image_rotations(input_folder, output_folder):
    """
    Iterate through all images in the input folder, fix their rotation using ImageOps.exif_transpose,
    and save the fixed images to the output folder.

    Parameters:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where fixed images will be saved.
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    files = list(input_path.rglob("*"))
    for file_path in tqdm(files, desc="Processing images"):
        if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"]:
            relative_path = file_path.relative_to(input_path)
            output_file_path = output_path / relative_path

            # Ensure output directory exists
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                # Open image
                img = Image.open(file_path)

                # Apply EXIF-based rotation correction
                img_fixed = ImageOps.exif_transpose(img)

                # Save the fixed image
                img_fixed.save(output_file_path)
                # print(f"Fixed and saved: {output_file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")


# Example usage
input_folder = "/home/argo/Desktop/Projects/Veryfi/detector/data/dataset_trans/images"
output_folder = "/home/argo/Desktop/Projects/Veryfi/detector/data/dataset_trans/images_f"

fix_image_rotations(input_folder, output_folder)

"""
python -m src.etl.fix_rotations
python -m src.etl.preprocess
python -m src.etl.match_labels_imgs
python -m src.etl.split
python -m src.etl.copy_data_on_csv
python -m src.etl.yolo2coco

"""
