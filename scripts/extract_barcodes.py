"""Crop barcodes."""
import os

import pandas as pd
from PIL import Image


# pylint: disable=fixme
def crop_images(
    input_folder: str = "data",
    output_folder: str = "data/barcodes",
    df_path: str = "data/annotations.tsv",
) -> None:
    """
    Crop images in the input folder and save them to the output folder.

    Args:
        input_folder (str): Path to the folder containing images to be cropped.
        output_folder (str): Path to the folder where cropped images will be saved.
        df_path (str): path to df.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df = pd.read_csv(df_path, sep="\t")

    for row in df.iterrows():
        file_path = os.path.join(input_folder, row[1]["filename"])
        x_from = row[1]["x_from"]
        y_from = row[1]["y_from"]
        x_end = x_from + row[1]["width"]
        y_end = y_from + row[1]["height"]
        if row[1]["width"] < row[1]["height"]:
            df = df.drop([row[0]])
            continue
            # TODO: FIX ALIGNMENT. Also need to fix minor distortions
        crop_box = (x_from, y_from, x_end, y_end)
        with Image.open(file_path) as img:
            cropped_img = img.crop(crop_box)
            output_path = os.path.join(output_folder, os.path.basename(file_path))
            cropped_img.save(output_path)
    df.to_csv(os.path.join(input_folder, "annotations_updated.tsv"), sep="\t")
    print("Cropping complete.")


if __name__ == "__main__":
    crop_images()
