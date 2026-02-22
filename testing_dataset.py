import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import piexif

from pathlib import Path
import json
import csv
from datetime import datetime, timezone

from utils import ImageViewer

DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/jk/namibie_corrected/"
CSV_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/jk/"

# DATASET_PATH = "/datagrid/Medical/archive/notebook_backup/digital_photography/cdroms/year2019/corrected/281_namibie/"
# CSV_PATH = "/datagrid/Medical/archive/notebook_backup/digital_photography/"
CSV_FILE = "photo_dataset20240119"

# CSV_FILE = "photo_dataset20240122train.csv"
# photo_dataset20240122test.csv, selected_photos20240119.txt

IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


if __name__ == "__main__":

    dataset_path = Path(DATASET_PATH)
    img_paths = sorted(
        img_file for img_file in dataset_path.iterdir()
        if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS
    )
    assert len(img_paths) > 0, "No images loaded!"

    n_sel = 0
    selection = [0.0] * len(img_paths)
    csv_path = Path(CSV_PATH) / CSV_FILE
    with open(csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            img_path = dataset_path / row[-2]
            img_sel = int(row[-1])
            if img_path in img_paths:
                idx = img_paths.index(img_path)
                print(img_path, idx, img_sel)
                selection[idx] = img_sel
                if img_sel:
                    n_sel += 1

    print(f"{n_sel} / {len(selection)}")

    scores = []
    viewer = ImageViewer(img_paths, scores, mode='select')
    viewer.update_selection(selection)

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)

