import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from pathlib import Path

from utils import ImageViewer

# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/LIVEwild/Images/trainingImages/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


if __name__ == "__main__":

    dataset_path = Path(DATASET_PATH)
    img_paths = sorted(
        img_file for img_file in dataset_path.iterdir()
        if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS
    )

    sel_percent = 10
    n_bins = 100 // sel_percent
    selection = [1.0 if i % n_bins == 0 else 0.0 for i in range(len(img_paths))]

    scores = []
    viewer = ImageViewer(img_paths, scores, mode='select')
    viewer.update_selection(selection)

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)



