import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from pathlib import Path

# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/LIVEwild/Images/trainingImages/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


def plot_images_grid(img_files, frame_colors, start_idx):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))

    for ax in axes.flat:
        if start_idx < len(img_files):
            f_color = frame_colors[start_idx]
            img_path = img_files[start_idx]

            with Image.open(img_path) as img:
                img = ImageOps.exif_transpose(img)  # apply EXIF orientation
                img = np.asarray(img)
        else:
            color = "black"
            img = np.zeros((64, 64, 3))
        start_idx += 1

        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        h, w = img.shape[:2]
        rect = Rectangle(
            (0, 0), w, h,
            linewidth=8,
            edgecolor=f_color,
            facecolor='none'
        )
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()
    # plt.close(fig)


dataset_path = Path(DATASET_PATH)
img_paths = sorted(
    img_file for img_file in dataset_path.iterdir()
    if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS
)

colors = ['green' if i % 2 == 0 else 'red' for i in range(len(img_paths))]

count = 0
while count < len(img_paths):
    plot_images_grid(img_paths, colors, count)
    count += 16



