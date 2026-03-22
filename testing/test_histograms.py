import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/full/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/grenoble/full/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/namibie/namibie_corrected/"

RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MAX_IMAGES = None


dataset_path = Path(DATASET_PATH)
print(f"Dataset_path: {dataset_path}")

img_paths = sorted(
    img_path for img_path in dataset_path.iterdir()
    if img_path.suffix.lower() in IMG_EXTS
)
assert len(img_paths) > 0, "No images loaded!"

if type(MAX_IMAGES) is int:
    max_idx = min(len(img_paths), MAX_IMAGES)
    img_paths = img_paths[:max_idx]
n_images = len(img_paths)


# todo: the goal is to achieve same comparison as in paper Automatic Summarization from 2003, they use
#       4x4x4 3d histograms from rgb chaneels which are normalized in the YIQ space (probably Y channel)
for img_path in img_paths[2:]:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # RGB -> YIQ
    yiq = np.dot(img/255.0, [[0.299, 0.587, 0.114],
                            [0.596, -0.274, -0.322],
                            [0.211, -0.523, 0.312]])
    Y = yiq[:,:,0]

    # histogram equalization (global image)
    Y_eq = cv2.equalizeHist((Y*255).astype(np.uint8)) / 255.0
    yiq[:,:,0] = Y_eq

    # YIQ -> RGB
    rgb = np.dot(yiq, [[1, 0.956, 0.621],
                       [1, -0.272, -0.647],
                       [1, -1.106, 1.703]])
    rgb = np.clip(rgb,0,1)

    fig, axes = plt.subplots(2, 2)
    for ax, img_to_plot in zip(axes[0], [img, rgb]):
        ax.clear()
        ax.imshow(img_to_plot)
        ax.axis("off")

    for ax, img_to_plot in zip(axes[1], [img, rgb]):
        ax.clear()
        ax.hist(img_to_plot[:, :, 0].flatten(), bins=256, color='red', alpha=0.5)
        ax.hist(img_to_plot[:, :, 1].flatten(), bins=256, color='green', alpha=0.5)
        ax.hist(img_to_plot[:, :, 2].flatten(), bins=256, color='blue', alpha=0.5)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
