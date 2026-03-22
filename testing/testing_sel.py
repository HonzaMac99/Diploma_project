import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from pathlib import Path
import json

from networkx.algorithms.tournament import score_sequence

from utils import ImageViewer, load_results_versioned

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/LIVEwild/Images/trainingImages/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/full/"

RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}


if __name__ == "__main__":

    dataset_path = Path(DATASET_PATH)
    img_paths = sorted(
        img_file for img_file in dataset_path.iterdir()
        if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS
    )

    paths_cfg = {
        "dataset_root": DATASET_ROOT,
        "dataset_path": DATASET_PATH,
        "results_root": RESULTS_ROOT
    }

    # sel_percent = 10
    # n_bins = 100 // sel_percent
    # selection = [1.0 if i % n_bins == 0 else 0.0 for i in range(len(img_paths))]

    # load the cluster data
    data_r = load_results_versioned(paths_cfg, "clusters_manual", load_method="json")
    cluster_imgs = [id for cluster in data_r["clusters"] for id in json.loads(cluster)]
    if "image_refs" in data_r:
        img_paths_pairs = []
        for pair_str in data_r["image_refs"]:
            name, idx = pair_str.split(",")
            img_paths_pairs.append((name.strip(), int(idx)))
        img_paths = [dataset_path / p for p, _ in sorted(img_paths_pairs, key=lambda x: x[1])]
    selection = [1 if id in cluster_imgs else 0 for id in range(len(img_paths))]

    viewer = ImageViewer(img_paths, scores=[], mode='select')
    viewer.update_selection(selection)

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)



