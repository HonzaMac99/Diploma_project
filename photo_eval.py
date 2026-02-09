from pathlib import Path
import os
import time

from utils import ImageViewer

from brisque_eval import compute_brisque_scores
from nima_eval import compute_nima_scores
from sift_eval import compute_sift_similarities
from efnetv2_eval import compute_efnetv2_similarities

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/LIVEwild/Images/trainingImages/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/tid2013/distorted_images"

RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MAX_IMAGES = 3
SAVE_SCORE_EXIF = False

if __name__ == "__main__":

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

    # # testing batching effectivity with Nima
    # for batch_size in [1, 4, 16, 32]:
    #     start_t = time.time()
    #     compute_nima_scores(dataset_path, img_files, batch_size=batch_size)
    #     end_t = time.time()
    #     time_diff = end_t-start_t
    #     print(f"NIMA took {time_diff:.2f} s for batch size {batch_size}")

    paths_cfg = {
        "dataset_root": DATASET_ROOT,
        "dataset_path": DATASET_PATH,
        "results_root": RESULTS_ROOT
    }

    scores = {
        "brisque":  compute_brisque_scores(paths_cfg, img_paths),
        "nima":     compute_nima_scores(paths_cfg, img_paths),
        "sift":     compute_sift_similarities(paths_cfg, img_paths),
        "efnetv2":  compute_efnetv2_similarities(paths_cfg, img_paths)
    }
    # np.random.normal(loc=50.0, scale=10.0, size=(n_images)).tolist()

    viewer = ImageViewer(img_paths, scores, mode='dual')
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)
