import torch
from tqdm import tqdm
from scipy.spatial import distance
import skimage

from torch import randint
from torchvision import transforms
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment

from utils import *

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/full/"
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MAX_IMAGES = None # maximum number of images to process (for debugging)
N_NEIGHBORS = 20
IMG_NUM_RES = 1    # orig_res = [3000 x 4000] --> [224, 244] (fixed nima input size)

SHOW_IMAGES = True
SAVE_SCORE_EXIF = False

SAVE_STATS = True
RECOMPUTE = False
OVERRIDE = True

_clip_obj = None

def get_clip():
    global _clip_obj
    if _clip_obj is None:
        _clip_obj = CLIPImageQualityAssessment()
    return _clip_obj


def compute_clip_scores(paths_cfg, img_paths, save_scores=True, load_scores=True):
    save_file_base = "clip-iqa_scores"

    # ver_idx = 0
    scores = None
    if load_scores:
        scores = load_results_versioned(paths_cfg, save_file_base, load_method="npz")

    if scores is None or len(scores) != len(img_paths):
        clip_obj = get_clip()
        transform = transforms.ToTensor()
        scores = []

        for img_path in tqdm(img_paths, desc="CLIP-IQA", unit="img"):
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)  # apply EXIF orientation
            img = np.array(img)

            img_tfd = img_resize(img, max_d=1024, tf_option=1) # using cv2.resize()

            # note: local mean is the simplest method that KEEPS STATISTICS, so the IQA is more or less unbiased
            # we don't use cv2.resize, because interpolation can create artifacts and bias the img statistics
            # --> from experiments turned out that the effect is negligible

            img_tfd = transform(img_tfd).unsqueeze(0) # tf to [C, W, H] and then to [B, C, W, H], float32, [0, 1]
            clip_score = clip_obj(img_tfd)
            scores.append(clip_score)

        # save scores after computation
        if save_scores:
            save_results_versioned(paths_cfg, scores, save_file_base, save_method="npz")

    return scores


if __name__ == "__main__":
    method_name = "Clip-iqa"
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

    paths_cfg = {
        "dataset_root": DATASET_ROOT,
        "dataset_path": DATASET_PATH,
        "results_root": RESULTS_ROOT
    }

    scores = compute_clip_scores(paths_cfg, img_paths)
    viewer = ImageViewer(img_paths, scores, mode='single', tool_name=method_name)

    # plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)

