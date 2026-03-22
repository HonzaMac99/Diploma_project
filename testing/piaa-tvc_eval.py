from PIL import Image
import torch
from pathlib import Path
from tqdm import tqdm

# import sys, os
# sys.path.append(os.path.abspath("Personalized_Aesthetics/"))

import Personalized_Aesthetics.utils.parser as parser
from Personalized_Aesthetics.models.iaa import MultiModalIAA
from Personalized_Aesthetics.dataset import DEFAULT_TRANSFORM

from utils import *

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MODEL_PATH = Path.cwd() / "Personalized_Aesthetics/checkpoints/clip_B_3fc_aes.pth"

MAX_IMAGES = None

_piaa_tvc_model = None
_device = None

def build_piaa_model(model_path: Path, cuda: bool = True, seed: int | None = None):
    """
    Loads NIMA model with pretrained VGG16 base and returns it on the proper device.
    """
    print("[Piaa-tvc]: Building model")
    if seed is not None:
        torch.manual_seed(seed)

    # device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    opt = parser.parse_args()
    device = torch.device(opt.device)
    model = MultiModalIAA(opt, device)

    print("Loading checkpoint from {}".format(model_path))
    state_dict = torch.load(model_path, map_location='cpu')['model']
    model.load_state_dict(state_dict=state_dict, strict=True)

    model.to(device)
    model.eval()

    return model, device


def get_piaa_model():
    global _piaa_tvc_model, _device
    if _piaa_tvc_model is None:
        _piaa_tvc_model, _device = build_piaa_model(MODEL_PATH)
    return _piaa_tvc_model, _device


def compute_piaa_scores(img_paths):
    save_file_base = "piaa-tvc_scores"

    # ver_idx = 0
    # scores = load_results_versioned(paths_cfg, save_file_base, load_method="npz")
    scores = None

    if scores is None or len(scores) != len(img_paths):
        model, device = get_piaa_model()
        scores = []

        for img_path in tqdm(img_paths, desc="PIAA-TVC", unit="img"):
            img = Image.open(img_path).convert('RGB')
            # img = ImageOps.exif_transpose(img)  # apply EXIF orientation

            img = DEFAULT_TRANSFORM(img).unsqueeze(0).to(device)

            # forward
            pred = model({'img': img}).squeeze(0)
            template = torch.arange(1, 11, dtype=torch.float32).to(device)
            score = pred @ template
            scores.append(score)

    # # save scores after computation
    # save_results_versioned(paths_cfg, scores, save_file_base, save_method="npz")

    return scores


if __name__ == "__main__":
    method_name = "Nima"
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

    model, device = build_piaa_model(MODEL_PATH)

    scores = compute_piaa_scores(img_paths)
    viewer = ImageViewer(img_paths, scores, mode='single', tool_name=method_name)

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)
