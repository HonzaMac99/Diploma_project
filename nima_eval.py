import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import skimage
from PIL import Image, ImageOps
import piexif
import time

from torchgen.model import BaseTy
from tqdm import tqdm
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import torchvision.models as models
import torchvision.transforms as transforms

from brisque_eval import SAVE_STATS
from utils import *

DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
WEIGHTS_PATH = "data/model.pth"

# NIMA has fixed input img size: orig_res = [3000 x 4000] --> [224, 244]
SAVE_STATS = False
OVERRIDE_JSON = True
SAVE_SCORE_EXIF = False
SHOW_IMAGES = True
MAX_IMAGES = 10 # maximum number of images to process
BATCHES = False

_nima_model = None

class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""

    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax(dim=1))

    def forward(self, x):
        out_f = self.features(x)
        out = out_f.view(out_f.size(0), -1)
        out = self.classifier(out)
        return out_f, out


def build_nima_model(weights_path: str, cuda: bool = True, seed: int | None = None):
    """
    Loads NIMA model with pretrained VGG16 base and returns it on the proper device.
    """
    print("[Nima]: Building model")
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model = NIMA(base_model)

    weights_path = Path(weights_path).resolve()
    state_dict = torch.load(weights_path, map_location="cpu") # first load to cpu, then to gpu with the model all at once
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    # model.requires_grad_(False)

    return model, device


def get_nima_model():
    global _nima_model
    if _nima_model is None:
        _nima_model, _ = build_nima_model(WEIGHTS_PATH)
    return _nima_model


def process_nima_batch(batch, model, device, indices):
    imgs = torch.stack(batch).to(device) # imgs: [B, C, H, W]

    with torch.no_grad():
        _, out_batch_classes = model(imgs)  # [B, 10]
    # out_class = out_class.squeeze(-1) # ensure shape [B, 10]

    # expected value: sum(p_i * i)
    batch_scores = (out_batch_classes * indices).sum(dim=1) # weighted sum per image → [B]
    return batch_scores.cpu().tolist()


def compute_nima_scores(dataset_path : Path, img_files : list, batch_size : int = 32, cuda=True):
    nima_model = get_nima_model()
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    nima_img_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.RandomCrop(224),           # Resize and crop (as in paper)
        transforms.CenterCrop(224),           # center crop for same scores
        transforms.ToTensor(),                # Convert to tensor and scale [0,255] -> [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    scores = []
    batch = []
    indices = torch.arange(1, 11, device=device).float()  # class indices: 1..10

    for img_name in tqdm(img_files, desc="NIMA", unit="img"):
        img_path = dataset_path / img_name

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # apply EXIF orientation

        img = nima_img_transform(img)  # transform for Nima
        batch.append(img)

        # img = img.unsqueeze(dim=0)
        # img = img.to(device)
        # with torch.no_grad():
        #     out_f, out_class = nima_model(img) # [B, C, H, W]

        if len(batch) == batch_size:
            with torch.no_grad():
                batch_scores = process_nima_batch(batch, nima_model, device, indices)
                scores.extend(batch_scores)
                batch.clear()

    # last partial batch
    if batch:
        batch_scores = process_nima_batch(batch, nima_model, device, indices)
        scores.extend(batch_scores)

    # todo: save scores

    return scores

# region Other experimental functions

def nima_eval(img, cuda = True):
    start_t = time.time()
    nima_model = get_nima_model()
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # for technical quality transform use   (skimage)
    # for aesthetic quality transform use   (torchvision)

    nima_img_transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.RandomCrop(224),           # Resize and crop (as in paper)
        transforms.CenterCrop(224),           # center crop for same scores
        transforms.ToTensor(),                # Convert to tensor and scale [0,255] -> [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    # possible alternatives to try:
    # (1) transforms.Resize((224, 224)),      # Resize directly to 224x224
    # (2) transforms.CenterCrop(224),         # Deterministic crop instead to random
    # todo: try these and put into report

    img = nima_img_transform(img)  # transform for Nima
    img = img.unsqueeze(dim=0)
    img = img.to(device)

    with torch.no_grad():
        out_f, out_class = nima_model(img)

    probs = out_class.cpu().view(-1) # flatten to [10]
    indices = torch.arange(1, 11) # class indices: 1..10
    nima_score = float((probs * indices).sum()) # weighted sum

    end_t = time.time()
    time_diff = end_t-start_t

    return nima_score, time_diff


def compute_scores(img_paths):
    global viewer
    plt.ion()

    print("Computing NIMA scores:")
    nima_scores = []
    nima_scores_raw = []
    img_stats_list = []  # list to store all data

    viewer.scores = nima_scores

    for i, img_path in enumerate(img_paths):
        img_idx = viewer.idx1

        # get exif orientation info
        exif_dict = piexif.load(str(img_path))
        orientation = exif_dict["0th"].get(piexif.ImageIFD.Orientation, 1)
        print(f"Orientation: {orientation}")

        img_raw = Image.open(img_path)
        img_rot = ImageOps.exif_transpose(img_raw)  # apply EXIF orientation

        nima_score, nima_time = nima_eval(img_rot)
        nima_scores.append(nima_score)

        print(f"{img_idx + 1}: score: {nima_score}")

        data = {
            "score_rot": nima_score,
            "score_raw": nima_score,
            "time_rot": nima_time,
            "time_raw": 0.
        }

        img_stats = {
            "id": i,
            "img": str(img_path),
            "nima_score": nima_score,
            "resolution": [224, 224, 3],
            "data": data
        }

        if SAVE_SCORE_EXIF:
            save_exif_comment(img_path, nima_score)

        if orientation == 1:
            nima_scores_raw.append(nima_score)
            print(f"           (no rotation)")
        else:
            nima_score_raw, nima_time_raw = nima_eval(img_raw)
            nima_scores_raw.append(nima_score_raw)
            score_diff = nima_scores[-1] - nima_scores_raw[-1]

            img_stats["data"]["score_raw"] = nima_score_raw,
            img_stats["data"]["time_raw"] = nima_time_raw

            print(f"    (raw): {nima_score_raw}")
            print(f"   (diff): {score_diff}")
            print("")

        if SHOW_IMAGES:
            viewer.show_current(interactive=False)

        img_stats_list.append(img_stats)
        viewer.idx1 = (viewer.idx1 + 1) % viewer.n_images

    # todo: image batches?

    dataset_stats = {
        "description": "BRISQUE statistics of dataset images computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(img_stats_list),
        "num_resolutions": 1,
        "statistics": img_stats_list
    }

    # result_pth = "results/image_statistics.json"
    # result_pth = Path.cwd() / result_pth
    # result_pth.parent.mkdir(parents=True, exist_ok=True)

    # with open(result_pth, "w") as write_file:
    #     json.dump(data, write_file, indent=2, ensure_ascii=False)

    return dataset_stats


def print_scores(dataset_stats):
    n_imgs = dataset_stats["num_images"]
    avg_time = 0
    for i in range(n_imgs):
        img_stats_data = dataset_stats["statistics"][i]["data"]
        print(f"{i + 1}: scores: {img_stats_data["score_rot"]}")

        if img_stats_data["score_rot"] == img_stats_data["score_raw"]:
            print(f"           (no rotation)")
        else:
            score_diff = img_stats_data["score_rot"] - img_stats_data["scores_raw"]
            print(f"    (raw): {img_stats_data["score_raw"]}")
            print(f"   (diff): {score_diff}")
            print("")

        avg_time += img_stats_data["time_rot"]

    avg_time /= n_imgs
    print(f"Average time: {avg_time}")


# endregion

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

    scores = []
    viewer = ImageViewer(img_paths, scores, mode='single', tool_name="Nima")

    method_stats = {}
    file_version_idx = 0
    file_name_base = "nima_stats"

    f_suffix = "" if file_version_idx == 0 else f"_{file_version_idx}"
    dataset_stats_path = Path(f"results/{file_name_base}{f_suffix}.json")

    if dataset_stats_path.exists() and not OVERRIDE_JSON:
        with open(dataset_stats_path, "r", encoding="utf-8") as f:
            method_stats = json.load(f)
        print_scores(method_stats)
    else:
        method_stats = compute_scores(img_paths)
        if SAVE_STATS:
            save_path = save_json_versioned(dataset_stats_path, file_version_idx, method_stats, override=OVERRIDE_JSON)
            print(f"Saved new data as: {save_path}")

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)
