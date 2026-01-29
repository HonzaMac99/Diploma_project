import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import skimage
from PIL import Image, ImageOps
import piexif
import time
from tqdm import tqdm
import os
import json
from pathlib import Path
from datetime import datetime, timezone
import torchvision.models as models
import torchvision.transforms as transforms

DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/"
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
WEIGHTS_PATH = "data/model.pth"

IMG_NUM_RES = 1    # orig_res = [3000 x 4000] --> [224, 244] (fixed nima input size)
OVERRIDE_JSON = True
SAVE_SCORE_EXIF = False
SHOW_IMAGES = True
MAX_IMAGES = None # maximum number of images to process

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


def build_nima_model(weights_path: Path, cuda: bool = True, seed: int | None = None):
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

################################################# Main script function #################################################
def compute_nima_scores(dataset_path : Path, img_files : list, cuda=True):
    nima_model = get_nima_model()
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    scores = []

    nima_img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),           # Resize and crop (as in paper)
        transforms.ToTensor(),                # Convert to tensor and scale [0,255] -> [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # iterator = tqdm(img_files, desc="BRISQUE", unit="img") if show_progress else img_files
    for img_name in tqdm(img_files, desc="NIMA", unit="img"):
        img_path = dataset_path / img_name

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # apply EXIF orientation

        img = nima_img_transform(img)  # transform for Nima
        img = img.unsqueeze(dim=0)
        img = img.to(device)

        with torch.no_grad():
            out_f, out_class = nima_model(img)

        probs = out_class.view(-1) # flatten to [10]
        indices = torch.arange(1, 11, dtype=probs.dtype, device=probs.device) # class indices: 1..10
        nima_score = float((probs * indices).sum()) # weighted sum


        scores.append(nima_score)

    # todo: image batches
    # with torch.no_grad():
    #     out_f, out_class = nima_model(imgs)  # imgs: [B, C, H, W]
    # out_class = out_class.squeeze(-1) # ensure shape [B, 10]
    # weights = torch.arange(1, 11, device=out_class.device, dtype=out_class.dtype) # class indices 1..10
    # nima_scores = (out_class * weights).sum(dim=1) # weighted sum per image → [B]

    # todo: save scores

    return scores
########################################################################################################################

def nima_eval(img, cuda = True):
    start_t = time.time()
    nima_model = get_nima_model()
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    # for technical quality transform use   (skimage)
    # for aesthetic quality transform use   (torchvision)

    nima_img_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),           # Resize and crop (as in paper)
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

    probs = out_class.view(-1) # flatten to [10]
    indices = torch.arange(1, 11) # class indices: 1..10
    nima_score = float((probs * indices).sum()) # weighted sum

    end_t = time.time()
    nima_time = end_t-start_t

    return nima_score, nima_time


def compute_scores():
    global img_idx
    plt.ion()

    print("Computing NIMA scores:")
    nima_scores = []
    nima_scores_raw = []
    img_stats_list = []  # list to store all data

    # print("Computing scores:", end="")
    for i, img_name in enumerate(img_files):
        img_path = os.path.join(dataset_path, img_name)

        # get exif orientation info
        exif_dict = piexif.load(img_path)
        orientation = exif_dict["0th"].get(piexif.ImageIFD.Orientation, 1)
        print(f"Orientation: {orientation}")

        # img1 = skimage.io.imread(img_path)
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
            "img": img_path,
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
            show(nima_scores, img_idx)

        img_stats_list.append(img_stats)
        img_idx = (img_idx + 1) % len(img_files)

        if MAX_IMAGES is not None and img_idx == MAX_IMAGES:
            break

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
    # result_pth = os.path.join(os.getcwd(), result_pth)
    # os.makedirs(os.path.dirname(result_pth), exist_ok=True)

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


def save_exif_comment(image_path, quality_score):
    """Save your calculated quality score back to EXIF"""
    # Note: piexif is lightweight and more suitable directly for exif than pyexiv2
    exif_dict = piexif.load(image_path)

    # Store quality score in UserComment (or create custom tag)
    user_comment = f"Nima score: {quality_score:.3f}".encode('utf-8')
    exif_dict['Exif'][piexif.ExifIFD.UserComment] = user_comment

    # problematic key Exif.Photo.SceneType = 41279 with int value
    exif_dict['Exif'].pop(41729, None)
    # print(exif_dict['Exif'])

    # Write back to image
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, image_path)

    # # from Mr B: Rating and RatingPercent
    # interval_size = (highest_q - lowest_q) / 5
    # quality = q["aesthetic_quality"] * (1 - t_a_ratio) + q["technical_quality"] * t_a_ratio
    # rating = int(((quality - lowest_q) / interval_size)) + 1
    # rating_percent = int(((quality - lowest_q) / (interval_size * 5)) * 100)
    #
    # try:
    #     with pyexiv2.Image(img) as handle:
    #         meta = {'Exif.Image.Rating': rating,
    #                 'Exif.Image.RatingPercent': rating_percent}
    #         handle.modify_exif(meta)
    # except Exception:
    #     raise Exception


def load_exif_comment(img_path):
    try:
        exif_dict = piexif.load(img_path)
        comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment, b"")
        return comment.decode("utf-8") if comment else "No EXIF data"
    except Exception:
        return "No EXIF data"


def show(nima_scores, img_idx, interactive=False):
    global ax
    ax.clear()
    img_path = os.path.join(dataset_path, img_files[img_idx])

    # img1 = skimage.io.imread(img_path)
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # apply EXIF orientation
    img = np.array(img)

    nima_score = nima_scores[img_idx]

    exif_text = "EXIF: " + load_exif_comment(img_path)

    ax.imshow(img)
    n_images = len(img_files) if type(MAX_IMAGES) is not int else MAX_IMAGES
    ax.set_title(f"Nima score: {nima_score:.2f}   [{img_idx+1}/{n_images}]")
    ax.axis("off")

    # Text under the image
    ax.text(
        0.5, -0.02,
        exif_text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=12,
        wrap=True
    )

    if interactive:
        fig.canvas.draw_idle()
    else:
        fig.canvas.draw()
        plt.show()
        plt.pause(0.001)


def on_key(event, nima_scores):
    global img_idx
    n_images = len(img_files) if type(MAX_IMAGES) is not int else MAX_IMAGES
    if event.key == "d":
        img_idx = (img_idx + 1) % n_images
    elif event.key == "a":
        img_idx = (img_idx - 1) % n_images
    elif event.key == "q":
        plt.close(fig)
        return
    show(nima_scores, img_idx, interactive=True)


def save_json_versioned(path: Path, idx, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists() or OVERRIDE_JSON:
        final_path = path
    else:
        # get the file name without the '_version' suffix; edge_case: idx=0 but "_" is in the string
        file_name_base = path.stem.rsplit('_', 1)[0] if idx != 0 else path.stem
        idx += 1
        while True:
            final_path = path.with_name(f"{file_name_base}_{idx}{path.suffix}")
            if not final_path.exists():
                break
            idx += 1

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return final_path


if __name__ == "__main__":
    default_path = Path(DATASET_PATH) / "selected_r30" # os.path.join(PHOTOS_PATH, "selected_r30")
    # default_path = Path("/home/honzamac/Edu/m5/Projekt_D/datasets/LIVEwild/Images/trainingImages/")
    print(f"Dataset_path: {default_path}")
    # input_str = input("Dataset path: ")
    # input_path = Path(input_str).resolve()

    dataset_path = default_path
    # dataset_path = input_path if input_str != "" else default_path

    img_files = sorted(
        img_file for img_file in dataset_path.iterdir()
        if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTS
    )
    assert len(img_files) > 0, "No images loaded!"

    if type(MAX_IMAGES) is int:
        max_idx = min(len(img_files), MAX_IMAGES)
        img_files = img_files[:max_idx]
    n_images = len(img_files)

    img_idx = 0
    fig, ax = plt.subplots()

    dataset_stats = {}
    file_version_idx = 0
    file_name_base = "nima_stats"

    f_suffix = "" if file_version_idx == 0 else f"_{file_version_idx}"
    dataset_stats_path = Path(f"data/{file_name_base}{f_suffix}.json")

    if dataset_stats_path.exists() and not OVERRIDE_JSON:
        with open(dataset_stats_path, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)
        print_scores(dataset_stats)
    else:
        dataset_stats = compute_scores()
        save_path = save_json_versioned(dataset_stats_path, file_version_idx, dataset_stats)
        print(f"Saved new data as: {save_path}")

    img_idx = 0
    nima_scores = [x["data"]["score_rot"] for x in dataset_stats["statistics"]]
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, nima_scores))
    plt.ioff()
    show(nima_scores, img_idx)
    plt.show()
