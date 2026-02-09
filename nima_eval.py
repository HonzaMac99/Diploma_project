import torch
import torch.nn as nn
import time
from tqdm import tqdm

import torchvision.models as models
import torchvision.transforms as transforms

from utils import *

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

WEIGHTS_PATH = Path.cwd() / "data/model.pth"

MAX_IMAGES = 10 # maximum number of images to process
# NIMA has fixed input img size: orig_res = [3000 x 4000] --> [224, 244]
# turn of batch processing by setting batch_size = 1

SHOW_IMAGES = True
SAVE_SCORE_EXIF = False

SAVE_STATS = True
RECOMPUTE = False
OVERRIDE = True

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


def compute_nima_scores(paths_cfg, img_paths, batch_size = 32, cuda=True):
    save_file_base = "nima_scores"

    # ver_idx = 0
    scores = load_results_versioned(paths_cfg, save_file_base, load_method="npz")

    if scores is None or len(scores) != len(img_paths):
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

        for img_path in tqdm(img_paths, desc="NIMA", unit="img"):

            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)  # apply EXIF orientation

            img = nima_img_transform(img)  # transform for Nima
            batch.append(img)

            if len(batch) == batch_size:
                with torch.no_grad():
                    batch_scores = process_nima_batch(batch, nima_model, device, indices)
                    scores.extend(batch_scores)
                    batch.clear()

        # last partial batch
        if batch:
            batch_scores = process_nima_batch(batch, nima_model, device, indices)
            scores.extend(batch_scores)

        # save scores after computation
        save_results_versioned(paths_cfg, scores, save_file_base, save_method="npz")

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
        # get exif orientation info
        exif_dict = piexif.load(str(img_path))
        orientation = exif_dict["0th"].get(piexif.ImageIFD.Orientation, 1)
        print(f"Orientation: {orientation}")

        img_raw = Image.open(img_path)
        img_rot = ImageOps.exif_transpose(img_raw)  # apply EXIF orientation

        nima_score, nima_time = nima_eval(img_rot)
        nima_scores.append(nima_score)

        print(f"{i + 1}: score: {nima_score}")

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
            print(f"   (no rotation)")
            print("--------------------------------")
        else:
            nima_score_raw, nima_time_raw = nima_eval(img_raw)
            nima_scores_raw.append(nima_score_raw)
            score_diff = nima_scores[-1] - nima_scores_raw[-1]

            img_stats["data"]["score_raw"] = nima_score_raw
            img_stats["data"]["time_raw"] = nima_time_raw

            print(f"    (raw): {nima_score_raw}")
            print(f"   (diff): {score_diff}")
            print("--------------------------------")

        img_stats_list.append(img_stats)

        if SHOW_IMAGES:
            viewer.idx = i
            viewer.show_current(interactive=False)

    # todo: image batches?

    dataset_stats = {
        "description": "BRISQUE statistics of dataset images computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(img_stats_list),
        "num_resolutions": 1,
        "statistics": img_stats_list
    }

    return dataset_stats


def get_scores_json(method_stats):
    return [x["data"]["score_rot"] for x in method_stats["statistics"]]


def print_scores(result_stats):
    print("Printing Nima scores:")
    n_imgs = result_stats["num_images"]
    avg_time = 0
    for i in range(n_imgs):
        img_stats_data = result_stats["statistics"][i]["data"]
        print(f"{i + 1}: scores: {img_stats_data["score_rot"]}")

        if img_stats_data["score_rot"] == img_stats_data["score_raw"]:
            print(f"   (no rotation)")
            print("--------------------------------")
        else:
            score_diff = img_stats_data["score_rot"] - img_stats_data["score_raw"]
            print(f"    (raw): {img_stats_data["score_raw"]}")
            print(f"   (diff): {score_diff}")
            print("--------------------------------")

        avg_time += img_stats_data["time_rot"]

    avg_time /= n_imgs
    print(f"Average time: {avg_time}")
# endregion

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

    scores = []
    viewer = ImageViewer(img_paths, scores, mode='single', tool_name=method_name)

    method_stats = {}
    file_name_base = "nima_stats_experimental"
    ver_idx = None

    if not RECOMPUTE:
        method_stats = load_results_versioned(paths_cfg, file_name_base, ver_idx=ver_idx, load_method="json")

    if method_stats:
        print_scores(method_stats)
        viewer.scores = get_scores_json(method_stats)
    else:
        method_stats = compute_scores(img_paths)
        if SAVE_STATS:
            save_path = save_results_versioned(paths_cfg, method_stats, file_name_base, save_method="json",
                                               override_last=OVERRIDE)
            print(f"Saved new data as: {save_path}")

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)
