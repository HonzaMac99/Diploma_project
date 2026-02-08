import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import skimage
from PIL import Image, ImageOps
import piexif
import time
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from brisque import BRISQUE
from tqdm import tqdm
import cv2

from utils import *

DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
IMG_NUM_RES = 6    # orig_res = [3000 x 4000] --> [375 x 500] (4)

SAVE_STATS = False
OVERRIDE_JSON = True
SAVE_SCORE_EXIF = True
SHOW_IMAGES = True
MAX_IMAGES = 3 # maximum number of images to process

_brisque_obj = None

def get_brisque():
    global _brisque_obj
    if _brisque_obj is None:
        _brisque_obj = BRISQUE(url=False)
    return _brisque_obj


def compute_brisque_scores(dataset_path : Path, img_paths : list):
    brisque_obj = get_brisque()
    scores = []

    # iterator = tqdm(img_files, desc="BRISQUE", unit="img") if show_progress else img_files
    for img_path in tqdm(img_paths, desc="BRISQUE", unit="img"):
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # apply EXIF orientation
        img = np.array(img)

        dsampl_lvl = 3  # 1 -> 6s; 2 -> 1.65s; 3 -> 0.5s; 4 -> 0.16s per image
        img_new_h = img.shape[0] // 2 ** dsampl_lvl
        img_new_w = img.shape[1] // 2 ** dsampl_lvl
        img_norm = img.astype(np.float32) / 255.0 # torch.tensor(img) / 255.0
        img_tfd = skimage.transform.resize_local_mean(img_norm, output_shape=[img_new_h, img_new_w])
        # local mean is the simplest method that KEEPS STATISTICS, so the IQA is more or less unbiased
        # we don't use cv2.resize, because interpolation can create artifacts and bias the img statistics

        scores.append(brisque_obj.score(img_tfd))

    # todo: save scores

    return scores

# region Other experimental functions

def brisque_eval(img):
    # plt.ioff()

    brisque_obj = get_brisque()
    b_scores_resls = []
    times = []
    for i in range(IMG_NUM_RES):
        start_t = time.time()
        img_new_h = img.shape[0] // 2**i
        img_new_w = img.shape[1] // 2**i
        img_norm = img.astype(np.float32) / 255.0

        # todo: check all transformations
        tf_option = 1
        if tf_option == 1: # local mean from skimage, keeps statistics
            img_tfd = skimage.transform.resize_local_mean(img_norm, output_shape=[img_new_h, img_new_w])
            # img_tfd = skimage.transform.resize_local_mean(img.astype(np.float32) / 255.0, output_shape=[img_new_h, img_new_w])
        elif tf_option == 2: # pooling with np.mean
            img_tfd = skimage.measure.block_reduce(img_norm, block_size=(2**i, 2**i, 1), func=np.mean)
        elif tf_option == 3: # allegedly the FASTEST
            img_tfd = cv2.resize(img_norm, (img_new_w, img_new_h), interpolation=cv2.INTER_AREA)
            # img_tfd = cv2.resize(img.astype(np.float32), (img_new_w, img_new_h))
        elif tf_option == 4:
            img_t = torch.tensor(img_norm).permute(2, 0, 1).unsqueeze(0) # (1, C, H, W)
            img_tfd = F.avg_pool2d(img_t, kernel_size=2**i)
            img_tfd = img_tfd.squeeze(0).permute(1, 2, 0).numpy()
        else:
            exit(-1)

        img_tfd = np.asarray(img_tfd)

        b_scores_resls.append(brisque_obj.score(img_tfd))
        end_t = time.time()
        times.append(end_t-start_t)

        # plt.imshow(img_tfd)
        # plt.axis("off")
        # plt.show()
        # print(f"[{img_h / 2**i}, {img_w / 2**i}]: {end_t-start_t:.4f} s")

    return b_scores_resls, times


def compute_scores(img_paths):
    global viewer 
    plt.ion()

    print("Computing Brisque scores:")
    b_scores = []
    b_scores_raw = []
    img_stats_list = []  # list to store all data

    # print("Computing scores:", end="")
    for img_path in img_paths:
        img_idx = viewer.idx1

        img_raw = Image.open(img_path)
        img_rot = ImageOps.exif_transpose(img_raw)  # apply EXIF orientation

        img_raw = np.array(img_raw)
        img_rot = np.array(img_rot)

        b_scores_resl, b_times = brisque_eval(img_rot)
        b_scores.append(b_scores_resl)

        # give the viewer just the most precise score (biggest resolution)
        viewer.scores.append(b_scores_resl[0])

        scores_txt = [f"{x:>4.2f}" for x in b_scores_resl]
        print(f"{img_idx + 1}: scores: {scores_txt}")

        data = {
            "scores_rot": b_scores_resl,
            "scores_raw": b_scores_resl,
            "times_rot": b_times,
            "times_raw": []
        }

        img_stats = {
            "id": img_idx,
            "img": str(img_path),
            "brisque_score": b_scores_resl[0],
            "resolution": img_rot.shape,
            "data": data
        }

        if SAVE_SCORE_EXIF:
            save_exif_comment(img_path, b_scores_resl[0])

        if np.array_equal(img_raw, img_rot):
            b_scores_raw.append(b_scores_resl)
            print(f"           (no rotation)")
        else:
            b_scores_raw_resl, b_times_raw = brisque_eval(img_raw)
            b_scores_raw.append(b_scores_raw_resl)
            scores_diff = [b_scores[-1][i] - b_scores_raw[-1][i] for i in range(len(b_scores[-1]))]

            img_stats["data"]["scores_raw"] = b_scores_raw_resl
            img_stats["data"]["times_raw"] = b_times_raw

            scores_raw_txt = [f"{x:>4.2f}" for x in b_scores_raw[-1]]
            scores_diff_txt = [f"{x:>4.2f}" for x in scores_diff]
            print(f"    (raw): {scores_raw_txt}")
            print(f"   (diff): {scores_diff_txt}")
            print("")

        if SHOW_IMAGES:
            viewer.show_current(interactive=False)

        img_stats_list.append(img_stats)
        viewer.idx1 = (viewer.idx1 + 1) % viewer.n_images

    dataset_stats = {
        "description": "BRISQUE statistics of dataset images computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(img_stats_list),
        "num_resolutions": IMG_NUM_RES,
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
    n_resls = dataset_stats["num_resolutions"]
    times = np.zeros(n_resls)
    for i in range(n_imgs):
        img_stats_data = dataset_stats["statistics"][i]["data"]
        scores_txt = [f"{x:>5.2f}" for x in img_stats_data["scores_rot"]]
        print(f"{i + 1}: scores: {scores_txt}")

        if not img_stats_data["scores_raw"]:
            print(f"           (no rotation)")
        else:
            scores_diff = [img_stats_data["scores_rot"][i] - img_stats_data["scores_raw"][i] for i in range(n_resls)]

            scores_raw_txt = [f"{x:>5.2f}" for x in img_stats_data["scores_raw"]]
            scores_diff_txt = [f"{x:>5.2f}" for x in scores_diff]
            print(f"    (raw): {scores_raw_txt}")
            print(f"   (diff): {scores_diff_txt}")
            print("")

        times += img_stats_data["times_rot"]

    times /= n_imgs
    print(f"Average times:")
    for i in range(n_resls):
        print(f"  [{4000//2**i : >4}, {3000//2**i : >4}]: {times[i]:>8.4f} s")


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
    viewer = ImageViewer(img_paths, scores, mode='single', tool_name="Brisque")

    method_stats = {}
    file_version_idx = 1
    file_name_base = "brisque_stats"

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
