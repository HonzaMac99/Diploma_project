import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps
from pathlib import Path
import cv2
import os
import json
from datetime import datetime, timezone
import time
from tqdm import tqdm

from brisque_eval import SAVE_STATS
from utils import *

DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
WEIGHTS_PATH = "data/model.pth"
IMG_NUM_RES = 1    # orig_res = [3000 x 4000] --> [224, 244] (fixed nima input size)
N_NEIGHBORS = 20
N_SIFT_FEATS = 1000
SAVE_STATS = False
OVERRIDE_JSON = True
SAVE_SCORE_EXIF = False

SHOW_IMAGES = True
MAX_IMAGES = 10 # maximum number of images to process (for debugging)

_sift = None
_bf = None

def get_sift_bf():
    global _sift, _bf
    if _sift is None:
        _sift = cv2.SIFT_create(nfeatures=N_SIFT_FEATS)  # SIFT algorithm with number of keypoints
    if _bf is None:
        _bf = cv2.BFMatcher() # keypoint matcher
    return _sift, _bf


# resize inspired by LB
def image_resize(image, max_d=1024):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if aspect_ratio < 1:
        new_height = max_d
        new_width = int(max_d * aspect_ratio)
    else:
        new_height = int(max_d / aspect_ratio)
        new_width = max_d
    image = cv2.resize(image, (new_width, new_height)) # cv2.resize takes (w, h) format
    return image


def compute_matches(descr_1, descr_2):
    _, bf = get_sift_bf()
    try:
        matches_12 = bf.knnMatch(descr_1, descr_2, k=2) # get cv2.DMatch objs
        matches_21 = bf.knnMatch(descr_2, descr_1, k=2)
    except:
        print("Something bad happened in compute_matches")
        return []

    dist_ratio_threshold = 0.7
    matches_12_robust = []
    matches_21_robust = []

    # Filter both directions first
    for m1_first, m1_second in matches_12:
        if m1_first.distance / m1_second.distance < dist_ratio_threshold:
            matches_12_robust.append(m1_first)

    for m2_first, m2_second in matches_21:
        if m2_first.distance / m2_second.distance < dist_ratio_threshold:
            matches_21_robust.append(m2_first)

    # Symmetry check
    matches_robust = []
    for m1 in matches_12_robust:
        for m2 in matches_21_robust:
            if m1.queryIdx == m2.trainIdx and m2.queryIdx == m1.trainIdx:
                matches_robust.append([m1])  # wrap if needed

    return matches_robust


def compute_sift_similarities(img_paths):
    sift, _ = get_sift_bf()

    n_images = len(img_paths)
    n_neighbors = N_NEIGHBORS

    features = {}  # keypoints and descriptors
    sift_scores = np.full((n_images, n_images), -np.inf)

    # todo: try different resolutions
    # todo: compute avg times
    for i, img_path in enumerate(tqdm(img_paths, desc="SIFT feats", unit="img")):
        img = image_resize(cv2.imread(img_path), 1024) # todo: why is he doing this? is 1024 max axis len??
        keypoints, descriptors = sift.detectAndCompute(img, None)
        features[i] = (keypoints, descriptors)

    for i, img_name in enumerate(tqdm(img_paths, desc="SIFT matches", unit="img&nbrs")):
        keypoints_i, descriptors_i = features[i]

        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                continue

            keypoints_j, descriptors_j = features[j]

            if len(keypoints_i) == 0 or len(keypoints_j) == 0:
                sift_score = 0
            else:
                matches = compute_matches(descriptors_i, descriptors_j)

                # todo: find the right number of sift feats for optimal time and score credibility
                # sift_score = compute_score(len(matches), len(keypoints_i), len(keypoints_j))
                sift_score = len(matches)

            sift_scores[i, j] = sift_score

    # todo: save scores
    # result_pth = "results/image_statistics.json"
    # result_pth = Path.cwd() / result_pth
    # result_pth.parent.mkdir(parents=True, exist_ok=True)

    # with open(result_pth, "w") as write_file:
    #     json.dump(data, write_file, indent=2, ensure_ascii=False)

    return sift_scores

############################################ Other experimental functions ##############################################

# inspired by LB
def compute_score(matches, keypoint1, keypoint2):
    score = 1000 * (matches / min(keypoint1, keypoint2))
    if score > 100:
        score = 100
    return score

# inspired by LB
def compute_similarities(img_paths, n_neighbors):
    global viewer
    sift, _ = get_sift_bf()
    plt.ion()

    print("Computing SIFT similarity scores:")
    n_images = len(img_paths)
    img_stats_list = []  # list to store results

    features = {}  # keypoints and descriptors
    sift_scores = np.ones((n_images, n_images)) * (-1)
    viewer.scores = sift_scores

    # todo: try different resolutions
    # todo: compute avg times
    for i, img_path in enumerate(img_paths):
        img = image_resize(cv2.imread(img_path), 1024)
        keypoints, descriptors = sift.detectAndCompute(img, None)
        features[i] = (keypoints, descriptors)

    for i, img_1_path in enumerate(img_paths):
        keypoints_i, descriptors_i = features[i]

        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                continue

            matches = []
            img_2_path = img_paths[j]
            keypoints_j, descriptors_j = features[j]

            if len(keypoints_i) == 0 or len(keypoints_j) == 0:
                sift_score = 0
            else:
                matches = compute_matches(descriptors_i, descriptors_j)
                sift_score = compute_score(len(matches), len(keypoints_i), len(keypoints_j))

            sift_scores[i, j] = sift_score
            print(f"({i+1}, {j+1}) score: {sift_score}, matches: {len(matches)} / ({len(keypoints_i)}, {len(keypoints_j)})")
            
            # todo: also some data?

            img_stats = {
                "id_1": i,
                "id_2": j,
                "img_1": str(img_1_path),
                "img_2": str(img_2_path),
                "sift_similarity_score": sift_score  # todo: list for more resolutions?
                # todo: times?
            }
            
            # todo: exif?

            if SHOW_IMAGES:
                viewer.idx1 = i
                viewer.idx2 = j
                viewer.show_current(interactive=False)

            img_stats_list.append(img_stats)

    imgs_stats = {
        "description": "SIFT statistics of image pairs from dataset computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(img_paths) if type(MAX_IMAGES) is not int else MAX_IMAGES,
        "num_resolutions": 1,
        "statistics": img_stats_list,
        # "features": features
    }

    # result_pth = "results/image_statistics.json"
    # result_pth = Path.cwd() / result_pth
    # result_pth.parent.mkdir(parents=True, exist_ok=True)

    # with open(result_pth, "w") as write_file:
    #     json.dump(data, write_file, indent=2, ensure_ascii=False)

    return imgs_stats

# region Other experimental functions

def get_scores_json(dataset_stats):
    n_images = dataset_stats["num_images"]
    scores = np.ones((n_images, n_images)) * (-1)
    img_stats_data = dataset_stats["statistics"]

    for stat in img_stats_data:
        i = stat["id_1"]
        j = stat["id_2"]
        if i < n_images and j < n_images:
            score = stat["sift_similarity_score"]
            scores[i, j] = score

    return scores


def print_scores(dataset_stats):
    n_imgs = dataset_stats["num_images"]
    scores = get_scores_json(dataset_stats)
    # avg_time = 0
    for i in range(n_imgs):
        for j in range(n_imgs):
            if scores[i, j] != -1:
                print(f"({i+1}, {j+1}) score: {scores[i, j]}")
                # avg_time += img_stats_data["time_rot"]
    # avg_time /= n_imgs
    # print(f"Average time: {avg_time}")

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
    viewer = ImageViewer(img_paths, scores, mode='dual', tool_name="Sift")

    method_stats = {}
    file_version_idx = 0
    file_name_base = "sift_stats"

    f_suffix = "" if file_version_idx == 0 else f"_{file_version_idx}"
    imgs_stats_path = Path(f"data/{file_name_base}{f_suffix}.json")

    if imgs_stats_path.exists() and not OVERRIDE_JSON:
        with open(imgs_stats_path, "r", encoding="utf-8") as f:
            method_stats = json.load(f)
        print_scores(method_stats)
    else:
        method_stats = compute_similarities(img_paths, N_NEIGHBORS)
        if SAVE_STATS:
            save_path = save_json_versioned(imgs_stats_path, file_version_idx, method_stats, override=OVERRIDE_JSON)
            print(f"Saved new data as: {save_path}")

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)
