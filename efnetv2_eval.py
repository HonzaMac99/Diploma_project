import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps
from pathlib import Path
import os
import json
from datetime import datetime, timezone
import time
from tqdm import tqdm
from scipy.spatial import distance

import tensorflow as tf
from keras import preprocessing
from keras.applications.efficientnet_v2 import preprocess_input, EfficientNetV2B1

from utils import *

DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
WEIGHTS_PATH = "data/efficientnetv2-b1.h5"
IMG_NUM_RES = 1    # orig_res = [3000 x 4000] --> [224, 244] (fixed nima input size)
N_NEIGHBORS = 20

SAVE_STATS = False
OVERRIDE_JSON = True
SAVE_SCORE_EXIF = False

SHOW_IMAGES = True
MAX_IMAGES = 3 # maximum number of images to process (for debugging)

################################################# Main script function #################################################
def compute_efnetv2_similarities(dataset_path, img_files):

    # todo: resolve tf incompatible gpu drivers
    # gpus = tf.config.list_physical_devices('GPU')
    # device_str = '/gpu:0' if gpus else '/cpu:0'
    device_str = '/cpu:0'

    contents = []

    with tf.device(device_str):
        # todo: get our own weights
        # class_model = EfficientNetV2B1(weights="imagenet")
        model = EfficientNetV2B1(weights=WEIGHTS_PATH)

        for i, img_name in enumerate(tqdm(img_files, desc="EFNETV2 contents", unit="img")):
            # todo: compute times
            img_path = dataset_path / img_name
            img_tfd = preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(240, 240))
            img_contents = compute_contents(img_tfd, model)

            contents.append(img_contents)

    n_images = len(img_files)
    n_neighbors = N_NEIGHBORS
    efnetv2_scores = np.full((n_images, n_images), -np.inf)

    for i, img_name in enumerate(tqdm(img_files, desc="EFNETV2 similarities", unit="img&nbrs")):
        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                if j+1 == MAX_IMAGES:
                    break
                else:
                    continue

            efnetv2_score = (1 - distance.cdist([contents[i]], [contents[j]], 'cosine').item()) * 100 # * 15
            efnetv2_scores[i, j] = efnetv2_score

    # todo: save scores
    # result_pth = "results/image_statistics.json"
    # result_pth = Path.cwd() / result_pth
    # result_pth.parent.mkdir(parents=True, exist_ok=True)

    # with open(result_pth, "w") as write_file:
    #     json.dump(data, write_file, indent=2, ensure_ascii=False)

    return efnetv2_scores

# region Other experimental functions
# inspired by LB
def compute_contents(img, model):
    t = preprocessing.image.img_to_array(img)
    t = np.expand_dims(t, axis=0)
    t = preprocess_input(t)
    f = model.predict(t, verbose=0, batch_size=8)
    f = f.tolist()
    return f[0]

# inspired by LB
def compute_imgs_contents(target_path, recompute=False):
    path = Path(target_path)

    if path.exists() and not recompute:
        return np.load(path) # load the .npz file

    # todo: resolve tf incompatible gpu drivers
    device_str = '/cpu:0'
    # gpus = tf.config.list_physical_devices('GPU')
    # device_str = '/gpu:0' if gpus else '/cpu:0'

    with tf.device(device_str):
        # todo: get our own weights
        # class_model = EfficientNetV2B1(weights="imagenet")
        model = EfficientNetV2B1(weights="data/efficientnetv2-b1.h5")
        dataset_stats = {
            "ids": [],
            "img_paths": [],
            "contents": []
        }
        for i, img_name in enumerate(img_paths):
            # todo: compute times
            img_path = dataset_path / img_name
            img_tfd = preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(240, 240))
            img_contents = compute_contents(img_tfd, model)

            dataset_stats["ids"].append(i)
            dataset_stats["img_paths"].append(img_paths[i])
            dataset_stats["contents"].append(img_contents)

        # with open(Path.cwd() / target_path), "w") as write_file:
        #     json.dump(content_list, write_file, indent=2)

        np.savez(target_path, ids=dataset_stats["ids"],
                              img_paths=dataset_stats["img_paths"],
                              contents=dataset_stats["contents"])
    return dataset_stats

# inspired by LB
def compute_similarities(img_paths, n_neighbors):
    global viewer
    plt.ion()

    path = Path("results/efnetv2_contents.npz")
    if path.exists():
        imgs_contents = np.load(path)["contents"]
    else:
        print("Computing EfNetV2 contents:")
        imgs_contents = compute_imgs_contents(path)["contents"]

    print("Computing EfNetV2 similarity scores:")
    n_images = len(img_paths)
    img_stats_list = []  # list to store results
    efnetv2_scores = np.ones((n_images, n_images)) * (-1)
    viewer.scores = efnetv2_scores

    for i, img_1_path in enumerate(img_paths):
        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                continue

            img_2_path = img_paths[j]

            efnetv2_score = (1 - distance.cdist([imgs_contents[i]], [imgs_contents[j]], 'cosine').item()) * 100 # 15
            efnetv2_scores[i, j] = efnetv2_score

            print(f"({i+1}, {j+1}) score: {efnetv2_score}")

            # todo: also some data?

            img_stats = {
                "id_1": i,
                "id_2": j,
                "img_1": str(img_1_path),
                "img_2": str(img_2_path),
                "efnetv2_similarity_score": efnetv2_score  # todo: list for more resolutions?
                # todo: times?
            }

            # todo: exif?

            if SHOW_IMAGES:
                viewer.idx1 = i
                viewer.idx2 = j
                viewer.show_current(interactive=False)

            img_stats_list.append(img_stats)

    imgs_stats = {
        "description": "EffectiveNetV2-B1 statistics of image pairs from dataset computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(img_paths),
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


def get_scores_json(dataset_stats):
    n_images = dataset_stats["num_images"]
    scores = np.ones((n_images, n_images)) * (-1)
    img_stats_data = dataset_stats["statistics"]

    for stat in img_stats_data:
        i = stat["id_1"]
        j = stat["id_2"]
        if i < n_images and j < n_images:
            score = stat["efnetv2_similarity_score"]
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
    file_version_idx = 1
    file_name_base = "efnetv2_stats"

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
