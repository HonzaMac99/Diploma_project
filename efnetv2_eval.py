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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
WEIGHTS_PATH = "data/efficientnetv2-b1.h5"
IMG_NUM_RES = 1    # orig_res = [3000 x 4000] --> [224, 244] (fixed nima input size)
N_NEIGHBORS = 20
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
    # result_pth = os.path.join(os.getcwd(), result_pth)
    # os.makedirs(os.path.dirname(result_pth), exist_ok=True)

    # with open(result_pth, "w") as write_file:
    #     json.dump(data, write_file, indent=2, ensure_ascii=False)

    return efnetv2_scores
########################################################################################################################


### Code inspired by Mr. B #############################################################################################
def compute_contents(img, model):
    t = preprocessing.image.img_to_array(img)
    t = np.expand_dims(t, axis=0)
    t = preprocess_input(t)
    f = model.predict(t, verbose=0, batch_size=8)
    f = f.tolist()
    return f[0]


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
        for i, img_name in enumerate(img_files):
            # todo: compute times
            img_path = os.path.join(dataset_path, img_name)
            img_tfd = preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(240, 240))
            img_contents = compute_contents(img_tfd, model)

            dataset_stats["ids"].append(i)
            dataset_stats["img_paths"].append(img_files[i])
            dataset_stats["contents"].append(img_contents)

        # with open(os.path.join(os.getcwd(), target_path), "w") as write_file:
        #     json.dump(content_list, write_file, indent=2)

        np.savez(target_path, ids=dataset_stats["ids"],
                              img_paths=dataset_stats["img_paths"],
                              contents=dataset_stats["contents"])
    return dataset_stats
########################################################################################################################


def compute_similarities():
    global img_idx, matches
    plt.ion()

    print("Computing EfNetV2 similarity scores:")
    path = Path("data/efnetv2_contents.npz")
    if path.exists():
        imgs_contents = np.load(path)["contents"]
    else:
        imgs_contents = compute_imgs_contents(path)["contents"]

    n_images = len(img_files)
    n_neighbors = N_NEIGHBORS
    img_stats_list = []  # list to store results
    efnetv2_scores = np.ones((n_images, n_images)) * (-1)

    for i, img_name in enumerate(img_files):
        img_1_path = os.path.join(dataset_path, img_name)
        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                if j+1 == MAX_IMAGES:
                    break
                else:
                    continue
            img_2_path = os.path.join(dataset_path, img_files[j])

            efnetv2_score = (1 - distance.cdist([imgs_contents[i]], [imgs_contents[j]], 'cosine').item()) * 15
            efnetv2_scores[i, j] = efnetv2_score

            print(f"({i+1}, {j+1}) score: {efnetv2_score}")

            # todo: also some data?

            img_stats = {
                "id_1": i,
                "id_2": j,
                "img_1": img_1_path,
                "img_2": img_2_path,
                "efnetv2_similarity_score": efnetv2_score  # todo: list for more resolutions?
                # todo: times?
            }

            # todo: exif?

            if SHOW_IMAGES:
                show(efnetv2_scores, (i, j))

            img_stats_list.append(img_stats)

            if type(MAX_IMAGES) is int and j+1 == MAX_IMAGES:
                break
        if type(MAX_IMAGES) is int and i+1 == MAX_IMAGES:
            break

    imgs_stats = {
        "description": "EffectiveNetV2-B1 statistics of image pairs from dataset computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(img_files) if type(MAX_IMAGES) is not int else MAX_IMAGES,
        "num_resolutions": 1,
        "statistics": img_stats_list,
        # "features": features
    }

    # result_pth = "results/image_statistics.json"
    # result_pth = os.path.join(os.getcwd(), result_pth)
    # os.makedirs(os.path.dirname(result_pth), exist_ok=True)

    # with open(result_pth, "w") as write_file:
    #     json.dump(data, write_file, indent=2, ensure_ascii=False)

    return imgs_stats

########################################################################################################################


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


def show(sift_scores, img_idxs, interactive=False):
    global fig, axes

    assert len(img_idxs) == 2, f"Size of img idxs has to be 2: {img_idxs}"

    for img_idx, ax in zip(img_idxs, axes):
        img_path = os.path.join(dataset_path, img_files[img_idx])

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # apply EXIF orientation
        img = np.array(img)

        ax.clear()
        ax.imshow(img)
        ax.axis("off")

    score = sift_scores[img_idxs[0]][img_idxs[1]]

    n_images = len(img_files) if type(MAX_IMAGES) is not int else MAX_IMAGES
    fig.suptitle(
        f"EfficientNetV2-B1 score: {score:.2f}   [{img_idxs[0]+1}, {img_idxs[1]+1} | {n_images}]",
        fontsize=14
    )
    if interactive:
        fig.canvas.draw_idle()
    else:
        fig.canvas.draw()
        plt.show()
        plt.pause(0.001)


def on_key(event, scores):
    global img_idx_1, img_idx_2
    n_images = len(img_files) if type(MAX_IMAGES) is not int else MAX_IMAGES
    if event.key == "d":
        img_idx_1 = (img_idx_1 + 1) % n_images
    elif event.key == "a":
        img_idx_1 = (img_idx_1 - 1) % n_images
    elif event.key == "w":
        img_idx_2 = (img_idx_2 + 1) % n_images
    elif event.key == "s":
        img_idx_2 = (img_idx_2 - 1) % n_images
    elif event.key == "q":
        plt.close(fig)
        return

    img_idxs = (img_idx_1, img_idx_2)
    show(scores, img_idxs, interactive=True)


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
        if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS
    )
    assert len(img_files) > 0, "No images loaded!"

    if type(MAX_IMAGES) is int:
        max_idx = min(len(img_files), MAX_IMAGES)
        img_files = img_files[:max_idx]
    n_images = len(img_files)

    img_idx_1 = 0
    img_idx_2 = 1
    fig, axes = plt.subplots(1, 2)
    matches = []

    dataset_stats = {}
    file_version_idx = 1
    file_name_base = "efnetv2_stats"

    f_suffix = "" if file_version_idx == 0 else f"_{file_version_idx}"
    imgs_stats_path = Path(f"data/{file_name_base}{f_suffix}.json")

    if imgs_stats_path.exists() and not OVERRIDE_JSON:
        with open(imgs_stats_path, "r", encoding="utf-8") as f:
            dataset_stats = json.load(f)
        print_scores(dataset_stats)
    else:
        dataset_stats = compute_similarities()
        save_path = save_json_versioned(imgs_stats_path, file_version_idx, dataset_stats)
        print(f"Saved new data as: {save_path}")

    img_idx_1 = 0
    img_idx_2 = 1
    efnetv2_scores = get_scores_json(dataset_stats)
    mpl.rcParams['keymap.save'] = [] # set w,s keys as a custom shortcuts to change the second image
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, efnetv2_scores))
    plt.ioff()
    show(efnetv2_scores, (img_idx_1, img_idx_2))
    plt.show()
