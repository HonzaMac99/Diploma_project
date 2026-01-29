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

PHOTO_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/"
IMG_NUM_RES = 1    # orig_res = [3000 x 4000] --> [224, 244] (fixed nima input size)
N_NEIGHBORS = 20
OVERRIDE_JSON = True
SAVE_SCORE_EXIF = False

SHOW_IMAGES = True
MAX_IMAGES = None # maximum number of images to process (for debugging)

dataset_path = os.path.join(PHOTO_PATH, "selected_r30")
images = sorted(f for f in os.listdir(dataset_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg")))

img_idx_1 = 0
img_idx_2 = 1
fig, axes = plt.subplots(1, 2)
matches = []

sift = cv2.SIFT_create(nfeatures=1000)  # SIFT algorithm with number of keypoints
bf = cv2.BFMatcher()  # keypoint matcher

########################################################################################################################

def compute_SIFT(image):
    return sift.detectAndCompute(image, None)


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


def compute_score(matches, keypoint1, keypoint2):
    score = 1000 * (matches / min(keypoint1, keypoint2))
    if score > 100:
        score = 100
    return score


def compute_similarities():
    global img_idx, matches
    plt.ion()

    print("Computing SIFT similarity scores:")
    n_images = len(images)
    n_neighbors = N_NEIGHBORS

    features = {}  # keypoints and descriptors
    sift_scores = np.ones((n_images, n_images)) * (-1)
    img_stats_list = []  # list to store results

    # todo: try different resolutions
    # todo: compute avg times
    for i, img_name in enumerate(images):
        img_path = os.path.join(dataset_path, img_name)
        img = image_resize(cv2.imread(img_path), 1024)
        keypoints, descriptors = compute_SIFT(img)
        features[i] = (keypoints, descriptors)

    for i, img_name in enumerate(images):
        img_1_path = os.path.join(dataset_path, img_name)
        keypoints_i, descriptors_i = features[i]

        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                if j+1 == MAX_IMAGES:
                    break
                else:
                    continue
            img_2_path = os.path.join(dataset_path, images[j])
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
                "img_1": img_1_path,
                "img_2": img_2_path,
                "sift_similarity_score": sift_score  # todo: list for more resolutions?
                # todo: times?
            }
            
            # todo: exif?

            if SHOW_IMAGES:
                show(sift_scores, (i, j))

            img_stats_list.append(img_stats)

            if type(MAX_IMAGES) is int and j+1 == MAX_IMAGES:
                break
        if type(MAX_IMAGES) is int and i+1 == MAX_IMAGES:
            break

    imgs_stats = {
        "description": "SIFT statistics of image pairs from dataset computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(images) if type(MAX_IMAGES) is not int else MAX_IMAGES,
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


def show(scores, img_idxs, interactive=False):
    global fig, axes

    assert len(img_idxs) == 2, f"Size of img idxs has to be 2: {img_idxs}"

    for img_idx, ax in zip(img_idxs, axes):
        img_path = os.path.join(dataset_path, images[img_idx])

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # apply EXIF orientation
        img = np.array(img)

        ax.clear()
        ax.imshow(img)
        ax.axis("off")

    score = scores[img_idxs[0]][img_idxs[1]]

    n_images = len(images) if type(MAX_IMAGES) is not int else MAX_IMAGES
    fig.suptitle(
        f"SIFT score: {score:.2f}   [{img_idxs[0]+1}, {img_idxs[1]+1} | {n_images}]",
        fontsize=14
    )
    if interactive:
        fig.canvas.draw_idle()
    else:
        fig.canvas.draw()
        plt.show()
        plt.pause(0.001)


def on_key(event, sift_scores):
    global img_idx_1, img_idx_2
    n_images = len(images) if type(MAX_IMAGES) is not int else MAX_IMAGES
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
    show(sift_scores, img_idxs, interactive=True)


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

    dataset_stats = {}
    file_version_idx = 0
    file_name_base = "sift_stats"

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
    sift_scores = get_scores_json(dataset_stats)
    mpl.rcParams['keymap.save'] = [] # set w,s keys as a custom shortcuts to change the second image
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, sift_scores))
    plt.ioff()
    show(sift_scores, (img_idx_1, img_idx_2))
    plt.show()
