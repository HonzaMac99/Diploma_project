import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from pathlib import Path
import json
from itertools import islice
import re
import csv
from collections import defaultdict
from scipy.io import loadmat
import shutil

from utils import ImageViewer, load_results_versioned
from nima_eval import compute_nima_scores
from clip_iqa_eval import compute_clip_scores

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"

MAX_IMAGES = None

SHOW_IMAGES = False
SAVE_IMAGES = False

COMPUTE_F1 = True
RECOMPUTE = True

# OVERRIDE = True

# -----------------
# download links:
# -----------------
# see README.md
# todo: try to get SPAQ from Baidu (google drive is unavailable) -> Baidu requires chinese phone number for registration

# ----------
# notes:
# ----------
# [aadb] consists of flickr images, each is evaluated by 5 different AMTurk workers with scores 1-5
# [flickr-aes, koniq10k] use 'zscore = (score - rater_mean) / rater_std' > this removes rater bias
# [flickr-aes] the testing set have separate AMT workers from train set to simulate real world scenario
# [real-cur] collection of 14 personal albums (each diff person) with their ratings on their own photos, INSPIRATION for ours
#            the images don't have rotation exif correction, doesn't really resemble MY ratings
# [para] enables to choose between aesthetic image score and global image score probably including other qualities
#   such as light, dof, color, composition
# [tid2013] is designed for techincal quality assessment
# [live-itw, para] best represents MY personal ratings
#                  "in the wild" means that it uses authentic, natural distortions rather than synthetic ones
# [tad66k] human perception: image --> theme --> aesthetics vs machine: image --> aesthetics

# line ~200: ':=' is a walrus - it assigns a value and checks at the same time
# for img_name, scores in sorted(image_scores.items(), key=lambda x: natural_key(x[0])):

dataset_img_dirs = {
    "aadb"          : "train/",
    "ava"           : "images/",
    "flickr-aes"    : "FLICKR-AES-001/40K/",
    "grenoble"      : "full/",
    "kaohsiung"     : "full/",
    "koniq10k"      : "1024x768/",
    "live-itw"      : "Images/",
    "namibie"       : "corrected/",
    "para"          : "imgs/",
    "real-cur"      : "",
    "tad66k"        : "TAD66K/",
    "tid2013"       : "distorted_images/",
}

dataset_data_files = {
    "aadb"          : "result_csv.csv",                                                             # scores 1-5 (5 AMT)
    "ava"           : "ground_truth_dataset.csv",                                                   # distr. 0-1 (for 0-10 ratings, NIMA style)
    "flickr-aes"    : "FLICKR-AES_image_score.txt", # orig fnames start with one whitespace!!       # scores 0-1 (5 AMT, zscores)
    # "flickr-aes"    : "FLICKR-AES-001/FLICKR-AES_image_labeled_by_each_worker.csv",
    "grenoble"      : "",                                                                           # -- selection --
    "kaohsiung"     : "",                                                                           # -- selection --
    "koniq10k"      : "koniq10k_scores_and_distributions/koniq10k_scores_and_distributions.csv",    # scores 0-100 (zscores, ??)
    "live-itw"      : "Data/",                                                                      # scores 0-100
    "namibie"       : "",                                                                           # -- selection --
    "para"          : "annotation/PARA-Images.csv",                                                 # scores 1-5 (~25 AMT)
    "real-cur"      : "",                                                                           # scores 1-5 (from the owners)
    "tad66k"        : "labels/merge/",                                                              # scores 0-10
    "tid2013"       : "mos.csv"                                                                     # scores 0-10 (max 7.2 though)
}


def get_visible_dir_list(in_dir):
    return [p for p in in_dir.iterdir() if p.is_dir() and not (p.name.startswith(".") or p.name.startswith("_"))]


def natural_key(path):
    name_s = path.name
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', name_s)]


def get_para_img_paths(dataset_img_path):
    img_paths = []
    img_limit_num = MAX_IMAGES
    for session_path in get_visible_dir_list(dataset_img_path):
        dir_it = islice(session_path.iterdir(), img_limit_num)
        session_img_paths = [
            img_file for img_file in dir_it
            if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS
        ]
        if isinstance(img_limit_num, int):
            img_limit_num -= len(session_img_paths)
        img_paths.extend(session_img_paths)

    img_paths = sorted(img_paths, key=natural_key)
    return img_paths


def get_real_cur_img_paths(dataset_img_path):
    img_paths = []
    img_limit_num = MAX_IMAGES
    for album_path in get_visible_dir_list(dataset_img_path):
        for score_dir_path in get_visible_dir_list(album_path):
            dir_it = islice(score_dir_path.iterdir(), img_limit_num)
            session_img_paths = [
                img_file for img_file in dir_it
                if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS
            ]
            if isinstance(img_limit_num, int):
                img_limit_num -= len(session_img_paths)
            img_paths.extend(session_img_paths)

    img_paths = sorted(img_paths, key=natural_key)
    return img_paths


def get_img_paths(dataset_path, dataset_name):
    # get the path to the image folder inside the dataset
    dataset_img_path = dataset_path / dataset_img_dirs[dataset_name]

    if dataset_name == "para":
        return get_para_img_paths(dataset_img_path)
    if dataset_name == "real-cur":
        return get_real_cur_img_paths(dataset_img_path)

    dir_it = dataset_img_path.iterdir()
    if isinstance(MAX_IMAGES, int):
        dir_it = islice(dir_it, MAX_IMAGES)

    img_paths = sorted(
        [img_file for img_file in dir_it
        if img_file.is_file() and img_file.suffix.lower() in IMG_EXTS],
        key=natural_key
    )
    return img_paths


# decorator wrapper for dedicated dataset score loading functions
SCORE_LOADERS = {}
def score_loader(name):

    def decorator(fn):
        SCORE_LOADERS[name] = fn
        return fn

    return decorator


@score_loader("aadb")
def get_aadb_scores(input_path, output_path=Path("aadb_iqa_scores.csv")):
    def parse(val):
        clean = re.sub(r"<[^>]+>", "", val).strip()
        try:
            return float({"Pos": "1", "Neg": "-1", "n": "0"}.get(clean, clean))
        except ValueError:
            return None

    avg_scores = defaultdict(float)

    with open(input_path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    header = next(csv.reader([lines[0]]))
    image_scores = defaultdict(list)

    for line in lines[1:]:
        if not line.strip():
            continue
        try:
            row = dict(zip(header, next(csv.reader([line]))))
        except Exception:
            continue
        for i in range(1, 11):
            url = row.get(f"Input.image_url{i}", "").strip()
            score = parse(row.get(f"Answer.overallScore{i}", ""))
            if url and score is not None:
                url_parts = url.split("/")
                url_farm_name = url_parts[2].split(".")[0]
                img_name = f"{url_farm_name}_{url_parts[-2]}_{url_parts[-1]}"

                image_scores[img_name].append(score)


    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "mos"])

        for img_name, scores in sorted(image_scores.items()):
            avg_scores[img_name] = round(sum(scores) / len(scores), 4) * 2
            writer.writerow([img_name, avg_scores[img_name]])

    print(f"Saved {len(image_scores)} image scores to '{output_path}'")
    return avg_scores

@score_loader("ava")
def get_ava_scores(input_path, output_path=Path("ava_iqa_scores.csv")):
    vote_cols = [f"vote_{i}" for i in range(1, 11)]

    with open(input_path, newline="", encoding="utf-8") as f:
        avg_scores = defaultdict(float)
        for row in csv.DictReader(f):
            img_name = f"{row["image_num"]}.jpg"
            avg_score = sum((i + 1) * float(row[v]) for i, v in enumerate(vote_cols))
            avg_scores[img_name] = round(avg_score, 4) * 10

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "mos"])
        for name, mos in sorted(avg_scores.items()):
            writer.writerow([name, mos])

    print(f"Saved {len(avg_scores)} image scores to '{output_path}'")
    return avg_scores

@score_loader("flickr-aes")
def get_flickr_aes_scores(input_path, output_path=Path("flickr_aes_iqa_scores.csv")):
    avg_scores = defaultdict(float)
    if input_path.name == "FLICKR-AES_image_score.txt":
        with open(input_path, encoding="utf-8") as f:
            avg_scores = {row[0]: float(row[1]) for line in f if (row := line.split())}

        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "mos"])
            for img_name, mos in sorted(avg_scores.items()):
                writer.writerow([img_name, mos])

    elif input_path.name == "FLICKR-AES_image_labeled_by_each_worker.csv":
        image_scores = defaultdict(list)

        with open(input_path, encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                if row := line.split():
                    image_scores[row[1]].append(float(row[2]))

        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_name", "mos"])
            for img_name, scores in sorted(image_scores.items()):
                avg_scores[img_name] = round(sum(scores) / len(scores), 4) * 2
                writer.writerow([img_name, avg_scores[img_name]])

    print(f"Saved {len(avg_scores)} image scores to '{output_path}'")
    return avg_scores

@score_loader("grenoble")
def get_grenoble_scores(input_path, output_path=Path("grenoble_iqa_scores.csv")):
    # Todo
    ...

@score_loader("kaohsiung")
def get_kaohsiung_scores(input_path, output_path=Path("kaohsiung_iqa_scores.csv")):
    # Todo
    ...

@score_loader("koniq10k")
def get_koniq10k_scores(input_path, output_path=Path("koniq10k_iqa_scores.csv")):
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        avg_scores = {
            row["image_name"]: float(row["MOS_zscore"])
            for row in reader
        }

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "mos"])
        for img_name, mos in sorted(avg_scores.items()):
            writer.writerow([img_name, mos])

    print(f"Saved {len(avg_scores)} image scores to '{output_path}'")
    return avg_scores

@score_loader("live-itw")
def get_live_itw_scores(input_path, output_path=Path("live_itw_iqa_scores.csv")):
    images = loadmat(input_path / "AllImages_release.mat")["AllImages_release"]
    mos    = loadmat(input_path / "AllMOS_release.mat")["AllMOS_release"]

    img_names = [row[0].item() for row in images]   # unwrap nested object arrays
    scores    = mos.flatten().tolist()               # flatten [[a, b, ...]] → [a, b, ...]

    avg_scores = {name: round(score, 4) for name, score in zip(img_names, scores)}

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "mos"])
        for img_name, mos in sorted(avg_scores.items()):
            writer.writerow([img_name, mos])

    print(f"Saved {len(avg_scores)} image scores to '{output_path}'")
    return avg_scores

@score_loader("namibie")
def get_namibie_scores(input_path, output_path=Path("namibie_iqa_scores.csv")):
    # Todo
    ...

@score_loader("para")
def get_para_scores(input_path, output_path=Path("para_iqa_scores.csv")):
    avg_scores = defaultdict(float)
    image_scores = defaultdict(list)

    with open(input_path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            if row := line.split(","):
                image_scores[row[1]].append(float(row[3]))

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "mos"])
        for img_name, scores in sorted(image_scores.items()):
            avg_scores[img_name] = round(sum(scores) / len(scores), 4)
            writer.writerow([img_name, avg_scores[img_name]])

    print(f"Saved {len(avg_scores)} image scores to '{output_path}'")
    return avg_scores

@score_loader("real-cur")
def get_real_cur_scores(input_path, output_path=Path("real_cur_iqa_scores.csv")):
    image_scores = defaultdict(float)

    for album_path in get_visible_dir_list(input_path):
        for score_dir_path in get_visible_dir_list(album_path):
            image_score = float(score_dir_path.name)
            for img_path in list(score_dir_path.iterdir()):
                image_scores[str(img_path.name)] = image_score

    return image_scores

@score_loader("tad66k")
def get_tad66k_scores(input_path, output_path=Path("tad66k_iqa_scores.csv")):
    train_data_path = input_path / "train.csv"
    test_data_path = input_path / "test.csv"

    with open(train_data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        avg_scores = {
            row["image"]: float(row["score"])
            for row in reader
        }

    with open(test_data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        new_scores = {
            row["image"]: float(row["score"])
            for row in reader
        }
        avg_scores |= new_scores  # avg_scores.update(new_scores)

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "mos"])
        for img_name, mos in sorted(avg_scores.items()):
            writer.writerow([img_name, mos])

    print(f"Saved {len(avg_scores)} image scores to '{output_path}'")
    return avg_scores

@score_loader("tid2013")
def get_tid2013_scores(input_path, output_path=Path("tid2013_iqa_scores.csv")):
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        avg_scores = {
            row["image_id"]: float(row["mean"])
            for row in reader
        }

    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "mos"])
        for img_name, mos in sorted(avg_scores.items()):
            writer.writerow([img_name, mos])

    print(f"Saved {len(avg_scores)} image scores to '{output_path}'")
    return avg_scores


def load_iqa_scores(path):
    print(f"Loading scores from: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return {row["image_name"]: float(row["mos"]) for row in csv.DictReader(f)}


def get_dataset_scores(dataset_path, dataset_name, img_paths, **kwargs):
    data_file_path = dataset_path / dataset_data_files[dataset_name]
    output_file_path = dataset_path / "my_calc/aes_scores.csv"
    # output_file_path = dataset_path / f"{dataset_name}_iqa_scores.csv"

    if not RECOMPUTE and output_file_path.is_file():
        img_scores = load_iqa_scores(output_file_path)
    else:
        if dataset_name not in SCORE_LOADERS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {list(SCORE_LOADERS)}")
        img_scores = SCORE_LOADERS[dataset_name](data_file_path, output_file_path, **kwargs)

    # for each img_path assign score from img_scores
    scores = []
    n_no_scores = 0
    for i, img_path in enumerate(img_paths):
        img_name = img_path.name
        if img_name not in img_scores:
            # print(f"{img_name} has no score data!")
            scores.append(-1.0)
            n_no_scores += 1
            continue
        img_score = img_scores[img_name]
        scores.append(img_score)

    print(f"{len(img_paths)} images, {len(img_paths)-n_no_scores} scores, {n_no_scores} with no score")
    return scores, img_paths


def calc_f1(predicted_ids, ground_truth_ids):
    predicted_set = set(predicted_ids)
    ground_truth_set = set(ground_truth_ids)

    tp = len(predicted_set & ground_truth_set)  # in both
    fp = len(predicted_set - ground_truth_set)  # predicted but not in GT
    fn = len(ground_truth_set - predicted_set)  # in GT but not predicted

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"f1": f1, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


if __name__ == "__main__":

    # dataset_names = ["aadb", "ava", "flickr-aes", "grenoble", "kaohsiung", "koniq10k", "live-itw", "namibie", "para", "real-cur", "tad66k", "tid2013"]
    dataset_names = ["aadb", "ava", "flickr-aes", "koniq10k", "live-itw", "para", "real-cur", "tad66k", "tid2013"]
    # dataset_names = ["real-cur", "para"]

    for dataset_name in dataset_names:
        print("----------------------")
        print(f"   {dataset_name}    ")
        print("----------------------")

        dataset_path = Path(DATASET_ROOT) / dataset_name
        img_paths = get_img_paths(dataset_path, dataset_name)
        scores, img_paths = get_dataset_scores(dataset_path, dataset_name, img_paths)

        if scores is None:
            scores = [0] * len(img_paths)

        # sort the images according to their iqa scores
        sorted_pairs = sorted(zip(img_paths, scores), key=lambda x: x[1], reverse=True)
        sorted_img_paths, scores = map(list, zip(*sorted_pairs))

        # get the last index of an image with a ground truth score (the rest have assigned values -1)
        last_score_idx = len(sorted_img_paths) - 1
        while scores[last_score_idx] == -1:
            last_score_idx -= 1

        # select 3*4=12 photos from each dataset representing good, medium and bad scores
        if SHOW_IMAGES:
            mid_score_idx = last_score_idx // 2

            show_img_paths = sorted_img_paths[:4] + sorted_img_paths[mid_score_idx-2 : mid_score_idx+2] + sorted_img_paths[last_score_idx-4:last_score_idx]
            show_scores = scores[:4] + scores[mid_score_idx-2 : mid_score_idx+2] + scores[last_score_idx-4:last_score_idx]

            assert len(show_img_paths) == len(show_scores), "Number of images is different than number of scores!"
            viewer = ImageViewer(show_img_paths, scores=show_scores, mode='single', tool_name=dataset_name)

            plt.ioff()
            viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
            viewer.show_current(interactive=False)

            dst_root = Path("/home/honzamac/Pictures/")
            lvls = ["top", "mid", "last"]
            for i, lvl in enumerate(lvls):
                for j in range(4):
                    idx = i*4 + j
                    img_path = show_img_paths[idx]
                    img_score = show_scores[idx]
                    print(f"{dataset_name}-{lvl}-{j+1}: {img_score} ({img_path.name})")

                    if SAVE_IMAGES:
                        if img_path.suffix == ".bmp":
                            save_name = f"{dataset_name}_{lvl}_{j + 1}.jpg"
                            dst_path = dst_root / save_name
                            img = Image.open(img_path).convert("RGB")  # ensures no alpha channel issues
                            img.save(dst_path, "JPEG", quality=90)  # quality: 1–95, default is 75
                            print(f"Saved {dst_path}")
                        else:
                            save_name = f"{dataset_name}_{lvl}_{j+1}{img_path.suffix}"
                            dst_path = dst_root / save_name
                            shutil.copy(img_path, dst_path)
                            print(f"Saved {dst_path}")

        # compare NIMA, CLIP and ... score based ranking with GT on 50%, 20%, 10%, 5%, 2%, 1% top selections - use F1 score
        if COMPUTE_F1:
            paths_cfg = {
                "dataset_root": DATASET_ROOT,
                "dataset_path": dataset_path,
                "results_root": RESULTS_ROOT
            }

            # we don't want to evaluate images with no ground truth scores
            sorted_img_paths = sorted_img_paths[:last_score_idx]
            scores = scores[:last_score_idx]

            nima_scores = compute_nima_scores(paths_cfg, sorted_img_paths, save_scores=True, load_scores=True)
            sorted_pairs = sorted(zip(sorted_img_paths, nima_scores), key=lambda x: x[1], reverse=True)
            nima_img_paths, scores = map(list, zip(*sorted_pairs))

            clip_scores = compute_clip_scores(paths_cfg, sorted_img_paths, save_scores=True, load_scores=True)
            sorted_pairs = sorted(zip(sorted_img_paths, clip_scores), key=lambda x: x[1], reverse=True)
            clip_iqa_img_paths, scores = map(list, zip(*sorted_pairs))

            top_splits = [2, 5, 10, 20, 50, 100]
            for top_split in top_splits:
                top_k = last_score_idx/top_split

                nima_f1 = calc_f1(nima_img_paths[:top_k], sorted_img_paths[:top_k])
                clip_iqa_f1 = calc_f1(clip_iqa_img_paths[:top_k], sorted_img_paths[:top_k])

                print(f"NIMA F1 for {100/top_split}% is {nima_f1}")
                print(f"CLIP-IQA F1 for {100/top_split}% is {clip_iqa_f1}")

        # assert len(sorted_img_paths) == len(scores), "Number of images is different than number of scores!"
        # viewer = ImageViewer(sorted_img_paths, scores=scores, mode='single', tool_name=dataset_name)

        # plt.ioff()
        # viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
        # viewer.show_current(interactive=False)



