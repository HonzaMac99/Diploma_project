import numpy as np
import matplotlib.pyplot as plt
import torch
import skimage
from PIL import Image, ImageOps
import piexif
import time
import os
import json
from pathlib import Path
from datetime import datetime, timezone
from brisque import BRISQUE

PHOTO_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/"
IMG_NUM_RES = 6    # orig_res = [3000 x 4000] --> [375 x 500] (4)
OVERRIDE_JSON = False
SAVE_SCORE_EXIF = True

SHOW_IMAGES = True
MAX_IMAGES = 3 # maximum number of images to process

dataset_path = os.path.join(PHOTO_PATH, "selected_r30")
images = sorted(f for f in os.listdir(dataset_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg")))

img_idx = 0
fig, ax = plt.subplots()

_brisque_obj = None


def get_brisque():
    global _brisque_obj
    if _brisque_obj is None:
        _brisque_obj = BRISQUE(url=False)
    return _brisque_obj


def brisque_eval(img):
    # plt.ioff()

    brisque_obj = get_brisque()
    img_h, img_w, _ = img.shape
    b_scores_resls = []
    times = []
    for i in range(IMG_NUM_RES):
        img_new_h = img_h // 2**i
        img_new_w = img_w // 2**i
        img_tfd = skimage.transform.resize_local_mean(torch.tensor(img) / 255.0,
                                                      output_shape=[img_new_h, img_new_w])
        img_tfd = np.asarray(img_tfd)

        start_t = time.time()
        b_scores_resls.append(brisque_obj.score(img_tfd))
        end_t = time.time()
        times.append(end_t-start_t)

        # plt.imshow(img_tfd)
        # plt.axis("off")
        # plt.show()
        # print(f"[{img_h / 2**i}, {img_w / 2**i}]: {end_t-start_t:.4f} s")

    return b_scores_resls, times


def compute_scores():
    global img_idx
    plt.ion()

    print("Computing Brisque scores:")
    b_scores = []
    b_scores_raw = []
    img_stats_list = []  # list to store all data

    # print("Computing scores:", end="")
    for img_name in images:
        img_path = os.path.join(dataset_path, img_name)

        # img1 = skimage.io.imread(img_path)
        img_raw = Image.open(img_path)
        img_rot = ImageOps.exif_transpose(img_raw)  # apply EXIF orientation

        img_raw = np.array(img_raw)
        img_rot = np.array(img_rot)

        b_scores_resl, b_times = brisque_eval(img_rot)
        b_scores.append(b_scores_resl)

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
            "img": img_path,
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
            show(b_scores, img_idx)

        img_stats_list.append(img_stats)
        img_idx = (img_idx + 1) % len(images)

        if type(MAX_IMAGES) is int and img_idx == MAX_IMAGES:
            break

    dataset_stats = {
        "description": "BRISQUE statistics of dataset images computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(img_stats_list),
        "num_resolutions": IMG_NUM_RES,
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


def save_exif_comment(image_path, quality_score):
    """Save your calculated quality score back to EXIF"""
    # Note: piexif is lightweight and more suitable directly for exif than pyexiv2
    exif_dict = piexif.load(image_path)

    # Store quality score in UserComment (or create custom tag)
    user_comment = f"Brisque score: {quality_score:.3f}".encode('utf-8')
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


def show(b_scores, img_idx, interactive=False):
    global ax
    ax.clear()
    img_path = os.path.join(dataset_path, images[img_idx])

    # img1 = skimage.io.imread(img_path)
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # apply EXIF orientation
    img = np.array(img)

    b_score = b_scores[img_idx][0] # pick the one from the highest resolution

    exif_text = "EXIF: " + load_exif_comment(img_path)

    ax.imshow(img)
    n_images = len(images) if type(MAX_IMAGES) is not int else MAX_IMAGES
    ax.set_title(f"Brisque score: {b_score:.2f}   [{img_idx+1}/{n_images}]")
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


def on_key(event, b_scores):
    global img_idx
    n_images = len(images) if type(MAX_IMAGES) is not int else MAX_IMAGES
    if event.key == "d":
        img_idx = (img_idx + 1) % n_images
    elif event.key == "a":
        img_idx = (img_idx - 1) % n_images
    elif event.key == "q":
        plt.close(fig)
        return
    show(b_scores, img_idx, interactive=True)


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
    file_version_idx = 1
    file_name_base = "brisque_stats"

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
    b_scores = [x["data"]["scores_rot"] for x in dataset_stats["statistics"]]
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, b_scores))
    plt.ioff()
    show(b_scores, img_idx)
    plt.show()
