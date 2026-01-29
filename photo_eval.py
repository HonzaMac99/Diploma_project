import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps
from pathlib import Path
import os
import json
import piexif

from brisque_eval import compute_brisque_scores
from nima_eval import compute_nima_scores

# Todo: import sift_eval
# Todo: import clip_eval

DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/"
IMAGE_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MAX_IMAGES = 3
IMG_NUM_RES = 6    # orig_res = [3000 x 4000] --> [375 x 500] (4)
SAVE_SCORE_EXIF = False

def save_json_versioned(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        final_path = path
    else:
        i = 1
        while True:
            final_path = path.with_name(f"{path.stem}_{i}{path.suffix}")
            if not final_path.exists():
                break
            i += 1

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return final_path


def save_exif_comment(img_path, score):
    # Convert to a readable string
    print("Score: ", score)
    comment = "Brisque score: {:.3f}".format(score)

    # Load image and existing EXIF (if any)
    img = Image.open(img_path)
    exif_dict = piexif.load(img.info.get("exif", b""))

    # Write into UserComment
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = comment.encode("utf-8")

    # Save back
    exif_bytes = piexif.dump(exif_dict)

    piexif.insert(exif_bytes, img_path)
    # img.save(img_path, exif=exif_bytes) # !!! changes the image information due to a new jpg compression

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
        img_pil = Image.open(img_path)
        exif_dict = piexif.load(img_pil.info.get("exif", b""))
        comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment, b"")
        return comment.decode("utf-8") if comment else "No EXIF data"
    except Exception:
        return "No EXIF data"


def show_img_scores(scores, img_idxs, interactive=False):
    global fig, axes
    global img_files

    assert len(img_idxs) == 2, f"Size of img idxs has to be 2: {img_idxs}"
    idx1, idx2 = img_idxs
    keys_list = list(scores.keys())
    n_images = len(img_files)

    # Remove old texts from figure (pair scores, info)
    if not hasattr(fig, 'custom_texts'):
        fig.custom_texts = []
    for txt in fig.custom_texts:
        txt.remove()
    fig.custom_texts = []

    for img_idx, ax in zip(img_idxs, axes):
        img_path = os.path.join(dataset_path, img_files[img_idx])

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # apply EXIF orientation
        img = np.array(img)

        ax.clear()
        ax.imshow(img)
        ax.axis("off")

        x_pos, y_pos = 0.05, 0.05
        score_text = (f"{keys_list[0]}: {scores[keys_list[0]][img_idx]:.2f},  "
                      f"{keys_list[1]}: {scores[keys_list[1]][img_idx]:.2f}")

        ax.text(x_pos, y_pos, score_text,
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.6, pad=3),
                transform=ax.transAxes)

    fig.subplots_adjust(top=0.8)  # leave space for text
    t1 = fig.text(0.5, 0.95, f"{keys_list[2]}: {scores[keys_list[2]][idx1][idx2]}", ha='center', fontsize=14, fontweight='bold')
    t2 = fig.text(0.5, 0.90, f"{keys_list[3]}: {scores[keys_list[3]][idx1][idx2]}", ha='center', fontsize=14, fontweight='bold')
    t3 = fig.text(0.5, 0.05, f"[{idx1+1}, {idx2+1} | {n_images}]", ha='center', fontsize=14, fontweight='bold')
    fig.custom_texts = [t1, t2, t3]

    fig.subplots_adjust(top=0.95, bottom=0.1)
    plt.tight_layout(pad=2.0)

    if interactive:
        fig.canvas.draw_idle()
    else:
        fig.canvas.draw()
        plt.show()
        plt.pause(0.001)


def on_key(event, scores):
    global img_files
    global img_idx_1, img_idx_2, n_images

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
    show_img_scores(scores, img_idxs, interactive=True)


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

    img_idx_1 = 0
    img_idx_2 = 0
    fig, axes = plt.subplots(1, 2)

    # todo: brisque, nima - loading from files
    # todo: sift, efnetv2
    scores = {
        "brisque":  compute_brisque_scores(dataset_path, img_files),
        "nima":     compute_nima_scores(dataset_path, img_files),
        "sift":     np.random.normal(loc=50.0, scale=10.0, size=(n_images, n_images)).tolist(),
        "efnetv2":  np.random.normal(loc=50.0, scale=10.0, size=(n_images, n_images)).tolist()
    }
    # np.random.normal(loc=50.0, scale=10.0, size=(n_images)).tolist()

    mpl.rcParams['keymap.save'] = [] # set w,s keys as a custom shortcuts to change the second image
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, scores))
    show_img_scores(scores, (img_idx_1, img_idx_2))
    plt.show()
