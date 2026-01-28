import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from pathlib import Path
import os
import json
import piexif

# Todo: import brisque_eval (or brisque_utils?)
# Todo: import nima_eval
# Todo: import sift_eval
# Todo: import clip_eval

PHOTOS_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/"
IMG_NUM_RES = 6    # orig_res = [3000 x 4000] --> [375 x 500] (4)
SAVE_SCORE_EXIF = False


dataset_path = os.path.join(PHOTOS_PATH, "selected_r30")
images = sorted(f for f in os.listdir(dataset_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg")))

img_idx = 0
fig, ax = plt.subplots()
obj = None


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


def show(b_scores_resols, interactive=False):
    global ax
    ax.clear()
    img_path = os.path.join(dataset_path, images[img_idx])

    # img1 = skimage.io.imread(img_path)
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # apply EXIF orientation
    img = np.array(img)

    b_score = b_scores_resols[0]

    if SAVE_SCORE_EXIF:
        save_exif_comment(img_path, b_score)
    exif_text = "EXIF: " + load_exif_comment(img_path)

    ax.imshow(img)
    ax.set_title(f"Brisque score [{img_idx+1}/{len(b_scores)}]: {b_score:.2f}")
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
    if event.key == "d":
        img_idx = (img_idx + 1) % len(b_scores)
    elif event.key == "a":
        img_idx = (img_idx - 1) % len(b_scores)
    elif event.key == "q":
        plt.close(fig)
        return
    show(b_scores[img_idx], interactive=True)


if __name__ == "__main__":

    file_idx = 0
    file_name = "brisque_stats"

    f_suffix = "" if file_idx == 0 else f"_{file_idx}"
    imgs_stats_path = Path(f"data/{file_name}{f_suffix}.json")
    imgs_stats = {}
    if imgs_stats_path.exists():
        with open(imgs_stats_path, "r", encoding="utf-8") as f:
            imgs_stats = json.load(f)
        print_scores(imgs_stats)
    else:
        imgs_stats = compute_scores()

    img_idx = 0
    b_scores = [x["data"]["scores_rot"] for x in imgs_stats["statistics"]]
    fig.canvas.mpl_connect("key_press_event", lambda event: on_key(event, b_scores))
    show(b_scores[img_idx])
    plt.show()
