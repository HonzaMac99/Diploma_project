import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image, ImageOps
import json
import hashlib
from pathlib import Path
from datetime import datetime
import sys
import piexif

RESULTS_ROOT = Path.home() / "Edu/m5/Projekt_D/projekt_testing/results"

# todo: refine this prototype

# region Saving and loading
def save_scores(
        dataset_path,
        dataset_root,
        results_root,
        tool_name,
        results,
        save_method="json",
        override=False
):
    """
    Prepare a directory for a dataset located under a common dataset root.
    Dir name corresponding to the dataset is derived from the relative path, ex.: coco/train2017/ -> coco_train2017
    """

    dataset_path = Path(dataset_path).expanduser().resolve()
    data_root = Path(dataset_root).expanduser().resolve()
    results_root = Path(results_root).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if not dataset_path.is_relative_to(data_root):
        raise ValueError(f"Dataset path {dataset_path} is not under data root {data_root}")

    # --- dataset name from relative path ---
    relative_parts = dataset_path.relative_to(data_root).parts
    dataset_name = "_".join(relative_parts[-3:]) # cap the name length just to 3 parts

    # # --- stable identity hash ---
    # dataset_hash = hashlib.sha1(
    #     str(dataset_path).encode("utf-8")
    # ).hexdigest()[:8]
    # results_dir = Path("data") / f"{dataset_name}__{dataset_hash}"

    results_dir = results_root / f"{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    if save_method == "json":
        with results_dir.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    elif save_method == "npz":
        scores = np.asarray(results, dtype=np.float32)
        assert results.dtype != object, "Results contain non-numeric objects"
        np.savez(f"{tool_name}_scores.npz", scores=scores)


def save_json_versioned(path: Path, idx, data: dict, override=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or override:
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

    with open(str(final_path), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return final_path


def load_scores(dataset_path, dataset_root, results_root, tool_name, load_method="json"):
    """
    Load scores for a dataset and tool. Dataset directory name is derived from the relative path to dataset_root,
    capped to last 3 components, plus a full-path hash.
    """
    dataset_path = Path(dataset_path).expanduser().resolve()
    data_root = Path(dataset_root).expanduser().resolve()
    results_root = Path(results_root).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if not dataset_path.is_relative_to(data_root):
        raise ValueError(f"Dataset path {dataset_path} is not under data root {data_root}")

    # --- dataset name from relative path ---
    relative_parts = dataset_path.relative_to(data_root).parts
    dataset_name = "_".join(relative_parts[-3:]) # cap the name length just to 3 parts

    results_dir = results_root / f"{dataset_name}"

    if load_method == "json":
        ...
    elif load_method == "npz":
        data = np.load(results_dir / f"{tool_name}_scores.npz")
        scores = data["scores"]
    else:
        print("[load_scores]: unknown method used for saving!")

    return scores


def prepare_dataset_cache(
    dataset_path: Path,
    data_root: Path,
    cache_root: Path = Path.home() / ".cache" / "nima",
    tool_name: str = "nima",
    tool_version: str | None = None,
) -> Path:
    """
    Prepare a cache directory for a dataset located under a common data root.

    Dataset name is derived from the relative path:
    e.g. coco/train2017 -> coco_train2017
    """

    dataset_path = Path(dataset_path).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    cache_root = Path(cache_root).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if not dataset_path.is_relative_to(data_root):
        raise ValueError(
            f"Dataset path {dataset_path} is not under data root {data_root}"
        )

    # --- dataset name from relative path ---
    relative_parts = dataset_path.relative_to(data_root).parts
    dataset_name = "_".join(relative_parts)

    # --- stable identity hash ---
    dataset_hash = hashlib.sha1(
        str(dataset_path).encode("utf-8")
    ).hexdigest()[:8]

    cache_dir = cache_root / f"{dataset_name}__{dataset_hash}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- metadata ---
    meta_path = cache_dir / "meta.json"

    if not meta_path.exists():
        meta = {
            "schema_version": 1,
            "dataset": {
                "name": dataset_name,
                "path": str(dataset_path),
                "relative_to": str(data_root),
                "hash": dataset_hash,
            },
            "run": {
                "tool": tool_name,
                "tool_version": tool_version,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "python": sys.version.split()[0],
            },
        }

        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return cache_dir


def save_exif_comment(image_path, quality_score):
    """Save your calculated quality score back to EXIF"""
    # Note: piexif is lightweight and more suitable directly for exif than pyexiv2
    exif_dict = piexif.load(str(image_path))

    # Store quality score in UserComment (or create custom tag)
    user_comment = f"Brisque score: {quality_score:.3f}".encode('utf-8')
    exif_dict['Exif'][piexif.ExifIFD.UserComment] = user_comment

    # problematic key Exif.Photo.SceneType = 41279 with int value
    exif_dict['Exif'].pop(41729, None)
    # print(exif_dict['Exif'])

    # Write back to image
    exif_bytes = piexif.dump(exif_dict)
    piexif.insert(exif_bytes, str(image_path))

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
        exif_dict = piexif.load(str(img_path))
        comment = exif_dict["Exif"].get(piexif.ExifIFD.UserComment, b"")
        return comment.decode("utf-8") if comment else "No EXIF data"
    except Exception:
        return "No EXIF data"
# endregion

# region Plotting

class ImageViewer:
    """
    Interactive image viewer for single and dual image evaluation
    Single image -> aesthetic and technical quality evaluation
    Dual image -> structural and content similarity evaluation

    Usage:
        viewer = ImageViewer(img_paths, scores, mode='single', tool_name="EfficientNetV2")
        fig = plt.figure()
        fig.canvas.mpl_connect('key_press_event', viewer.on_key)
        viewer.show_current(interactive=False)
    """

    def __init__(self, img_paths, scores, mode='single', tool_name=""):
        """
        Parameters
        ----------
        img_paths : list of Path or str
            List of image file paths
        scores : list or dict
            Scores for each image, or for each pair of images
        mode : str
            'single' or 'dual'
        tool_name : str
            Name of the scoring tool for titles
        """
        self.img_paths = img_paths
        self.scores = scores
        self.mode = mode
        self.tool_name = tool_name

        self.n_images = len(img_paths)
        self.idx1, self.idx2 = 0, 1  # dual-view indices

        # for evaluation with multiple tools at once
        self.multi_tools = isinstance(scores, dict)
        self.tool_names = list(self.scores.keys()) if self.multi_tools else [tool_name]
        self.custom_texts = [] # Keep track of texts with all scores in dual-view

        # Create figure and axes depending on mode
        if mode == 'single':
            self.fig, self.ax = plt.subplots()
            # self.fig, self.ax = plt.subplots(figsize=(6, 6))
        elif mode == 'dual':
            self.fig, self.axes = plt.subplots(1, 2)
            # self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        else:
            raise ValueError("mode must be 'single' or 'dual'")

        # disable 's' key shortcut to allow navigating the second image view with 'w' and 's'
        if mode == 'dual':
            mpl.rcParams['keymap.save'] = []

    def clear_texts(self):
        for txt in self.custom_texts:
            txt.remove()
        self.custom_texts = []

    # ----------------------------
    # Single-view rendering with scores
    # ----------------------------
    def show_single(self, interactive=False):
        img_path = self.img_paths[self.idx1]
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        img = np.array(img)

        if self.idx1 < len(self.scores):
            score = self.scores[self.idx1]
            score = f"{score:.2f}"
        else:
            score = "---"

        exif_text = "EXIF: " + load_exif_comment(img_path)

        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis("off")
        self.ax.set_title(f"{self.tool_name} score: {score}   [{self.idx1+1}/{self.n_images}]")

        # Text under the image
        self.ax.text(
            0.5, -0.02,
            exif_text,
            transform=self.ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            wrap=True
        )

        if interactive:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()
            plt.show()
            plt.pause(0.001)

    # ----------------------------
    # Dual-view rendering with scores of one or all methods
    # ----------------------------
    def show_dual(self, interactive=False):
        idx1, idx2 = self.idx1, self.idx2

        if self.multi_tools:
            self.clear_texts()

        for ax, img_idx in zip(self.axes, (idx1, idx2)):
            img_path = self.img_paths[img_idx]
            img = Image.open(img_path)
            img = ImageOps.exif_transpose(img)
            img = np.array(img)

            ax.clear()
            ax.imshow(img)
            ax.axis("off")

            if self.multi_tools:
                x_pos, y_pos = 0.05, 0.05
                score_text = (f"{self.tool_names[0]}: {self.scores[self.tool_names[0]][img_idx]:.2f},  "
                              f"{self.tool_names[1]}: {self.scores[self.tool_names[1]][img_idx]:.2f}")
                ax.text(x_pos, y_pos, score_text,
                        color='white', fontsize=8, fontweight='bold',
                        bbox=dict(facecolor='black', alpha=0.6, pad=3),
                        transform=ax.transAxes)
        if self.multi_tools:
            self.fig.subplots_adjust(top=0.8)  # leave space for text
            t1 = self.fig.text(
                0.5, 0.95,
                f"{self.tool_names[2]:<8}: {self.scores[self.tool_names[2]][idx1][idx2]:7.2f}",
                ha='center', fontsize=14, fontweight='bold', family="monospace"
            )
            t2 = self.fig.text(
                0.5, 0.90,
                f"{self.tool_names[3]:<8}: {self.scores[self.tool_names[3]][idx1][idx2]:7.2f}",
                ha='center', fontsize=14, fontweight='bold', family="monospace"
            )
            t3 = self.fig.text(
                0.5, 0.05,
                f"[{idx1 + 1}, {idx2 + 1} | {self.n_images}]",
                ha='center', fontsize=14, fontweight='bold'
            )
            self.custom_texts = [t1, t2, t3]
            self.fig.subplots_adjust(top=0.95, bottom=0.1)
            plt.tight_layout(pad=2.0)
        else:
            if idx1 < len(self.scores) and idx2 < len(self.scores):
                score = self.scores[idx1][idx2]
                score = f"{score:.2f}"
            else:
                score = "---"
            self.fig.suptitle(
                f"{self.tool_name} score: {score}   [{idx1+1}, {idx2+1} | {self.n_images}]",
                fontsize=14
            )

        if interactive:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()
            plt.show()
            plt.pause(0.001)

    # ----------------------------
    # Public method to show current images
    # ----------------------------
    def show_current(self, interactive=False):
        if self.mode == 'single':
            self.show_single(interactive=interactive)
        else:
            self.show_dual(interactive=interactive)

    # ----------------------------
    # Key callback handler
    # ----------------------------
    def on_key(self, event):
        if event.key == "q":
            plt.close(self.fig)
            return

        if event.key == "d":
            self.idx1 = (self.idx1 + 1) % self.n_images
        elif event.key == "a":
            self.idx1 = (self.idx1 - 1) % self.n_images
        elif self.mode == 'dual':
            if event.key == "w":
                self.idx2 = (self.idx2 + 1) % self.n_images
            elif event.key == "s":
                self.idx2 = (self.idx2 - 1) % self.n_images

        self.show_current(interactive=True)
# endregion