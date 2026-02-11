import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle
import math

from PIL import Image, ImageOps
import piexif

from pathlib import Path
import json
from datetime import datetime, timezone
import hashlib
# DO NOT DELETE ABOVE

# region Saving and loading

def save_results_versioned(paths_cfg, results, file_name_base, save_method="json", override_last=True):
    """
    Prepare a directory for a dataset located under a common dataset root.
    Dir name corresponding to the dataset is derived from the relative path, ex.: coco/train2017/ -> coco_train2017
    """
    dataset_root = Path(paths_cfg["dataset_root"]).expanduser().resolve()
    dataset_path = Path(paths_cfg["dataset_path"]).expanduser().resolve()
    results_root = Path(paths_cfg["results_root"]).expanduser().resolve()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if not dataset_path.is_relative_to(dataset_root):
        raise ValueError(f"Dataset path {dataset_path} is not under data root {dataset_root}")

    # --- dataset name from relative path ---
    relative_parts = dataset_path.relative_to(dataset_root).parts
    dataset_name = "_".join(relative_parts[-3:]) # cap the name length just to 3 parts (no error for short arrays)

    # # --- stable identity hash ---
    # dataset_hash = hashlib.sha1(
    #     str(dataset_path).encode("utf-8")
    # ).hexdigest()[:8]
    #
    # results_dir = Path("data") / f"{dataset_name}__{dataset_hash}"

    results_dir = results_root / f"{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{file_name_base}.{save_method}"

    # if no overriding, add a version suffix to the file: '_v2', '_v3' etc.
    ver_idx = 0
    new_path = results_path
    while new_path.exists():
        ver_idx += 1
        new_path = results_path.with_name(f"{file_name_base}_v{ver_idx}.{save_method}") # changing just the file name

    if override_last:
        if ver_idx <= 1:
            new_path = results_path
        else:
            new_path = results_path.with_name(f"{file_name_base}_v{ver_idx-1}.{save_method}") # changing just the file name

    assert not isinstance(results, dict) or save_method == 'json', "Dict formats are only for json!"
    assert not isinstance(results, list) or save_method == 'npz', "List formats are only for npz!"

    if save_method == "json":
        with open(str(new_path), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    elif save_method == "npz":
        scores = np.asarray(results, dtype=np.float32)
        assert scores.dtype != object, "Results contain non-numeric objects"
        np.savez(str(new_path), scores=scores)
    else:
        raise ValueError(f"Save_method {save_method} is not 'json' or 'npz'")

    return new_path


def load_results_versioned(paths_cfg, file_name_base, ver_idx=None, load_method="json"):
    """
    Load scores for a dataset and tool. Dataset directory name is derived from the relative path to dataset_root,
    capped to last 3 components, plus a full-path hash.
    """
    dataset_path = Path(paths_cfg["dataset_path"]).expanduser().resolve()
    dataset_root = Path(paths_cfg["dataset_root"]).expanduser().resolve()
    results_root = Path(paths_cfg["results_root"]).expanduser().resolve()

    # if not dataset_path.exists():
    #     raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if not dataset_path.is_relative_to(dataset_root):
        raise ValueError(f"Dataset path {dataset_path} is not under data root {dataset_root}")

    # --- dataset name from relative path ---
    relative_parts = dataset_path.relative_to(dataset_root).parts
    dataset_name = "_".join(relative_parts[-3:]) # cap the name length just to 3 parts

    # # --- stable identity hash ---
    # dataset_hash = hashlib.sha1(
    #     str(dataset_path).encode("utf-8")
    # ).hexdigest()[:8]
    #
    # results_dir = Path("data") / f"{dataset_name}__{dataset_hash}"

    results_dir = results_root / f"{dataset_name}"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{file_name_base}.{load_method}"

    if ver_idx is None: # just load the last version, if version not specified
        last_ver_idx = 0
        new_path = results_dir / f"{file_name_base}.{load_method}"
        while new_path.exists():
            results_path = new_path
            last_ver_idx += 1
            new_path = results_path.with_name(f"{file_name_base}_v{ver_idx}.{load_method}")
    elif ver_idx > 0: # version idx corresponds to: '_v2', '_v3' etc.
        results_path = results_dir / f"{file_name_base}_v{ver_idx}.{load_method}"

    if not results_path.exists():
        print(f"[Loading]: {results_path} not found")
        return None

    if load_method == "json":
        with open(results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    elif load_method == "npz":
        data = np.load(results_path)
        # convert to normal dict if you want
        results = {k: data[k] for k in data.files}
        scores = data["scores"]
        return scores
    else:
        raise ValueError(f"Unsupported file type: {load_method}")


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
    Single image -> aesthetic and technical quality
    Dual image -> structural and content similarity
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
        self.n_frames = self.n_images
        self.idx1, self.idx2 = 0, 1  # dual-view indices

        # for evaluation with multiple tools at once
        self.multi_tools = isinstance(scores, dict)
        self.tool_names = list(self.scores.keys()) if self.multi_tools else [tool_name]
        self.custom_texts = [] # Keep track of texts with all scores in dual-view
        self.selection = [0.0 for i in range(len(img_paths))]

        # Create figure and axes depending on mode
        if mode == 'single':
            self.fig, self.ax = plt.subplots()
            # self.fig, self.ax = plt.subplots(figsize=(6, 6))
        elif mode == 'dual':
            self.fig, self.axes = plt.subplots(1, 2)
            # self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        elif mode == 'select':
            self.n_frames = math.ceil(len(img_paths) / 16)
            self.fig, self.axes = plt.subplots(4, 4, figsize=(8, 8))
        else:
            raise ValueError("mode must be 'single', 'dual', 'select'")

        # disable 's' key shortcut to allow navigating the second image view with 'w' and 's'
        mpl.rcParams['keymap.save'] = []

    def clear_texts(self):
        for txt in self.custom_texts:
            txt.remove()
        self.custom_texts = []

    def update_selection(self, selection):
        self.selection = selection

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
        self.ax.set_title(f"{self.tool_name} score: {score}   [{self.idx1+1}/{self.n_frames}]")

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
            # create texts with both (sift, efnetv2) similarity scores + both image ids at the bottom
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
                f"[{idx1 + 1}, {idx2 + 1} | {self.n_frames}]",
                ha='center', fontsize=14, fontweight='bold'
            )
            self.custom_texts = [t1, t2, t3]

            self.fig.subplots_adjust(top=0.95, bottom=0.1) # leave space for text
            # plt.tight_layout(pad=2.0)
            plt.tight_layout(rect=(0.0, 0.1, 1.0, 0.95))
        else:
            if idx1 < len(self.scores) and idx2 < len(self.scores):
                score = self.scores[idx1][idx2]
                score = f"{score:.2f}"
            else:
                score = "---"
            self.fig.suptitle(
                f"{self.tool_name} score: {score}   [{idx1+1}, {idx2+1} | {self.n_frames}]",
                fontsize=14
            )

        if interactive:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.draw()
            plt.show()
            plt.pause(0.001)

    # ----------------------------
    # Selection rendering with colored frames depicting the selection
    # ----------------------------
    def show_selection(self, interactive=False):
        # start_idx = (self.idx1 // 16) + (self.idx1 % 16) # first img idx at the 4x4 frame
        start_idx = (self.idx1 * 16) % self.n_images


        self.clear_texts()

        frame_colors = ['limegreen' if self.selection[i] else 'none' for i in range(self.n_images)]
        img_idx = start_idx
        for ax in self.axes.flat:
            print(f"Plotting image {img_idx+1}")
            if img_idx < self.n_images:
                f_color = frame_colors[img_idx]
                img_path = self.img_paths[img_idx]

                with Image.open(img_path) as img:
                    img = ImageOps.exif_transpose(img)  # apply EXIF orientation
                    img = np.asarray(img)
            else:
                # remove everything from the ax frame and make it invisible
                ax.clear()
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_edgecolor("white")
                continue
                # f_color = "none"
                # img = np.zeros((64, 64, 3)) # white image (float 1.0 represents is white)

            img_idx += 1

            ax.clear()
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])

            h, w = img.shape[:2]
            rect = Rectangle(
                (0, 0), w, h,
                linewidth=6,
                edgecolor=f_color,
                facecolor='none'
            )
            ax.add_patch(rect)

        t1 = self.fig.text(0.5, 0.95, f"Images {start_idx+1} - {min(start_idx+16, self.n_images)}",
                           ha='center', fontsize=14, fontweight='bold', family="monospace"
                           )
        t2 = self.fig.text(0.5, 0.05, f"[{self.idx1+1}/{self.n_frames}]",
                           ha='center', fontsize=14, fontweight='bold'
                           )
        self.custom_texts = [t1, t2]
        # self.fig.subplots_adjust(top=0.8, bottom=0.1) # leave space for text
        plt.tight_layout(rect=(0.0, 0.1, 1.0, 0.95))

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
        elif self.mode == 'dual':
            self.show_dual(interactive=interactive)
        elif self.mode == 'select':
            self.show_selection(interactive=interactive)
        else:
            raise ValueError("mode must be 'single', 'dual', 'select'")

    # ----------------------------
    # Key callback handler
    # ----------------------------
    def on_key(self, event):
        if event.key == "q":
            plt.close(self.fig)
            return

        if event.key == "d":
            self.idx1 = (self.idx1 + 1) % self.n_frames
        elif event.key == "a":
            self.idx1 = (self.idx1 - 1) % self.n_frames
        elif self.mode == 'dual':
            if event.key == "w":
                self.idx2 = (self.idx2 + 1) % self.n_frames
            elif event.key == "s":
                self.idx2 = (self.idx2 - 1) % self.n_frames

        self.show_current(interactive=True)
# endregion