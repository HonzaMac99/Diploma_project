import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
import cv2

import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import json

from utils import save_results_versioned, load_results_versioned, remove_all_files_by_name, img_resize

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/full/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/grenoble/full/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/namibie/namibie_corrected/"

RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MAX_IMAGES = None

DISPLAY_GRID_HEIGHT = 3
DISPLAY_GRID_WIDTH = 5

CLUSTER_DIFF_THR = 10.0    # [s]
CLUSTER_MAX_MULT = 2  # Include a new photo in the cluster if its time difference
                       # is max x times bigger than the biggest in the cluster
NEIGHBORS_RANGE = 15  # range of the scope for similar photos search, ex. range = 10 -> 19 neighbors



class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.members = {i: {i} for i in range(n)}  # cluster members

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra

        self.parent[rb] = ra
        self.members[ra].update(self.members[rb])
        del self.members[rb]

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def get_cluster(self, x):
        root = self.find(x)
        return self.members[root]

# region cluster_editor

user_text = ""
plot_texts = []

def on_key(event):
    global user_text

    if event.key == "enter":
        plt.close()
    elif event.key == "backspace":
        user_text = user_text[:-1]
    else:
        user_text += event.key
    print("\rCurrent input:", user_text, end="", flush=True)



def parse_input(user_input):
    user_input = user_input.strip()
    if user_input == "":
        return "keep", None
    elif user_input in ["b", "back"]:
        return "back", None

    # find all bracket groups
    groups = re.findall(r"\[([^\]]+)\]", user_input)

    parsed = []
    for g in groups:
        nums = [int(x.strip()) for x in g.split(",")]
        parsed.append(nums)

    if len(parsed) == 1:
        return "select", parsed[0]
    return "split", parsed


def get_time(img_path):
    img = Image.open(img_path)
    exif_data = img.getexif()

    # time_str = exif_data.get(36867)     # DateTimeOriginal - mostly not present so switching to 306
    time_str = exif_data.get(306)       # DateTime
    subsec = exif_data.get(37521)       # SubSecTimeOriginal

    if time_str is None:
        return None

    time_data = datetime.strptime(time_str, "%Y:%m:%d %H:%M:%S")
    if subsec:
        microseconds = int(subsec.ljust(6, "0"))
        time_data = time_data.replace(microsecond=microseconds)
    # print("Photo taken at:", time_data)
    return time_data


def show_cluster(cluster, clusters, img_paths):
    global user_text, plot_texts
    img_grid_h = DISPLAY_GRID_HEIGHT
    img_grid_w = DISPLAY_GRID_WIDTH
    frames_thickness = 8

    n_images = len(img_paths)
    img_grid_n = img_grid_h * img_grid_w
    fig, axes = plt.subplots(img_grid_h, img_grid_w, figsize=(img_grid_w * 3, img_grid_h * 2.6))

    if len(cluster) > img_grid_n:
        print("W: Larger cluster than display!")

    offset = img_grid_w
    start_idx = min(cluster) - offset
    start_idx = max(min(start_idx, n_images-1-img_grid_n), 0)

    for txt in plot_texts:
        txt.remove()

    # creating an array JUST for the current plot scope with frame colors
    frame_colors = ['limegreen' if i in cluster else 'none' for i in range(start_idx, start_idx + img_grid_n)]
    frame_colors[min(cluster)-start_idx] = 'lime'

    # display other clusters in red
    for check_cluster in clusters:
        if max(check_cluster) < start_idx or min(check_cluster) > start_idx + img_grid_n:
            continue
        else:
            frame_colors_new = ['black' if start_idx + i in check_cluster else frame_colors[i] for i in range(img_grid_n)]
            frame_colors = frame_colors_new

    frame_idx = 0
    for ax in tqdm(axes.flat, desc="Plotting images", unit="plot"):
        img_idx = start_idx + frame_idx
        if img_idx < n_images:
            f_color = frame_colors[frame_idx]
            img_path = img_paths[img_idx]

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

        frame_idx += 1

        ax.clear()
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])

        h, w = img.shape[:2]
        rect = Rectangle(
            (0, 0), w, h,
            linewidth=frames_thickness,
            edgecolor=f_color,
            facecolor='none'
        )
        ax.add_patch(rect)

    t1 = fig.text(0.5, 0.95, f"Images {start_idx} - {min(start_idx + img_grid_n, n_images)}",
                  ha='center', fontsize=14, fontweight='bold', family="monospace"
                  )
    t2 = fig.text(0.5, 0.06, f"New cluster: {cluster}",
                       ha='center', fontsize=14, fontweight='bold'
                       )
    t3 = fig.text(0.5, 0.03, "(Type in the window to edit)",
                  ha='center', fontsize=14, fontweight='bold'
                  )
    plot_texts = [t1, t2, t3]
    # self.fig.subplots_adjust(top=0.8, bottom=0.1) # leave space for text

    plt.tight_layout(rect=(0.0, 0.1, 1.0, 0.95))

    user_text = ""
    # fig.canvas.draw()
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    # plt.pause(0.001)
    print("")

    return user_text


# manually edit saved clusters - intended for clusters by time
def clusters_editor(new_cluster, clusters, image_paths):
    user_input = show_cluster(new_cluster, clusters, image_paths)
    start_idx = min(new_cluster)

    try:
        action, data = parse_input(user_input)
    except:
        print(f"WRONG INPUT FORMAT '{user_input}', try again")
        clusters_editor(new_cluster, clusters, image_paths)
        return

    if action == "keep":
        clusters.append(new_cluster)
        print(f"Keeping: {new_cluster}")
    elif action == "back":
        last_cluster = clusters.pop()
        new_cluster = last_cluster + new_cluster
        clusters_editor(new_cluster, clusters, image_paths)
    elif action == "select":
        new_cluster = [start_idx + i for i in data]
        clusters.append(new_cluster)
        print(f"Keeping: {new_cluster}")
    elif action == "split":
        new_clusters = []
        for group in data:
            new_clusters.append([start_idx + i  for i in group])
        clusters.extend(new_clusters)
        print(f"Keeping: {new_clusters}")


def is_new_cl(new_cluster, clusters):
    for cluster in clusters:
        for img_id in cluster:
            if img_id in new_cluster:
                return False
    return True


# todo: make it universal for any clusters?
# create clusters manually aided by the time-wise cluster suggestions
def create_man_clusters(img_paths, thr=10.0, max_mult=2.0):
    print("------------------------------------------------")
    print("|         Welcome to Clusters editor!!         |")
    print("------------------------------------------------")
    print("| Usage (type into the plot window):           |")
    print("|   Enter       = keep cluster                 |")
    print("|   'b', 'back' = edit with previous cluster   |")
    print("|   'x', '[]'   = don't keep anything          |")
    print("|   [-1,0,2,3]  = edit selection               |")
    print("|   [0,1],[2,4] = split cluster                |")
    print("| >>> First image in cluster has idx 0! <<<    |")
    print("------------------------------------------------")

    photo_times = []
    for img_path in img_paths:
        time_data = get_time(img_path)
        photo_times.append((img_path, time_data))

    original = photo_times.copy()
    photo_times.sort(key=lambda x: x[1])
    if photo_times != original:
        print("W: Photo times do not match the file names!")

    # get sorted img paths as well
    img_paths = [x[0] for x in photo_times]

    clusters = []
    last_time = photo_times[0][1]
    new_cluster = [0]
    max_cluster_diff = thr
    for i, (img_path, time) in enumerate(photo_times[1:], start=1):
        diff = (time - last_time).total_seconds()
        # print(f"img{i-1} - img{i} diff: {diff}")

        outlier_diff_thr = max_cluster_diff * max_mult
        if diff < outlier_diff_thr:
            new_cluster.append(i)
            if diff > max_cluster_diff:
                max_cluster_diff = diff
        elif diff > outlier_diff_thr:
            if len(new_cluster) > 1 and is_new_cl(new_cluster, clusters):
                print(f"New_cluster: {new_cluster}")
                clusters_editor(new_cluster, clusters, img_paths)
            new_cluster = [i]
            max_cluster_diff = thr

        last_time = time
    return clusters
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
    n_images = len(img_paths)

    paths_cfg = {
        "dataset_root": DATASET_ROOT,
        "dataset_path": DATASET_PATH,
        "results_root": RESULTS_ROOT
    }

    img_clusters = create_man_clusters(img_paths, thr=CLUSTER_DIFF_THR, max_mult=CLUSTER_MAX_MULT)
    data = {
        "clusters":     [str(cluster) for cluster in img_clusters],  # for better json formatting
        "image_refs":   [f"{img_path.name}, {i}" for i, img_path in enumerate(img_paths)]
    }
    save_results_versioned(paths_cfg, data, "clusters_manual_v2", save_method="json")


    # # additional cluster checking
    # cluster_to_see = [2688]
    # show_cluster(cluster_to_see, img_paths)


    # # remove all files with clusters
    # remove_all_files_by_name(paths_cfg["results_root"], "clusters_manual")
