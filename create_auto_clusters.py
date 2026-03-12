import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2
import torch, clip

import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import json

from utils import save_results_versioned, load_results_versioned, remove_all_files_by_name, img_resize
from sift_eval import compute_sift_similarities
from clip_eval import compute_clip_similarities

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/full/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/grenoble/full/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/namibie/namibie_corrected/"

RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MAX_IMAGES = None
N_SIFT_FEATS = 1000
SIFT_RES = 1024

CLUSTER_DIFF_THR = 10.0    # [s]
CLUSTER_MAX_MULT = 2  # Include a new photo in the cluster if its time difference
                       # is max x times bigger than the biggest in the cluster
NEIGHBORS_RANGE = 15  # range of the scope for similar photos search, ex. range = 10 -> 19 neighbors
THR_HIST = 0.75
THR_SIFT = 0.7
THR_CLIP = 0.85

SHOW_CLUSTERS = False

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

def on_key(event):
    global user_text

    if event.key == "enter":
        plt.close()
    elif event.key == "backspace":
        user_text = user_text[:-1]
    else:
        user_text += event.key
    print("\rCurrent input:", user_text, end="", flush=True)


def show_cluster(cluster, image_paths):
    global user_text
    n = len(cluster)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1:
        axes = [axes]

    for i, (ax, img_idx) in enumerate(zip(axes, cluster)):
        img = Image.open(image_paths[img_idx])
        img = ImageOps.exif_transpose(img)
        ax.imshow(img)
        ax.set_title(f"{i}: image{img_idx}")

        # todo: nazev souboru pod fotku
        # filename = image_paths[img_idx].name
        # ax.set_xlabel(filename, fontsize=8)

        ax.axis("off")

    plt.tight_layout()
    user_text = ""
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    print("")
    return user_text


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


# manually edit saved clusters - intended for clusters by time
def clusters_editor(new_cluster, clusters, image_paths):
    user_input = show_cluster(new_cluster, image_paths)

    action, data = parse_input(user_input)
    if action == "keep":
        clusters.append(new_cluster)
        print(f"Keeping: {new_cluster}")
    elif action == "back":
        last_cluster = clusters.pop()
        new_cluster = last_cluster + new_cluster
        clusters_editor(new_cluster, clusters, image_paths)
    elif action == "select":
        new_cluster = [new_cluster[i] for i in data]
        clusters.append(new_cluster)
        print(f"Keeping: {new_cluster}")
    elif action == "split":
        new_clusters = []
        for group in data:
            new_clusters.append([new_cluster[i] for i in group])
        clusters.extend(new_clusters)
        print(f"Keeping: {new_clusters}")

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
    print("|   [0,2,3]     = select subset                |")
    print("|   [0,1],[2,4] = split cluster                |")
    print("| >> Clusters are always indexed from 0! <<    |")
    print("------------------------------------------------")

    photo_times = []
    for img_path in img_paths:
        time_data = get_time(img_path)
        photo_times.append((img_path, time_data))
    # print([x[1] for x in photo_times])
    photo_times.sort(key=lambda x: x[1])
    # print([x[1] for x in photo_times])

    # get sorted img paths as well
    img_paths = [x[0] for x in photo_times]

    clusters = []
    last_time = photo_times[0][1]
    new_cluster = [0]
    max_cluster_diff = thr
    for i, (img_path, time) in enumerate(photo_times[1:], start=1):
        diff = (time - last_time).total_seconds()
        print(f"img{i-1} - img{i} diff: {diff}")

        outlier_diff_thr = max_cluster_diff * max_mult
        if diff < outlier_diff_thr:
            new_cluster.append(i)
            if diff > max_cluster_diff:
                max_cluster_diff = diff
        elif diff > outlier_diff_thr:
            if len(new_cluster) > 1:
                print(new_cluster)
                clusters_editor(new_cluster, clusters, img_paths)
            new_cluster = [i]
            max_cluster_diff = thr

        last_time = time
    return clusters
# endregion

# region time clusters

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


def create_time_clusters(img_paths, thr=10.0, max_mult=2.0):
    photo_times = []
    for img_path in img_paths:
        time_data = get_time(img_path)
        photo_times.append((img_path, time_data))

    original = photo_times.copy()
    photo_times.sort(key=lambda x: x[1])
    if photo_times != original:
        print("W: Photo times do not match the file names!")

    # get sorted img paths as well - for img displaying
    # img_paths = [x[0] for x in photo_times]

    clusters_list = []
    last_time = photo_times[0][1]
    new_cluster = [0]
    max_cluster_diff = thr
    for i, (img_path, time) in enumerate(photo_times[1:], start=1):
        diff = (time - last_time).total_seconds()
        print(f"img{i-1} - img{i} diff: {diff}")

        outlier_diff_thr = max_cluster_diff * max_mult
        if diff < outlier_diff_thr:
            new_cluster.append(i)
            if diff > max_cluster_diff:
                max_cluster_diff = diff
        elif diff > outlier_diff_thr:
            if len(new_cluster) > 1:
                print(new_cluster)
                clusters_list.append(new_cluster)
            new_cluster = [i]
            max_cluster_diff = thr

        last_time = time

    if SHOW_CLUSTERS:
        for cluster in clusters_list:
            print(cluster)
            show_cluster(cluster, img_paths)

    return clusters_list
# endregion

# region histogram clusters

def calc_color_hist(img_path, method='hsv'):
    img = cv2.imread(img_path)
    if method == "hsv":    # calc. 2D histogram from [hue, saturation] with 50x60 bins and [0, 180, 0, 256] ranges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    elif method == "rgb":  # calc. 3D histotram from rgb with 4x4x4 bins - based on Automatic Summarization paper
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hist = cv2.calcHist([rgb], [0, 1, 2], None, [4, 4, 4], [0, 256, 0, 256, 0, 256])
    else:
        raise ValueError("Only 'hsv' and 'rgb' methods implemented.")
    cv2.normalize(hist, hist)
    return hist


def calc_edge_hist(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # calculate the hist using canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    hist = cv2.calcHist([edges], [0], None, [2], [0, 256])

    cv2.normalize(hist, hist)
    return hist


def calc_color_hist_simil(h1, h2):
    # # EMD - lower better, reshape into signature format
    # sig1 = np.array([[v, i] for i, v in enumerate(h1)], dtype=np.float32)
    # sig2 = np.array([[v, i] for i, v in enumerate(h2)], dtype=np.float32)
    # distance, _, _ = cv2.EMD(sig1, sig2, cv2.DIST_L2)

    # methods: cv2.HISTCMP_INTERSECT, cv2.HISTCMP_CORREL |inverse:| cv2.HISTCMP_CHISQR, cv2.HISTCMP_BHATTACHARYYA
    hist_simil = 1 - cv2.compareHist(h1, h2, cv2.HISTCMP_BHATTACHARYYA)
    return hist_simil


def calc_edge_simil(h1, h2):
    edge_simil = np.dot(h1.flatten(), h2.flatten())
    return edge_simil


def is_cluster_similar(new_id, cluster, similarities, thr):
    for img_id in cluster:
        hist_simil = max(similarities[img_id, new_id], similarities[new_id, img_id])
        if hist_simil > thr:
            return False
    return True


def create_hist_clusters(img_paths, thr=0.8):
    n_images = len(img_paths)
    nb_range = NEIGHBORS_RANGE
    color_hists = []
    edge_hists = []
    for img_path in tqdm(img_paths, desc="imgs hists", unit="img"):
        color_hists.append(calc_color_hist(img_path, method='hsv'))
        edge_hists.append(calc_edge_hist(img_path))

    uf = UnionFind(n_images)
    simil_mtx = np.zeros((n_images, n_images))
    np.fill_diagonal(simil_mtx, 1.0)
    for i, _ in enumerate(tqdm(img_paths, desc="hist similarities", unit="img&nbrs")):
        for j in range(max(0, i - nb_range), min(n_images, i + nb_range + 1)):
            if i == j:
                continue
            edge_simil = calc_edge_simil(edge_hists[i], edge_hists[j])
            color_simil = calc_color_hist_simil(color_hists[i], color_hists[j])
            hist_simil = 0.3 * edge_simil + 0.7 * color_simil
            simil_mtx[i, j] = hist_simil
            if hist_simil > thr:
                i_cluster = uf.get_cluster(i)
                # check the similarity with every element of the cluster that is being extended
                if len(i_cluster) == 1 or is_cluster_similar(j, i_cluster, simil_mtx, thr):
                    uf.union(i, j)
            print(f"Similarity of {i}-{j} pair is {hist_simil}")

    clusters = defaultdict(list)
    for i in range(n_images):
        root = uf.find(i)
        clusters[root].append(i)

    clusters_list = [cluster for cluster in list(clusters.values()) if len(cluster) > 1]
    if SHOW_CLUSTERS:
        for cluster in clusters_list:
            print(cluster)
            show_cluster(cluster, img_paths)
    # print(clusters_list)
    return clusters_list
# endregion

# region sift clusters

# inspired by LB
def compute_matches(descr_1, descr_2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    if descr_1 is None or descr_2 is None:
        print("[compute_matches]: At least one descriptor array is None")
        return []

    if len(descr_1) < 2 or len(descr_2) < 2:
        print("[compute_matches]: At least one descriptor array has len < 2")
        return []

    # get cv2.DMatch objs
    matches_12 = bf.knnMatch(descr_1, descr_2, k=2)
    matches_21 = bf.knnMatch(descr_2, descr_1, k=2)

    matches_12_robust = []
    matches_21_robust = []

    # Filter both directions first - Lowe ratio check from SIFT paper
    for m1_first, m1_second in matches_12:
        if m1_first.distance < 0.75 * m1_second.distance:
            matches_12_robust.append(m1_first)

    for m2_first, m2_second in matches_21:
        if m2_first.distance < 0.75 * m2_second.distance:
            matches_21_robust.append(m2_first)

    # Symmetry check
    matches_robust = []
    for m1 in matches_12_robust:
        for m2 in matches_21_robust:
            if m1.queryIdx == m2.trainIdx and m2.queryIdx == m1.trainIdx:
                matches_robust.append([m1])  # wrap if needed

    return matches_robust


def create_sift_clusters(img_paths, thr=0.7):
    sift = cv2.SIFT_create(nfeatures=N_SIFT_FEATS)  # SIFT algorithm with number of keypoints
    n_images = len(img_paths)
    nb_range = NEIGHBORS_RANGE

    keypoints = {}
    descriptors = {}
    simil_mtx = np.zeros((n_images, n_images))
    np.fill_diagonal(simil_mtx, 1.0)
    for i, img_path in enumerate(tqdm(img_paths, desc="SIFT features", unit="img")):
        # img = img_resize(cv2.imread(str(img_path)), SIFT_RES)
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)  # apply EXIF orientation
        img = np.asarray(img)
        img_tfd = img_resize(img, max_d=SIFT_RES, tf_option=1)
        img_uint8 = (img_tfd * 255).astype(np.uint8)

        keypoints[i], descriptors[i] = sift.detectAndCompute(img_uint8, None)

    uf = UnionFind(n_images)
    for i in tqdm(range(n_images), desc="SIFT matches", unit="img&nbrs"):
        for j in range(max(0, i - nb_range), min(n_images, i + nb_range + 1)):
            if i == j:
                continue

            sift_simil = simil_mtx[i, j]
            if sift_simil == 0:
                matches = compute_matches(descriptors[i], descriptors[j])
                sift_simil = pow((len(matches) / N_SIFT_FEATS), 1/8)
                simil_mtx[i, j] = sift_simil
                simil_mtx[j, i] = sift_simil  # spare future calculations

            if sift_simil > thr:
                i_cluster = uf.get_cluster(i)
                # check the similarity with every element of the cluster that is being extended
                if len(i_cluster) == 1 or is_cluster_similar(j, i_cluster, simil_mtx, thr):
                    uf.union(i, j)
            # print(f"Similarity of {i}-{j} pair is {sift_simil}")

    clusters = defaultdict(list)
    for i in range(n_images):
        root = uf.find(i)
        clusters[root].append(i)

    clusters_list = [cluster for cluster in list(clusters.values()) if len(cluster) > 1]
    if SHOW_CLUSTERS:
        for cluster in clusters_list:
            print(cluster)
            show_cluster(cluster, img_paths)
    # print(clusters_list)
    return clusters_list
# endregion

# region clip clusters

def create_clip_clusters(img_paths, thr=0.8, batch_size=32, cuda=True):
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)

    contents = []
    batch = []

    with torch.no_grad():
        for img_path in tqdm(img_paths, desc="CLIP contents", unit="img"):

            img = Image.open(img_path)
            img_tfd = preprocess(img)
            batch.append(img_tfd)

            # When batch is full → predict
            if len(batch) == batch_size:
                batch_torch = torch.stack(batch).to(device)  # [B, C, H, W]
                f_batch = model.encode_image(batch_torch)
                contents.append(f_batch.cpu().numpy())
                batch.clear()

        # Last partial batch
        if batch:
            batch_torch = torch.stack(batch).to(device)  # [B, C, H, W]
            f_batch = model.encode_image(batch_torch)
            contents.append(f_batch.cpu().numpy())

        # Convert to numpy array for faster indexing later
        contents = np.concatenate(contents)
        contents /= np.linalg.norm(contents, axis=1, keepdims=True)

    nb_range = NEIGHBORS_RANGE
    n_images = len(img_paths)
    # creating matrix with size in the order of MB
    if n_images <= 1000:
        simil_mtx = contents @ contents.T
    else:
        simil_mtx = np.zeros((n_images, n_images))
        np.fill_diagonal(simil_mtx, 1.0)

    uf = UnionFind(n_images)
    for i in tqdm(range(n_images), desc="CLIP similarities", unit="img&nbrs"):
        for j in range(max(0, i - nb_range), min(n_images, i + nb_range + 1)):
            if i == j:
                continue

            clip_simil = simil_mtx[i, j]
            if clip_simil == 0:
                clip_simil = contents[i] @ contents[j]
                simil_mtx[i, j] = clip_simil
                simil_mtx[j, i] = clip_simil  # spare future calculations

            if clip_simil > thr:
                i_cluster = uf.get_cluster(i)
                # check the similarity with every element of the cluster that is being extended
                if len(i_cluster) == 1 or is_cluster_similar(j, i_cluster, simil_mtx, thr):
                    uf.union(i, j)
            print(f"Similarity of {i}-{j} pair is {clip_simil}")

    clusters = defaultdict(list)
    for i in range(n_images):
        root = uf.find(i)
        clusters[root].append(i)

    clusters_list = [cluster for cluster in list(clusters.values()) if len(cluster) > 1]
    if SHOW_CLUSTERS:
        for cluster in clusters_list:
            print(cluster)
            show_cluster(cluster, img_paths)
    # print(clusters_list)
    return clusters_list
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

    # img_clusters = create_man_clusters(img_paths, thr=CLUSTER_DIFF_THR, max_mult=CLUSTER_MAX_MULT)
    # data = {
    #     "clusters":     [str(cluster) for cluster in img_clusters],  # for better json formatting
    #     "image_refs":   [f"{img_path.name}, {i}" for i, img_path in enumerate(img_paths)]
    # }
    # save_results_versioned(paths_cfg, data, "clusters_manual", save_method="json")

    img_clusters = {
        "clusters_time": create_time_clusters(img_paths, thr=CLUSTER_DIFF_THR, max_mult=CLUSTER_MAX_MULT),
        "clusters_hists": create_hist_clusters(img_paths, thr=THR_HIST),
        "clusters_sift": create_sift_clusters(img_paths, thr=THR_SIFT),
        "clusters_clip": create_clip_clusters(img_paths, thr=THR_CLIP)
    }
    method_names = img_clusters.keys()

    for method_name in method_names:
        data = {
            "clusters":     [str(cluster) for cluster in img_clusters[method_name]],  # str for better json formatting
            "image_refs":   [f"{img_path.name}, {i}" for i, img_path in enumerate(img_paths)]
        }
        save_results_versioned(paths_cfg, data, method_name, save_method="json")

    # # additional cluster checking
    # cluster_to_see = [2688]
    # show_cluster(cluster_to_see, img_paths)

    # load the cluster data for comparison
    data_r = load_results_versioned(paths_cfg, "clusters_manual", load_method="json")
    man_clusters = [json.loads(cluster) for cluster in data_r["clusters"]]
    if "image_refs" in data_r:
        img_paths_pairs = []
        for pair_str in data_r["image_refs"]:
            name, idx = pair_str.split(",")
            img_paths_pairs.append((name.strip(), int(idx)))
        img_paths = [dataset_path / p for p, _ in sorted(img_paths_pairs, key=lambda x: x[1])]

    list_of_img_clusters = []
    method_names = ["clusters_time", "clusters_hists", "clusters_sift", "clusters_clip"]
    for method_name in method_names:
        data_r = load_results_versioned(paths_cfg, method_name, load_method="json")
        method_clusters = [json.loads(cluster) for cluster in data_r["clusters"]]
        list_of_img_clusters.append(method_clusters)

    # compare the clusters from each method with manual
    method_tps = [0] * len(list_of_img_clusters)
    total_pairs = 0
    nb_range = NEIGHBORS_RANGE
    for i in tqdm(range(n_images), desc="SIFT matches", unit="img&nbrs"):
        for j in range(max(0, i - nb_range), min(n_images, i + nb_range + 1)):
            if j >= i:
                continue

            same_man_cluster = False
            for cluster in man_clusters:
                if i in cluster and j in cluster:
                    same_man_cluster = True
                    break

            for k, method_clusters in enumerate(list_of_img_clusters):
                same_cluster = False
                for cluster in method_clusters:
                    if i in cluster and j in cluster:
                        same_cluster = True
                        break
                if same_man_cluster == same_cluster:
                    method_tps[k] += 1
            total_pairs += 1

    for i, method_name in enumerate(method_names):
        print(f"{method_name}: {method_tps[i]} / {total_pairs}")


    # # remove all files with clusters
    # remove_all_files_by_name(paths_cfg["results_root"], "clusters_manual")
