from pathlib import Path
import os
import time

# treshold
import numpy as np
import networkx as nx

# hierarchical
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

# spectral
from sklearn.cluster import SpectralClustering
from torchmetrics.functional.clustering import normalized_mutual_info_score

from utils import *

from brisque_eval import compute_brisque_scores
from nima_eval import compute_nima_scores
from sift_eval import compute_sift_similarities
from efnetv2_eval import compute_efnetv2_similarities

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/jk/namibie_corrected/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/LIVEwild/Images/trainingImages/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/tid2013/distorted_images"

RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MAX_IMAGES = 100
SAVE_SCORE_EXIF = False
DEF_THRESHOLD = 0.5

SAVE_STATS = True
RECOMPUTE = True
OVERRIDE = True

def normalize_matrix(matrix):
    matrix = np.asarray(matrix)

    # get rid of -inf
    matrix = np.clip(matrix, 0, None)

    matrix = matrix / matrix.max() # normalize to range 0..1
    np.fill_diagonal(matrix, 1)

    return matrix  # return numpy matrix


# create clusters using Disjoint-set (Union-Find) greedy approach
def create_clusters_greedy(scores, thr=DEF_THRESHOLD, do_half=False):
    scores = normalize_matrix(scores)

    # create pairs wiht scores and sort them
    pairs_with_scores = [((i, j), scores[i][j]) for i in range(len(scores)) for j in range(len(scores[0])) if i != j]
    # sorted_pairs = sorted(pairs_with_scores, key=lambda x: x[1], reverse=True)[:k_best]
    sorted_pairs = sorted(pairs_with_scores, key=lambda x: x[1], reverse=True)

    sorted_values = [float(pair[1]) for pair in sorted_pairs]
    print(sorted_values[:100])

    n = len(scores) # number of images
    parent = list(range(n))
    rank = [0] * n

    def get_root(x):
        if parent[x] != x:
            parent[x] = get_root(parent[x])  # path compression
        return parent[x]

    def union(x, y):
        root_x = get_root(x)
        root_y = get_root(y)

        if root_x == root_y:
            return False  # already same cluster

        # union by rank
        if rank[root_x] < rank[root_y]:
            parent[root_x] = root_y
        elif rank[root_x] > rank[root_y]:
            parent[root_y] = root_x
        else:
            parent[root_y] = root_x
            rank[root_x] += 1

        return True

    # Greedy merging
    count = 0
    n_pairs = len(sorted_pairs)
    for (i, j), score in sorted_pairs:
        if not do_half and (thr is not None and score < thr):
            break
        if do_half and (count >= n_pairs/2):
            print(f"threshold of half: {score}")
            break
        print(f"({i}, {j}), {score}")
        union(i, j)
        count += 1

    # Collect clusters
    clusters = {}
    for i in range(n):
        root = get_root(i)
        clusters.setdefault(root, []).append(i)

    # print(clusters.values())
    for cluster in clusters:
        if len(cluster) > 1:
            print(cluster)
    return list(clusters.values())


def create_clusters_treshold(scores, thr=DEF_THRESHOLD):
    scores = normalize_matrix(scores)

    G = nx.Graph()

    n = scores.shape[0]
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if scores[i, j] > thr:
                G.add_edge(i, j)

    clusters = list(nx.connected_components(G))
    print(clusters)
    return clusters


def create_clusters_hierarchical(scores, dist_thr=1-DEF_THRESHOLD):
    scores = normalize_matrix(scores)

    distances= 1 - scores
    condensed = squareform(distances)

    Z = linkage(condensed, method='average')

    clusters = fcluster(Z, t=dist_thr, criterion='distance')
    print(clusters)
    return clusters


def create_clusters_spectral(scores):
    scores = normalize_matrix(scores)

    from sklearn.cluster import SpectralClustering

    clustering = SpectralClustering(n_clusters=2, affinity='precomputed')

    labels = clustering.fit_predict(scores)
    print(labels)
    return labels


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

    scores = {
        # "brisque":  compute_brisque_scores(paths_cfg, img_paths),
        "nima":     compute_nima_scores(paths_cfg, img_paths),
        "sift":     compute_sift_similarities(paths_cfg, img_paths),
        # "efnetv2":  compute_efnetv2_similarities(paths_cfg, img_paths)
    }
    # np.random.normal(loc=50.0, scale=10.0, size=(n_images)).tolist()

    create_clusters_greedy(scores["sift"], thr=0.5)
    # create_clusters_treshold(scores["sift"])
    # create_clusters_hierarchical(scores["sift"])
    # create_clusters_spectral(scores["sift"])
    exit(0)

    sel_percent = 10
    n_bins = 100 // sel_percent
    selection = [1.0 if i % n_bins == 0 else 0.0 for i in range(len(img_paths))]

    scores = []
    viewer = ImageViewer(img_paths, scores, mode='select')
    viewer.update_selection(selection)

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)
