import torch
import clip
import tensorflow as tf
from scipy.spatial import distance
from tqdm import tqdm
import time

from torch.utils.data import Dataset, DataLoader

from utils import *

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
# DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/full/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/grenoble/full/"
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

WEIGHTS_PATH = Path.cwd() / "data/efficientnetv2-b1.h5"

MAX_IMAGES = 150 # [int|None] maximum number of images to process (for debugging)
N_NEIGHBORS = 20
# CLIP_RES = 224 # automatically solved by the implemented clip preprocess
CLIP_THR = 0.95

SHOW_IMAGES = False
SAVE_SCORE_EXIF = False

SAVE_STATS = False
RECOMPUTE = True
OVERRIDE = True


class CLIPSimilarityDataset(Dataset):
    def __init__(self, img_paths, preprocess):
        self.img_paths = img_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert("RGB")
            return self.preprocess(img)
        except Exception as e:
            print(f"Skipping corrupted image: {self.img_paths[idx]} — {e}")
            return None


def collate_skip_none(batch):
    batch = [img for img in batch if img is not None]
    if not batch:
        return None
    return torch.stack(batch)


def compute_clip_embeddings(img_paths, batch_size=32, cuda=True):
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)

    dataset = CLIPSimilarityDataset(img_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_skip_none,
    )

    contents = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="CLIP embeddings"):
            if batch is None:
                continue
            batch = batch.to(device)
            features = model.encode_image(batch)
            contents.append(features.cpu().numpy())

    contents = np.concatenate(contents)
    contents /= np.linalg.norm(contents, axis=1, keepdims=True)

    return contents


def compute_clip_similarities(paths_cfg, img_paths, batch_size=32, cuda=True,
                              save_scores=True, load_scores=True):

    save_file_base = "clip_scores"

    simil_mtx = None
    if load_scores:
        simil_mtx = load_results_versioned(paths_cfg, save_file_base, load_method="npz")

    if simil_mtx is None or len(simil_mtx) != len(img_paths):

        contents = compute_clip_embeddings(img_paths, batch_size, cuda)

        n_images = len(img_paths)

        # Fast full similarity
        if n_images <= 1000:
            simil_mtx = contents @ contents.T
        else:
            # Sparse neighborhood similarity
            simil_mtx = np.zeros((n_images, n_images))
            np.fill_diagonal(simil_mtx, 1.0)

            for i in tqdm(range(n_images), desc="CLIP similarities"):
                for j in range(max(0, i - N_NEIGHBORS), min(n_images, i + N_NEIGHBORS + 1)):
                    if i == j:
                        continue
                    simil_mtx[i, j] = contents[i] @ contents[j]

        if save_scores:
            save_results_versioned(paths_cfg, simil_mtx, save_file_base, save_method="npz")

    return simil_mtx


# region other experimental functions

def get_resized_img(img_path, max_dim_len):
    img = Image.open(img_path)
    img = ImageOps.exif_transpose(img)  # apply EXIF orientation
    img = np.asarray(img)
    img_tfd = img_resize(img, max_d=max_dim_len, tf_option=1)
    img_tfd = (img_tfd * 255).astype(np.uint8)
    cv2.cvtColor(img_tfd, cv2.COLOR_RGB2GRAY)
    return img_tfd


def show_pair(i, j, img_paths, sim_score, max_dim_len=1024):

    plt.close("all")

    img1 = get_resized_img(img_paths[i], max_dim_len)
    img2 = get_resized_img(img_paths[j], max_dim_len)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Show images
    axes[0].imshow(img1)
    axes[0].axis("off")

    axes[1].imshow(img2)
    axes[1].axis("off")

    plt.suptitle(f"Similarity score: {sim_score:.4f}", fontsize=16)

    plt.tight_layout()
    plt.show(block=True)


def compute_clip_similarities_exp(img_paths, n_neighbors, batch_size=32, cuda=True):
    global viewer
    plt.ion()

    contents = compute_clip_embeddings(img_paths, batch_size, cuda)

    print("Computing CLIP similarity scores...")
    n_images = len(img_paths)

    scores = np.ones((n_images, n_images)) * (-1)
    viewer.scores = scores

    img_stats_list = []

    for i in range(n_images):
        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                continue

            clip_sim_score = contents[i] @ contents[j]
            scores[i, j] = clip_sim_score

            print(f"({i+1}, {j+1}) score: {clip_sim_score:.4f}")

            img_stats = {
                "id_1": i,
                "id_2": j,
                "img_1": str(img_paths[i]),
                "img_2": str(img_paths[j]),
                "clip_similarity_score": float(clip_sim_score),
            }

            if SHOW_IMAGES:
                viewer.idx1 = i
                viewer.idx2 = j
                viewer.show_current(interactive=False)

            print(clip_sim_score)

            # if [i, j] in [[141, 142], [142, 143]]:
            if clip_sim_score > CLIP_THR:
                show_pair(i, j, img_paths, clip_sim_score)

            img_stats_list.append(img_stats)

    imgs_stats = {
        "description": "CLIP similarity statistics of image pairs",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": n_images,
        "statistics": img_stats_list,
    }

    return imgs_stats


def get_scores_json(method_stats):
    n_images = method_stats["num_images"]
    scores = np.ones((n_images, n_images)) * (-1)
    img_stats_data = method_stats["statistics"]

    for stat in img_stats_data:
        i = stat["id_1"]
        j = stat["id_2"]
        if i < n_images and j < n_images:
            score = stat["efnetv2_similarity_score"]
            scores[i, j] = score

    return scores


def print_scores(dataset_stats):
    print("Printing Efnetv2 scores:")
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

# endregion

if __name__ == "__main__":
    method_name = "CLIP"
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

    paths_cfg = {
        "dataset_root": DATASET_ROOT,
        "dataset_path": DATASET_PATH,
        "results_root": RESULTS_ROOT
    }

    # # measuring times for batching influence on speed
    # batch_sizes = [1, 2, 16, 32, 64]
    # for b_size in batch_sizes:
    #     start_t = time.time()
    #     compute_clip_similarities(paths_cfg, img_paths, batch_size=b_size, load_scores=False, save_scores=False)
    #     end_t = time.time()
    #     time_diff = end_t-start_t
    #     print(f"B {b_size}: {time_diff:.4f}")
    # exit()

    scores = []
    viewer = ImageViewer(img_paths, scores, mode='dual', tool_name=method_name)

    method_stats = {}
    file_name_base = "clip_stats_experimental"
    ver_idx = None

    if not RECOMPUTE:
        method_stats = load_results_versioned(paths_cfg, file_name_base, ver_idx=ver_idx, load_method="json")

    if method_stats:
        print_scores(method_stats)
        viewer.scores = get_scores_json(method_stats)
    else:
        method_stats = compute_clip_similarities_exp(img_paths, N_NEIGHBORS)
        if SAVE_STATS:
            save_path = save_results_versioned(paths_cfg, method_stats, file_name_base, save_method="json",
                                               override_last=OVERRIDE)
            print(f"Saved new data as: {save_path}")

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)
