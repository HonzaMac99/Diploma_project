import torch
import clip
import tensorflow as tf
from scipy.spatial import distance
from tqdm import tqdm
import time

from keras import preprocessing
from keras.applications.efficientnet_v2 import preprocess_input, EfficientNetV2B1

from utils import *

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

WEIGHTS_PATH = Path.cwd() / "data/efficientnetv2-b1.h5"

MAX_IMAGES = None # [int|None] maximum number of images to process (for debugging)
N_NEIGHBORS = 20
# EFNETV2 has fixed input img size: orig_res = [3000 x 4000] --> [240, 240]
EFNETV2_RES = 240

SHOW_IMAGES = True
SAVE_SCORE_EXIF = False

SAVE_STATS = False
RECOMPUTE = True
OVERRIDE = True


def compute_clip_similarities(paths_cfg, img_paths, batch_size=32, cuda=True):
    save_file_base = "clip_scores"

    # # ver_idx = 0
    # clip_scores = load_results_versioned(paths_cfg, save_file_base, load_method="npz")
    clip_scores = None

    if clip_scores is None or len(clip_scores) != len(img_paths):

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

        n_images = len(img_paths)
        if n_images <= 1000:
            simil_mtx = contents @ contents.T
        else:
            n_neighbors = N_NEIGHBORS
            simil_mtx = np.zeros((n_images, n_images))
            np.fill_diagonal(simil_mtx, 1.0)
            for i in tqdm(range(n_images), desc="CLIP similarities", unit="img&nbrs"):
                for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
                    if i == j:
                        continue
                    simil_mtx[i, j] = contents[i] @ contents[j]

        # save scores after computation
        save_results_versioned(paths_cfg, simil_mtx, save_file_base, save_method="npz")

    return simil_mtx


# region Other experimental functions

# inspired by LB
def compute_contents(img, model):
    t = preprocessing.image.img_to_array(img)
    t = np.expand_dims(t, axis=0)
    t = preprocess_input(t)
    f = model.predict(t, verbose=0, batch_size=8)
    f = f.tolist()
    return f[0]

# inspired by LB
def compute_imgs_contents(target_path, recompute=False, target_res=240):
    path = Path(target_path)

    if path.exists() and not recompute:
        return np.load(path) # load the .npz file

    # todo: resolve tf incompatible gpu drivers
    device_str = '/cpu:0'
    # gpus = tf.config.list_physical_devices('GPU')
    # device_str = '/gpu:0' if gpus else '/cpu:0'

    with tf.device(device_str):
        # todo: get our own weights
        # class_model = EfficientNetV2B1(weights="imagenet")
        model = EfficientNetV2B1(weights="data/efficientnetv2-b1.h5")
        dataset_stats = {
            "ids": [],
            "img_paths": [],
            "contents": []
        }

        times_compute = []
        for i, img_path in enumerate(img_paths):
            start_t = time.time()
            img_tfd = preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(target_res, target_res))
            img_contents = compute_contents(img_tfd, model)
            end_t = time.time()
            times_compute.append(end_t-start_t)

            dataset_stats["ids"].append(i)
            dataset_stats["img_paths"].append(img_paths[i])
            dataset_stats["contents"].append(img_contents)

        # with open(Path.cwd() / target_path), "w") as write_file:
        #     json.dump(content_list, write_file, indent=2)

        np.savez(target_path, ids=dataset_stats["ids"],
                              img_paths=dataset_stats["img_paths"],
                              contents=dataset_stats["contents"])
    return dataset_stats, times_compute

# inspired by LB
def compute_similarities(img_paths, n_neighbors):
    global viewer
    plt.ion()

    target_res = EFNETV2_RES
    times_compute = []

    path = Path("results/efnetv2_contents.npz")
    if path.exists():
        imgs_contents = np.load(path)["contents"]
    else:
        print("Computing EfNetV2 contents:")
        imgs_data, times_compute = compute_imgs_contents(path, target_res=target_res)
        imgs_contents = imgs_data["contents"]

    print("Computing EfNetV2 similarity scores:")
    n_images = len(img_paths)
    img_stats_list = []  # list to store results
    efnetv2_scores = np.ones((n_images, n_images)) * (-1)
    viewer.scores = efnetv2_scores

    times_match = []
    for i, img_1_path in enumerate(img_paths):
        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                continue

            img_2_path = img_paths[j]

            start_t = time.time()
            efnetv2_score = (1 - distance.cdist([imgs_contents[i]], [imgs_contents[j]], 'cosine').item()) * 100 # 15
            end_t = time.time()
            times_match.append(end_t - start_t)

            efnetv2_scores[i, j] = efnetv2_score

            print(f"({i+1}, {j+1}) score: {efnetv2_score}")

            # todo: also some data?

            img_stats = {
                "id_1": i,
                "id_2": j,
                "img_1": str(img_1_path),
                "img_2": str(img_2_path),
                "efnetv2_similarity_score": efnetv2_score,  # todo: list for more resolutions?
                "resolution": target_res,
            }

            if SHOW_IMAGES:
                viewer.idx1 = i
                viewer.idx2 = j
                viewer.show_current(interactive=False)

            img_stats_list.append(img_stats)

    avg_time_detect = sum(times_compute) / len(times_compute)
    avg_time_match = sum(times_match) / len(times_match)
    print("-----------------------------------------")
    print(f"{target_res}: {avg_time_detect:.4f}, {avg_time_match:.4f}")
    print("-----------------------------------------")

    imgs_stats = {
        "description": "EffectiveNetV2-B1 statistics of image pairs from dataset computed across multiple resolutions",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "author": "Jan Macalík",
        "num_images": len(img_paths),
        "num_resolutions": 1,
        "statistics": img_stats_list,
        # "features": features
    }

    # result_pth = "results/image_statistics.json"
    # result_pth = Path.cwd() / result_pth
    # result_pth.parent.mkdir(parents=True, exist_ok=True)

    # with open(result_pth, "w") as write_file:
    #     json.dump(data, write_file, indent=2, ensure_ascii=False)

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
    method_name = "Efnetv2"
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

    # measuring times for batching influence on speed
    batch_sizes = [1, 2, 16, 32, 64]
    for b_size in batch_sizes:
        start_t = time.time()
        compute_efnetv2_similarities(paths_cfg, img_paths, batch_size=b_size)
        end_t = time.time()
        time_diff = end_t-start_t
        print(f"B {b_size}: {time_diff:.4f}")
    exit()

    scores = []
    viewer = ImageViewer(img_paths, scores, mode='dual', tool_name=method_name)

    method_stats = {}
    file_name_base = "efnetv2_stats_experimental"
    ver_idx = None

    if not RECOMPUTE:
        method_stats = load_results_versioned(paths_cfg, file_name_base, ver_idx=ver_idx, load_method="json")

    if method_stats:
        print_scores(method_stats)
        viewer.scores = get_scores_json(method_stats)
    else:
        method_stats = compute_similarities(img_paths, N_NEIGHBORS)
        if SAVE_STATS:
            save_path = save_results_versioned(paths_cfg, method_stats, file_name_base, save_method="json",
                                               override_last=OVERRIDE)
            print(f"Saved new data as: {save_path}")

    plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)
