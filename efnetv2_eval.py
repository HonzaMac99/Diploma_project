import time

from torch.distributed.rpc.api import method_name
from tqdm import tqdm
from scipy.spatial import distance

import tensorflow as tf
from keras import preprocessing
from keras.applications.efficientnet_v2 import preprocess_input, EfficientNetV2B1

from utils import *

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/selected_r30/"
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

WEIGHTS_PATH = Path.cwd() / "data/efficientnetv2-b1.h5"

MAX_IMAGES = 3 # maximum number of images to process (for debugging)
N_NEIGHBORS = 20
IMG_NUM_RES = 1    # orig_res = [3000 x 4000] --> [224, 244] (fixed nima input size)

SHOW_IMAGES = True
SAVE_SCORE_EXIF = False

SAVE_STATS = True
RECOMPUTE = False
OVERRIDE = True


def compute_efnetv2_similarities(paths_cfg, img_paths):
    save_file_base = "efnetv2_scores"

    # ver_idx = 0
    efnetv2_scores = load_results_versioned(paths_cfg, save_file_base, load_method="npz")

    if efnetv2_scores is None or len(efnetv2_scores) != len(img_paths):
        # todo: resolve tf incompatible gpu drivers
        # gpus = tf.config.list_physical_devices('GPU')
        # device_str = '/gpu:0' if gpus else '/cpu:0'
        device_str = '/cpu:0'

        contents = []

        with tf.device(device_str):
            # todo: get our own weights
            # class_model = EfficientNetV2B1(weights="imagenet")
            model = EfficientNetV2B1(weights=WEIGHTS_PATH)

            for i, img_path in enumerate(tqdm(img_paths, desc="EFNETV2 contents", unit="img")):
                # todo: compute times
                img_tfd = preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(240, 240))
                img_contents = compute_contents(img_tfd, model)

                contents.append(img_contents)

        n_images = len(img_paths)
        n_neighbors = N_NEIGHBORS
        efnetv2_scores = np.full((n_images, n_images), -np.inf)

        for i, img_name in enumerate(tqdm(img_paths, desc="EFNETV2 similarities", unit="img&nbrs")):
            for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
                if i == j:
                    if j+1 == MAX_IMAGES:
                        break
                    else:
                        continue

                efnetv2_score = (1 - distance.cdist([contents[i]], [contents[j]], 'cosine').item()) * 100 # * 15
                efnetv2_scores[i, j] = efnetv2_score

        # save scores after computation
        save_results_versioned(paths_cfg, efnetv2_scores, save_file_base, save_method="npz")

    return efnetv2_scores

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
def compute_imgs_contents(target_path, recompute=False):
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
        for i, img_path in enumerate(img_paths):
            # todo: compute times
            img_tfd = preprocessing.image.load_img(img_path, color_mode='rgb', target_size=(240, 240))
            img_contents = compute_contents(img_tfd, model)

            dataset_stats["ids"].append(i)
            dataset_stats["img_paths"].append(img_paths[i])
            dataset_stats["contents"].append(img_contents)

        # with open(Path.cwd() / target_path), "w") as write_file:
        #     json.dump(content_list, write_file, indent=2)

        np.savez(target_path, ids=dataset_stats["ids"],
                              img_paths=dataset_stats["img_paths"],
                              contents=dataset_stats["contents"])
    return dataset_stats

# inspired by LB
def compute_similarities(img_paths, n_neighbors):
    global viewer
    plt.ion()

    path = Path("results/efnetv2_contents.npz")
    if path.exists():
        imgs_contents = np.load(path)["contents"]
    else:
        print("Computing EfNetV2 contents:")
        imgs_contents = compute_imgs_contents(path)["contents"]

    print("Computing EfNetV2 similarity scores:")
    n_images = len(img_paths)
    img_stats_list = []  # list to store results
    efnetv2_scores = np.ones((n_images, n_images)) * (-1)
    viewer.scores = efnetv2_scores

    for i, img_1_path in enumerate(img_paths):
        for j in range(max(0, i - n_neighbors), min(n_images, i + n_neighbors + 1)):
            if i == j:
                continue

            img_2_path = img_paths[j]

            efnetv2_score = (1 - distance.cdist([imgs_contents[i]], [imgs_contents[j]], 'cosine').item()) * 100 # 15
            efnetv2_scores[i, j] = efnetv2_score

            print(f"({i+1}, {j+1}) score: {efnetv2_score}")

            # todo: also some data?

            img_stats = {
                "id_1": i,
                "id_2": j,
                "img_1": str(img_1_path),
                "img_2": str(img_2_path),
                "efnetv2_similarity_score": efnetv2_score  # todo: list for more resolutions?
                # todo: times?
            }

            # todo: exif?

            if SHOW_IMAGES:
                viewer.idx1 = i
                viewer.idx2 = j
                viewer.show_current(interactive=False)

            img_stats_list.append(img_stats)

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
