import torch
from tqdm import tqdm
from scipy.spatial import distance
import skimage

from torch import randint
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment

from utils import *

DATASET_ROOT = "/home/honzamac/Edu/m5/Projekt_D/datasets/"
DATASET_PATH = "/home/honzamac/Edu/m5/Projekt_D/datasets/kaohsiung/full/"
RESULTS_ROOT = "/home/honzamac/Edu/m5/Projekt_D/projekt_testing/results/"
IMG_EXTS = {".bmp", ".png", ".jpg", ".jpeg"}

MAX_IMAGES = None # maximum number of images to process (for debugging)
N_NEIGHBORS = 20
IMG_NUM_RES = 1    # orig_res = [3000 x 4000] --> [224, 244] (fixed nima input size)
CLIP_RES = 512 # 1024

SHOW_IMAGES = True
SAVE_SCORE_EXIF = False

SAVE_STATS = True
RECOMPUTE = False
OVERRIDE = True

_clip_obj = None

def get_clip():
    global _clip_obj
    if _clip_obj is None:
        _clip_obj = CLIPImageQualityAssessment()
    return _clip_obj


class CLIPDataset(Dataset):
    def __init__(self, img_paths, max_dim):
        self.img_paths = img_paths
        self.max_dim = max_dim
        # self.transform = transforms.ToTensor()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)), # ensuring fixed size for batch stacking
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.img_paths[idx]).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img = np.array(img)
            img = img_resize(img, max_d=self.max_dim, tf_option=1)
            return self.transform(img)
        except Exception as e:
            print(f"Skipping corrupted image: {self.img_paths[idx]} — {e}")
            return None


def collate_skip_none(batch):
    batch = [img for img in batch if img is not None]
    if not batch:
        return None
    return torch.stack(batch)


def compute_clip_scores(paths_cfg, img_paths, cuda=True, max_dim=None, batch_size=64, save_scores=True, load_scores=True):
    save_file_base = "clip-iqa_scores"

    scores = None
    if load_scores:
        scores = load_results_versioned(paths_cfg, save_file_base, load_method="npz")

    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")

    if scores is None or len(scores) != len(img_paths):
        clip_obj = get_clip().to(device)
        max_dim = max_dim if max_dim is not None else CLIP_RES

        dataset = CLIPDataset(img_paths, max_dim)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=collate_skip_none,
        )

        scores = []
        clip_obj.eval()
        for batch in tqdm(loader, desc="CLIP-IQA", unit="img", unit_scale=batch_size):
            if batch is None:
                continue
            with torch.no_grad():
                batch = batch.to(device)
                clip_score = clip_obj(batch)
                scores.extend(clip_score.cpu().tolist())

        if save_scores:
            save_results_versioned(paths_cfg, scores, save_file_base, save_method="npz")

    return scores

if __name__ == "__main__":
    method_name = "Clip-iqa"
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

    scores = compute_clip_scores(paths_cfg, img_paths)
    viewer = ImageViewer(img_paths, scores, mode='single', tool_name=method_name)

    # plt.ioff()
    viewer.fig.canvas.mpl_connect('key_press_event', lambda event: viewer.on_key(event))
    viewer.show_current(interactive=False)

