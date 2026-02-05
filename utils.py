import json
import hashlib
from pathlib import Path
from datetime import datetime
import sys

# todo: refine this prototype

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
