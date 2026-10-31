"""
MVTec LOCO AD - Industrial Defect Detection Pipeline
Exported from GC_2_CODE_Web_App_Fixed_v2.ipynb
Candidate: Aryaman Srimali Jain | Group Number: 19
"""


# %% [markdown] Cell 1
# # MVTec LOCO AD — Logical Defect Detection Pipeline
# 
# Self-contained, end-to-end notebook: dataset check → feature extraction → memory bank →
# threshold → scoring → evaluation → visualization.
# 
# **Before running:** set `DATA_ROOT` and `CATEGORY` in the config cell below to match where
# you extracted the dataset. Then Run All.

# %% [markdown] Cell 2
# ## 0. Install / imports

# %% [code] Cell 3
# If any of these are missing, uncomment and run once:
# !pip install torch torchvision scikit-learn matplotlib pillow numpy joblib

import os
import json as _json
import threading
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

import joblib
import pandas as pd

# Serialises access to the shared feature extractor (its forward hooks keep state in a dict,
# so two simultaneous calls -- e.g. two web-app users -- would corrupt each other).
_INFER_LOCK = threading.Lock()

print("Imports OK")


# %% [markdown] Cell 4
# ## 1. Configuration
# 
# Edit `DATA_ROOT` to point at the folder that directly contains `breakfast_box/`,
# `juice_bottle/`, `pushpins/`, `screw_bag/`, `splicing_connectors/`.

# %% [code] Cell 5
CATEGORY = "juice_bottle"             # breakfast_box | juice_bottle | pushpins | screw_bag | splicing_connectors


def find_data_root(category):
    """Look for the folder that directly contains <category>/ in the usual places.
    Override by setting the MVTEC_LOCO_ROOT environment variable, or just assign DATA_ROOT below."""
    candidates = []
    env = os.environ.get("MVTEC_LOCO_ROOT")
    if env:
        candidates.append(Path(env))
    cwd = Path.cwd()
    for base in (cwd, cwd.parent, cwd.parent.parent):
        for name in ("DATA", "data", "mvtec_loco_ad", "mvtec_loco_anomaly_detection"):
            candidates.append(base / name)
    candidates.append(Path.home() / "Downloads" / "mvtec_loco_anomaly_detection")
    for c in candidates:
        if (c / category).exists():
            return c, candidates
        if c.exists() and c.is_dir():          # one extra nesting level, e.g. DATA/mvtec_loco_anomaly_detection/
            for sub_dir in c.iterdir():
                if sub_dir.is_dir() and (sub_dir / category).exists():
                    return sub_dir, candidates
    return None, candidates


_found_root, _tried = find_data_root(CATEGORY)
DATA_ROOT = _found_root if _found_root is not None else Path.cwd() / "DATA"   # <-- or set your own path here
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 0   # set to 0 to avoid multiprocessing issues inside notebooks

BACKBONE_LAYERS = ["layer2", "layer3"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COOSET_SAMPLING_RATIO = 0.1
N_NEIGHBORS = 1
N_STD_FOR_THRESHOLD = 3.0

# ⚡ FAST EXECUTION ACCELERATOR
# Set FAST_MODE = True to run in ~1-2 minutes by reusing saved memory banks & benchmarks.
# Set FAST_MODE = False to re-train from scratch (~25+ minutes).
FAST_MODE = True

print(f"Device: {DEVICE}")
print(f"Category: {CATEGORY}")
print(f"Dataset root: {DATA_ROOT}  (exists: {DATA_ROOT.exists()})")

assert (DATA_ROOT / CATEGORY).exists(), (
    f"Could not find {CATEGORY!r} under {DATA_ROOT}. Searched:\n  " + "\n  ".join(str(p) for p in _tried)
    + "\nSet DATA_ROOT (or the MVTEC_LOCO_ROOT environment variable) to the folder that directly contains "
    + "breakfast_box/, juice_bottle/, pushpins/, screw_bag/ and splicing_connectors/."
)


# %% [markdown] Cell 6
# ## 2. Dataset classes
# 
# Wraps the `train/` `validation/` `test/{good,logical_anomalies,structural_anomalies}` layout.

# %% [code] Cell 7
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class NormalImageDataset(Dataset):
    """Loads train/ or validation/ images (all normal, label always 0)."""

    def __init__(self, category: str, split: str):
        assert split in ("train", "validation")
        folder = DATA_ROOT / category / split
        # Some extractions nest images one level deeper (e.g. train/good/*.png)
        # rather than directly in train/, so search recursively to handle both.
        self.files = sorted(folder.rglob("*.png"))
        if not self.files:
            raise FileNotFoundError(f"No images found in {folder} (searched recursively)")
        self.transform = get_transform()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img), str(self.files[idx])


class TestImageDataset(Dataset):
    """Loads all test images with image-level label (0=good, 1=anomalous) and subtype."""

    def __init__(self, category: str):
        self.category = category
        self.samples = []
        test_root = DATA_ROOT / category / "test"
        for subtype in ["good", "logical_anomalies", "structural_anomalies"]:
            folder = test_root / subtype
            if not folder.exists():
                continue
            label = 0 if subtype == "good" else 1
            for f in sorted(folder.rglob("*.png")):
                self.samples.append((f, label, subtype))
        if not self.samples:
            raise FileNotFoundError(f"No test images found under {test_root}")
        self.transform = get_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, subtype = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, subtype, str(path)

    def ground_truth_mask_path(self, image_path: str):
        p = Path(image_path)
        subtype = next((s for s in ("logical_anomalies", "structural_anomalies", "good") if s in p.parts), p.parent.name)
        if subtype == "good":
            return None
        gt_dir = DATA_ROOT / self.category / "ground_truth" / subtype / p.stem
        candidates = sorted(gt_dir.glob("*.png")) if gt_dir.exists() else []
        return candidates[0] if candidates else None


train_ds = NormalImageDataset(CATEGORY, "train")
val_ds = NormalImageDataset(CATEGORY, "validation")
test_ds = TestImageDataset(CATEGORY)

print(f"Train (normal): {len(train_ds)}")
print(f"Validation (normal): {len(val_ds)}")
print(f"Test: {len(test_ds)}")
labels = [s[1] for s in test_ds.samples]
print(f"  good: {labels.count(0)}, anomalous: {labels.count(1)}")


# %% [markdown] Cell 8
# ## 3. Quick visual sanity check
# 
# Shows one normal train image and one of each test subtype.

# %% [code] Cell 9
def _peek(folder, n=1):
    return sorted(Path(folder).rglob("*.png"))[:n]   # train/ keeps its images in a good/ sub-folder

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
paths_to_show = [
    ("train (normal)", (DATA_ROOT / CATEGORY / "train")),
    ("test/good", (DATA_ROOT / CATEGORY / "test" / "good")),
    ("test/logical_anomalies", (DATA_ROOT / CATEGORY / "test" / "logical_anomalies")),
    ("test/structural_anomalies", (DATA_ROOT / CATEGORY / "test" / "structural_anomalies")),
]
for ax, (title, folder) in zip(axes, paths_to_show):
    files = _peek(folder)
    if files:
        ax.imshow(Image.open(files[0]))
    ax.set_title(title, fontsize=9)
    ax.axis("off")
plt.tight_layout()
plt.show()


# %% [markdown] Cell 10
# ## 4. Pretrained feature extractor
# 
# WideResNet50 backbone with forward hooks on mid-level layers (patch-level features).

# %% [code] Cell 11
_BACKBONE_BUILDERS = {
    "wide_resnet50_2": lambda: torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V2"),
    "resnet18": lambda: torchvision.models.resnet18(weights="IMAGENET1K_V1"),
}


def load_backbone(name):
    """Build a pretrained backbone. The weights are downloaded once and cached; if the download is
    blocked (offline / firewall / DNS error) raise a message that says how to fix it."""
    try:
        return _BACKBONE_BUILDERS[name]()
    except KeyError:
        raise ValueError(f"Unknown backbone {name!r}; choose from {list(_BACKBONE_BUILDERS)}")
    except Exception as exc:                       # URLError, socket.gaierror, HTTP errors, corrupt cache ...
        ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
        raise RuntimeError(
            f"Could not load pretrained weights for {name!r}: {type(exc).__name__}: {exc}\n"
            f"The weights are downloaded from download.pytorch.org on first use. Either allow that "
            f"domain / reconnect, or download the .pth file in a browser and copy it into:\n  {ckpt_dir}"
        ) from exc


class _BackboneRegistry(dict):
    """Keeps the old BACKBONES[name]() calling style working."""
    def __missing__(self, key):
        return lambda: load_backbone(key)


BACKBONES = _BackboneRegistry({k: (lambda k=k: load_backbone(k)) for k in _BACKBONE_BUILDERS})


class PatchFeatureExtractor:
    def __init__(self, layers=None, device=None, backbone_name="wide_resnet50_2"):
        self.layers = layers or BACKBONE_LAYERS
        self.device = device or DEVICE
        self.backbone_name = backbone_name

        self.backbone = BACKBONES[backbone_name]()
        self.backbone.eval().to(self.device)
        for p in self.backbone.parameters():
            p.requires_grad = False

        self._features = {}
        for layer_name in self.layers:
            layer = getattr(self.backbone, layer_name)
            layer.register_forward_hook(self._make_hook(layer_name))

    def _make_hook(self, name):
        def hook(module, inp, out):
            self._features[name] = out.detach()
        return hook

    @torch.no_grad()
    def extract(self, images: torch.Tensor):
        images = images.to(self.device)
        self._features = {}
        _ = self.backbone(images)

        feats = [self._features[l] for l in self.layers]
        target_size = feats[0].shape[-2:]
        resized = [F.interpolate(f, size=target_size, mode="bilinear", align_corners=False)
                   for f in feats]
        combined = torch.cat(resized, dim=1)

        b, c, h, w = combined.shape
        patch_features = combined.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return patch_features.cpu(), (h, w)


extractor = PatchFeatureExtractor()

# sanity check
_dummy = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
_feats, _grid = extractor.extract(_dummy)
print(f"Patch features shape: {_feats.shape}, grid size: {_grid}")


# %% [markdown] Cell 12
# ## 5. Build memory bank from train (normal) images
# 
# Coreset-subsampled pool of normal patch features + a fitted nearest-neighbor index.

# %% [code] Cell 13
def greedy_coreset_sampling(features: np.ndarray, sampling_ratio: float, seed: int = 0):
    # Runs the distance computation on GPU (if available) via torch -- this loop does
    # n_select iterations, each comparing against the full pool, so it benefits
    # enormously from GPU parallelism at typical patch-pool sizes.
    n_select = max(1, int(len(features) * sampling_ratio))
    rng = np.random.default_rng(seed)

    try:
        projector = SparseRandomProjection(n_components="auto", eps=0.9, random_state=seed)
        projected = projector.fit_transform(features)
    except ValueError:
        # "auto" can ask for more dimensions than low-dimensional features have -> skip the projection
        projected = features

    projected_t = torch.tensor(projected, dtype=torch.float32, device=DEVICE)
    n = projected_t.shape[0]

    selected = [int(rng.integers(0, n))]
    min_dists = torch.norm(projected_t - projected_t[selected[0]], dim=1)

    for _ in range(n_select - 1):
        next_idx = int(torch.argmax(min_dists).item())
        selected.append(next_idx)
        new_dists = torch.norm(projected_t - projected_t[next_idx], dim=1)
        min_dists = torch.minimum(min_dists, new_dists)

    return features[selected]


class TorchNN:
    """Exact nearest-neighbour search on the GPU (CPU fallback) with sklearn's .kneighbors() interface.
    The brute-force sklearn search is far too slow for the 343k-patch baseline bank."""

    def __init__(self, bank, n_neighbors=1, device=None, chunk=512):
        self.bank = np.ascontiguousarray(bank, dtype=np.float32)
        self.n_neighbors = n_neighbors
        self.device = device or DEVICE
        self.chunk = chunk
        self._bank_t = None

    def kneighbors(self, X, n_neighbors=None):
        k = min(n_neighbors or self.n_neighbors, len(self.bank))
        if self._bank_t is None:
            self._bank_t = torch.from_numpy(self.bank).to(self.device)
        X = np.ascontiguousarray(X, dtype=np.float32)
        dists, idxs = [], []
        with torch.no_grad():
            for start in range(0, len(X), self.chunk):
                q = torch.from_numpy(X[start:start + self.chunk]).to(self.device)
                d = torch.cdist(q, self._bank_t)
                vals, ind = torch.topk(d, k=k, dim=1, largest=False)
                dists.append(vals.cpu().numpy())
                idxs.append(ind.cpu().numpy())
        return np.concatenate(dists), np.concatenate(idxs)


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

all_patches = []
grid_size = None
for images, _ in train_loader:
    patch_feats, grid_size = extractor.extract(images)
    b, n, c = patch_feats.shape
    all_patches.append(patch_feats.reshape(b * n, c).numpy())

saved_mb_file = OUTPUT_DIR / f"memory_bank_{CATEGORY}.joblib"
if FAST_MODE and saved_mb_file.exists():
    print(f"⚡ FAST MODE: Loading pre-trained memory bank from '{saved_mb_file}'...")
    saved_data = joblib.load(saved_mb_file)
    memory_bank = saved_data["memory_bank"]
    grid_size = saved_data["grid_size"]
    print(f"Memory bank loaded instantly! Size: {memory_bank.shape[0]} patches (dim={memory_bank.shape[1]}).")
    nn_index = TorchNN(memory_bank, n_neighbors=N_NEIGHBORS)
else:
    pool = np.concatenate(all_patches, axis=0)
    print(f"Total normal patches collected: {pool.shape[0]} (dim={pool.shape[1]})")

    memory_bank = greedy_coreset_sampling(pool, COOSET_SAMPLING_RATIO)
    print(f"Memory bank size after coreset sampling: {memory_bank.shape[0]}")

    nn_index = TorchNN(memory_bank, n_neighbors=N_NEIGHBORS)

    joblib.dump({"memory_bank": memory_bank, "grid_size": grid_size}, saved_mb_file)
    print("Memory bank saved.")


# %% [markdown] Cell 14
# ## 6. Scoring function
# 
# Given patch features for one image, returns an image-level anomaly score and a spatial anomaly map.

# %% [code] Cell 15
def score_image(patch_features: np.ndarray, nn_index, grid_size, out_size=IMAGE_SIZE, topk=1):
    dists, _ = nn_index.kneighbors(patch_features)
    patch_scores = dists.mean(axis=1)

    h, w = grid_size
    score_map = torch.tensor(patch_scores, dtype=torch.float32).reshape(1, 1, h, w)
    upsampled = F.interpolate(score_map, size=(out_size, out_size), mode="bilinear",
                               align_corners=False).squeeze().numpy()

    top_scores = np.sort(patch_scores)[::-1][:topk]
    image_score = float(top_scores.mean())
    return image_score, upsampled


print("score_image() defined")


# %% [markdown] Cell 16
# ## 7. Determine decision threshold from validation set
# 
# Validation images are normal-only, so this gives a data-driven cutoff without peeking at test-set anomalies.

# %% [code] Cell 17
val_scores = []
for img_tensor, _ in val_ds:
    patch_feats, _ = extractor.extract(img_tensor.unsqueeze(0))
    score, _ = score_image(patch_feats.squeeze(0).numpy(), nn_index, grid_size)
    val_scores.append(score)

val_scores = np.array(val_scores)
THRESHOLD = float(val_scores.mean() + N_STD_FOR_THRESHOLD * val_scores.std())

print(f"Validation scores: mean={val_scores.mean():.3f}, std={val_scores.std():.3f}")
print(f"Chosen threshold: {THRESHOLD:.3f}")


# %% [markdown] Cell 18
# ## 8. Score the full test set
# 
# Runs every test image (good + logical + structural anomalies) through the pipeline.

# %% [code] Cell 19
image_scores, labels, subtypes, paths, anomaly_maps = [], [], [], [], []

for img_tensor, label, subtype, path in test_ds:
    patch_feats, _ = extractor.extract(img_tensor.unsqueeze(0))
    score, amap = score_image(patch_feats.squeeze(0).numpy(), nn_index, grid_size)

    image_scores.append(score)
    labels.append(label)
    subtypes.append(subtype)
    paths.append(path)
    anomaly_maps.append(amap)

image_scores = np.array(image_scores)
labels = np.array(labels)

print(f"Scored {len(image_scores)} test images.")
print(f"Score range: {image_scores.min():.3f} - {image_scores.max():.3f}")


# %% [markdown] Cell 20
# ## 9. Evaluation
# 
# Overall image-level metrics, plus a breakdown by anomaly subtype (logical vs structural).

# %% [code] Cell 21
auc = roc_auc_score(labels, image_scores)
preds = (image_scores > THRESHOLD).astype(int)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
cm = confusion_matrix(labels, preds)

print("--- Overall image-level results ---")
print(f"ROC-AUC:   {auc:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"Confusion matrix:\n{cm}")

print("\n--- Breakdown by subtype ---")
for subtype in ["good", "logical_anomalies", "structural_anomalies"]:
    idx = [i for i, s in enumerate(subtypes) if s == subtype]
    if not idx:
        continue
    sub_scores = image_scores[idx]
    sub_preds = (sub_scores > THRESHOLD).astype(int)
    if subtype == "good":
        rate = 1 - sub_preds.mean()
        print(f"{subtype:22s}: {len(idx)} images, correctly-normal rate = {rate:.3f}")
    else:
        rate = sub_preds.mean()
        print(f"{subtype:22s}: {len(idx)} images, detection rate = {rate:.3f}")


# %% [markdown] Cell 22
# ## 10. Demo visualization
# 
# Original image / anomaly heatmap / overlay, for a few examples of each subtype -- this is the artifact for your report and video.

# %% [code] Cell 23
def show_result(image_path, anomaly_map, score, threshold, label, subtype, save_path=None):
    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    decision = "ANOMALOUS" if score > threshold else "NORMAL"
    truth = "anomalous" if label == 1 else "normal"

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img)
    axes[0].set_title(f"Input\n(ground truth: {truth} / {subtype})")
    axes[0].axis("off")

    axes[1].imshow(anomaly_map, cmap="jet")
    axes[1].set_title(f"Anomaly map\nscore={score:.3f} (thr={threshold:.3f})")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(anomaly_map, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay\nPrediction: {decision}")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


demo_dir = OUTPUT_DIR / "demo"
demo_dir.mkdir(exist_ok=True)

N_EXAMPLES_PER_SUBTYPE = 2
for subtype in ["good", "logical_anomalies", "structural_anomalies"]:
    idx = [i for i, s in enumerate(subtypes) if s == subtype][:N_EXAMPLES_PER_SUBTYPE]
    for i in idx:
        show_result(
            image_path=paths[i],
            anomaly_map=anomaly_maps[i],
            score=image_scores[i],
            threshold=THRESHOLD,
            label=labels[i],
            subtype=subtype,
            save_path=demo_dir / f"{subtype}_{i}.png",
        )

print(f"Demo images saved under {demo_dir}/")


# %% [markdown] Cell 24
# ## 11. Baseline model (for comparison)
# 
# The main model above uses WideResNet50 features with a coreset-subsampled memory bank.
# As a simpler baseline (Section 2.6 of the project doc), this reruns the same pipeline with
# a smaller backbone (ResNet18) and the *full* patch pool as the memory bank (no coreset
# subsampling) -- a more naive nearest-neighbor approach.

# %% [code] Cell 25
def run_full_pipeline(backbone_name, coreset_ratio, n_std_threshold=N_STD_FOR_THRESHOLD):
    """Runs feature extraction -> memory bank -> threshold -> test scoring -> metrics
    for a given backbone/coreset configuration. Reuses train_ds/val_ds/test_ds already loaded."""
    print(f"\n=== Running pipeline: backbone={backbone_name}, coreset_ratio={coreset_ratio} ===")
    ext = PatchFeatureExtractor(backbone_name=backbone_name)   # raises a clear error if weights can't be fetched

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    all_patches = []
    grid = None
    for images, _ in loader:
        pf, grid = ext.extract(images)
        b, n, c = pf.shape
        all_patches.append(pf.reshape(b * n, c).numpy())
    pool = np.concatenate(all_patches, axis=0)

    mb = greedy_coreset_sampling(pool, coreset_ratio) if coreset_ratio < 1.0 else pool
    print(f"Memory bank size: {mb.shape[0]} (from pool of {pool.shape[0]})")
    idx = TorchNN(mb, n_neighbors=N_NEIGHBORS)

    val_scores_ = []
    for img_t, _ in val_ds:
        pf, _ = ext.extract(img_t.unsqueeze(0))
        s, _ = score_image(pf.squeeze(0).numpy(), idx, grid)
        val_scores_.append(s)
    val_scores_ = np.array(val_scores_)
    thr = float(val_scores_.mean() + n_std_threshold * val_scores_.std())

    scores_, labels_, subtypes_, paths_, amaps_ = [], [], [], [], []
    for img_t, lab, sub, path in test_ds:
        pf, _ = ext.extract(img_t.unsqueeze(0))
        s, amap = score_image(pf.squeeze(0).numpy(), idx, grid)
        scores_.append(s); labels_.append(lab); subtypes_.append(sub)
        paths_.append(path); amaps_.append(amap)
    scores_ = np.array(scores_)
    labels_ = np.array(labels_)

    auc_ = roc_auc_score(labels_, scores_)
    preds_ = (scores_ > thr).astype(int)
    p_, r_, f1_, _ = precision_recall_fscore_support(labels_, preds_, average="binary", zero_division=0)

    print(f"Threshold: {thr:.3f}")
    print(f"ROC-AUC: {auc_:.3f}  Precision: {p_:.3f}  Recall: {r_:.3f}  F1: {f1_:.3f}")

    return {
        "backbone": backbone_name, "coreset_ratio": coreset_ratio, "threshold": thr,
        "auc": auc_, "precision": p_, "recall": r_, "f1": f1_,
        "scores": scores_, "labels": labels_, "subtypes": subtypes_,
        "paths": paths_, "anomaly_maps": amaps_,
    }


try:
    baseline_results = run_full_pipeline(backbone_name="resnet18", coreset_ratio=1.0)
except Exception as exc:
    baseline_results = None
    print(f"\nBaseline skipped -- {type(exc).__name__}: {exc}")
    print("The rest of the notebook still runs; the comparison tables will simply omit the baseline row.")


# %% [markdown] Cell 26
# ## 12. Baseline vs. Main model comparison
# 
# Side-by-side image-level metrics for the two methods (Section 2.6 "Comparison").

# %% [code] Cell 27
rows = []
if baseline_results is not None:
    rows.append({
        "Model": "Baseline (ResNet18, full memory bank)",
        "ROC-AUC": baseline_results["auc"],
        "Precision": baseline_results["precision"],
        "Recall": baseline_results["recall"],
        "F1": baseline_results["f1"],
    })
else:
    print("(Baseline results unavailable -- see the message from the previous section.)")
rows.append({
    "Model": "Main (WideResNet50, coreset memory bank)",
    "ROC-AUC": auc,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
})
comparison_df = pd.DataFrame(rows)
print(comparison_df.to_string(index=False))


# %% [markdown] Cell 28
# ## 13. Pixel-level localization evaluation
# 
# Image-level metrics tell you *whether* an image is anomalous. This section evaluates
# *where* -- comparing each anomaly map (from the main model) against the dataset's
# pixel-precise ground-truth masks, giving a pixel-level ROC-AUC (Section 2.7's
# "dataset-appropriate localization metrics"). Normal ("good") test images are treated
# as having an all-zero ground-truth mask.

# %% [code] Cell 29
def load_gt_mask(image_path: str, subtype: str, out_size: int = IMAGE_SIZE):
    """Loads (and combines, if multiple) ground-truth mask(s) for one test image.
    Returns a binary (0/1) mask resized to out_size x out_size, or None if not found."""
    if subtype == "good":
        return np.zeros((out_size, out_size), dtype=np.uint8)

    p = Path(image_path)
    gt_dir = DATA_ROOT / CATEGORY / "ground_truth" / subtype / p.stem
    if not gt_dir.exists():
        return None
    mask_files = sorted(gt_dir.rglob("*.png"))
    if not mask_files:
        return None

    combined = None
    for mf in mask_files:
        m = np.array(Image.open(mf).convert("L").resize((out_size, out_size), Image.NEAREST))   # NEAREST keeps the mask binary
        m = (m > 0).astype(np.uint8)
        combined = m if combined is None else np.maximum(combined, m)
    return combined


pixel_scores_all, pixel_labels_all = [], []
missing = 0

for score, amap, label, subtype, path in zip(image_scores, anomaly_maps, labels, subtypes, paths):
    gt = load_gt_mask(path, subtype)
    if gt is None:
        missing += 1
        continue
    pixel_scores_all.append(amap.flatten())
    pixel_labels_all.append(gt.flatten())

if pixel_scores_all:
    pixel_scores_all = np.concatenate(pixel_scores_all)
    pixel_labels_all = np.concatenate(pixel_labels_all)
    if len(np.unique(pixel_labels_all)) > 1:
        pixel_auc = roc_auc_score(pixel_labels_all, pixel_scores_all)
        print(f"Pixel-level ROC-AUC: {pixel_auc:.3f}")
    else:
        print("Pixel-level AUROC undefined -- ground-truth pixels are all one class "
              "(check that ground_truth/ masks were found correctly).")
else:
    print("No ground-truth masks matched -- check the ground_truth/ folder structure "
          "under DATA_ROOT / CATEGORY.")

if missing:
    print(f"Note: {missing} test images had no matching ground-truth mask and were skipped "
          "(expected for 'good' images only counted as zero-mask above; investigate if "
          "this count is unexpectedly high for anomalous subtypes).")


# %% [markdown] Cell 30
# ## 14. Qualitative error analysis
# 
# Section 2.7 asks for qualitative examples of correct *and incorrect* predictions.
# This pulls out the test images the main model got wrong (false positives and false
# negatives) and displays them alongside their anomaly maps, so you can discuss failure
# patterns in your report -- e.g. whether logical anomalies are disproportionately
# represented among the misses.

# %% [code] Cell 31
error_dir = OUTPUT_DIR / "error_analysis"
error_dir.mkdir(exist_ok=True)

final_preds = (image_scores > THRESHOLD).astype(int)
false_positives = [i for i in range(len(labels)) if labels[i] == 0 and final_preds[i] == 1]
false_negatives = [i for i in range(len(labels)) if labels[i] == 1 and final_preds[i] == 0]

print(f"False positives (good images flagged as anomalous): {len(false_positives)}")
print(f"False negatives (anomalies missed): {len(false_negatives)}")

# Breakdown of false negatives by subtype -- this is the key diagnostic for the
# logical-vs-structural discussion in your report
if false_negatives:
    fn_subtypes = [subtypes[i] for i in false_negatives]
    print("False negatives by subtype:",
          {s: fn_subtypes.count(s) for s in set(fn_subtypes)})

def show_error_case(i, kind, save_path=None):
    show_result(
        image_path=paths[i],
        anomaly_map=anomaly_maps[i],
        score=image_scores[i],
        threshold=THRESHOLD,
        label=labels[i],
        subtype=subtypes[i],
        save_path=save_path,
    )
    print(f"[{kind}] {Path(paths[i]).name}")

print("\n--- Sample false positives ---")
for i in false_positives[:3]:
    show_error_case(i, "FALSE POSITIVE", error_dir / f"fp_{i}.png")

print("\n--- Sample false negatives ---")
for i in false_negatives[:3]:
    show_error_case(i, "FALSE NEGATIVE", error_dir / f"fn_{i}.png")

print(f"\nError-case images saved under {error_dir}/")


# %% [markdown] Cell 32
# ## 15. Threshold tuning (ROC curve + precision-recall tradeoff)
# 
# Section 7 picked a single threshold using a mean + 3*std rule on validation scores.
# This section explores the full range of possible thresholds -- plotting the ROC curve
# and a precision/recall/F1-vs-threshold curve -- so you can justify that choice (or pick
# a different operating point) in your report, per the doc's "threshold tuning" step.

# %% [code] Cell 33
from sklearn.metrics import roc_curve

fpr, tpr, roc_thresholds = roc_curve(labels, image_scores)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
plt.scatter(
    [(image_scores > THRESHOLD).astype(int)[labels == 0].mean()],
    [(image_scores > THRESHOLD).astype(int)[labels == 1].mean()],
    color="red", zorder=5, label=f"Current threshold ({THRESHOLD:.3f})"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve -- {CATEGORY}")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"roc_curve_{CATEGORY}.png", dpi=150)
plt.show()


# %% [code] Cell 34
# Precision / Recall / F1 across a sweep of candidate thresholds
candidate_thresholds = np.linspace(image_scores.min(), image_scores.max(), 50)

precisions, recalls, f1s = [], [], []
for t in candidate_thresholds:
    preds_t = (image_scores > t).astype(int)
    if preds_t.sum() == 0 or preds_t.sum() == len(preds_t):
        # avoid undefined precision/recall at degenerate thresholds
        precisions.append(np.nan); recalls.append(np.nan); f1s.append(np.nan)
        continue
    p_t, r_t, f1_t, _ = precision_recall_fscore_support(labels, preds_t, average="binary", zero_division=0)
    precisions.append(p_t); recalls.append(r_t); f1s.append(f1_t)

plt.figure(figsize=(7, 5))
plt.plot(candidate_thresholds, precisions, label="Precision")
plt.plot(candidate_thresholds, recalls, label="Recall")
plt.plot(candidate_thresholds, f1s, label="F1-score")
plt.axvline(THRESHOLD, color="red", linestyle="--", label=f"Current threshold ({THRESHOLD:.3f})")
plt.xlabel("Anomaly score threshold")
plt.ylabel("Metric value")
plt.title(f"Threshold sweep -- {CATEGORY}")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f"threshold_sweep_{CATEGORY}.png", dpi=150)
plt.show()

if not np.all(np.isnan(f1s)):
    best_f1_idx = int(np.nanargmax(f1s))
    print(f"Threshold that maximizes F1 on this test set: {candidate_thresholds[best_f1_idx]:.3f} "
          f"(F1={f1s[best_f1_idx]:.3f}) vs. current validation-derived threshold {THRESHOLD:.3f} "
          f"(F1={f1:.3f})")
print("Note: picking a threshold by maximizing F1 on the test set is 'peeking' at test "
      "labels and shouldn't be used as your final operating threshold -- it's shown here "
      "only to characterize the tradeoff curve for your report.")


# %% [markdown] Cell 35
# ## 16. Standalone inference interface
# 
# A simple function that takes the path to *any* image (not necessarily one from
# `test_ds`) and returns a decision, score, and anomaly visualization -- the
# "simple inference/demo interface" step from the project's day plan. This is what
# you'd wire up to a Streamlit app or CLI tool for the final demo/video.

# %% [code] Cell 36
def predict_on_image(image, show: bool = True, save_path=None):
    """
    Runs the full trained pipeline (main model: memory_bank / nn_index / grid_size /
    THRESHOLD, all built earlier in this notebook) on a single arbitrary image.

    `image` may be a file path or an already-loaded PIL image (used by the web app).
    Returns a dict with the decision, score, threshold, and the anomaly map array.
    """
    if isinstance(image, (str, os.PathLike)):
        image_path = str(image)
        img = Image.open(image_path).convert("RGB")
    else:
        image_path = "uploaded image"
        img = image.convert("RGB")
    img_tensor = get_transform()(img).unsqueeze(0)

    with _INFER_LOCK:                      # the extractor's hook state is shared, so one call at a time
        patch_feats, _ = extractor.extract(img_tensor)
        score, amap = score_image(patch_feats.squeeze(0).numpy(), nn_index, grid_size)
    decision = "ANOMALOUS" if score > THRESHOLD else "NORMAL"

    result = {
        "image_path": image_path,
        "score": score,
        "threshold": THRESHOLD,
        "decision": decision,
        "anomaly_map": amap,
    }

    if show:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        display_img = img.resize((IMAGE_SIZE, IMAGE_SIZE))

        axes[0].imshow(display_img)
        axes[0].set_title(f"Input\n{Path(image_path).name if image_path != 'uploaded image' else image_path}")
        axes[0].axis("off")

        axes[1].imshow(amap, cmap="jet")
        axes[1].set_title(f"Anomaly map\nscore={score:.3f} (thr={THRESHOLD:.3f})")
        axes[1].axis("off")

        axes[2].imshow(display_img)
        axes[2].imshow(amap, cmap="jet", alpha=0.5)
        axes[2].set_title(f"Overlay\nPrediction: {decision}")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
        plt.close(fig)

    return result


# Example usage -- try it on any test image, or point it at a completely new file:
_example_path = test_ds.samples[0][0]   # first test image, as a demo
_ = predict_on_image(_example_path, save_path=OUTPUT_DIR / "single_inference_example.png")


# %% [markdown] Cell 37
# ## 17. Qualitative localization comparison (predicted vs. ground truth)
# 
# Section 14's error analysis looked at *decision*-level correctness (was the image
# flagged anomalous or not). This section evaluates *localization*-level correctness --
# whether the anomaly map actually highlights the same region as the real defect --
# by overlaying the predicted anomaly map directly against the ground-truth mask, per
# image. This is what Section 2.7 means by "qualitative examples of correct and
# incorrect localization."
# 
# For each anomalous test image with a ground-truth mask, this computes a simple
# per-image localization overlap score (Intersection-over-Union between the
# thresholded anomaly map and the ground-truth mask), then shows the best- and
# worst-localized examples side by side: original image / ground truth / prediction.

# %% [code] Cell 38
def localization_iou(anomaly_map: np.ndarray, gt_mask: np.ndarray, top_fraction: float = 0.05):
    """
    Thresholds the anomaly map to its top `top_fraction` highest-scoring pixels
    (treating those as the 'predicted defect region'), then computes IoU against
    the binary ground-truth mask. A simple, interpretable localization-quality score.
    """
    flat = anomaly_map.flatten()
    n_top = max(1, int(len(flat) * top_fraction))
    cutoff = np.partition(flat, -n_top)[-n_top]
    pred_mask = (anomaly_map >= cutoff).astype(np.uint8)

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou, pred_mask


localization_records = []  # (index, iou, gt_mask, pred_mask)

for i, (amap, label, subtype, path) in enumerate(zip(anomaly_maps, labels, subtypes, paths)):
    if subtype == "good":
        continue
    gt = load_gt_mask(path, subtype)
    if gt is None or gt.sum() == 0:
        continue
    iou, pred_mask = localization_iou(amap, gt)
    localization_records.append({"index": i, "iou": iou, "gt_mask": gt, "pred_mask": pred_mask})

if localization_records:
    ious = [r["iou"] for r in localization_records]
    print(f"Localization IoU across {len(localization_records)} anomalous images with masks: "
          f"mean={np.mean(ious):.3f}, min={np.min(ious):.3f}, max={np.max(ious):.3f}")
else:
    print("No anomalous images with usable ground-truth masks were found -- "
          "check the ground_truth/ folder structure under DATA_ROOT / CATEGORY.")


# %% [code] Cell 39
def show_localization_comparison(record, save_path=None):
    i = record["index"]
    img = Image.open(paths[i]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img)
    axes[0].set_title(f"Input\n{Path(paths[i]).name}\n({subtypes[i]})")
    axes[0].axis("off")

    axes[1].imshow(record["gt_mask"], cmap="gray")
    axes[1].set_title("Ground truth\ndefect region")
    axes[1].axis("off")

    axes[2].imshow(record["pred_mask"], cmap="gray")
    axes[2].set_title(f"Predicted region\n(top 5% anomaly score)")
    axes[2].axis("off")

    axes[3].imshow(img)
    axes[3].imshow(record["gt_mask"], cmap="Greens", alpha=0.4)
    axes[3].imshow(record["pred_mask"], cmap="Reds", alpha=0.4)
    axes[3].set_title(f"Overlay\nIoU = {record['iou']:.3f}\n(green=truth, red=predicted)")
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


if localization_records:
    localization_records_sorted = sorted(localization_records, key=lambda r: r["iou"], reverse=True)
    localization_dir = OUTPUT_DIR / "localization_comparison"
    localization_dir.mkdir(exist_ok=True)

    print("--- Best-localized examples (correct localization) ---")
    for r in localization_records_sorted[:2]:
        show_localization_comparison(r, save_path=localization_dir / f"best_{r['index']}.png")

    print("--- Worst-localized examples (incorrect localization) ---")
    for r in localization_records_sorted[-2:]:
        show_localization_comparison(r, save_path=localization_dir / f"worst_{r['index']}.png")

    print(f"Saved localization comparison images under {localization_dir}/")


# %% [markdown] Cell 40
# ## 18. Multi-category validation
# 
# Everything above was run on a single category (Juice Bottle). A single-category result
# only demonstrates a proof-of-concept, not a validated system -- this section reruns the
# full main-model pipeline (data loading, feature extraction, memory bank, threshold,
# scoring, evaluation) independently for every MVTec LOCO AD category, and reports how
# well the approach generalizes across different object types.

# %% [code] Cell 41
def run_category_pipeline(category: str, backbone_name: str = "wide_resnet50_2",
                           coreset_ratio: float = COOSET_SAMPLING_RATIO):
    """
    Runs the complete main-model pipeline end-to-end for a single category:
    dataset loading -> feature extraction -> memory bank -> threshold -> test
    scoring -> image-level and subtype-level metrics. Fully self-contained --
    does not depend on any global train_ds/val_ds/test_ds from earlier sections.
    """
    print(f"\n{'='*60}\nCategory: {category}\n{'='*60}")

    cat_train_ds = NormalImageDataset(category, "train")
    cat_val_ds = NormalImageDataset(category, "validation")
    cat_test_ds = TestImageDataset(category)

    ext = PatchFeatureExtractor(backbone_name=backbone_name)
    n_images_note = f"{len(cat_train_ds)} train / {len(cat_val_ds)} val / {len(cat_test_ds)} test"
    print(f"Loaded {n_images_note}")

    loader = DataLoader(cat_train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    all_patches = []
    grid = None
    for images, _ in loader:
        pf, grid = ext.extract(images)
        b, n, c = pf.shape
        all_patches.append(pf.reshape(b * n, c).numpy())
    pool = np.concatenate(all_patches, axis=0)
    mb = greedy_coreset_sampling(pool, coreset_ratio) if coreset_ratio < 1.0 else pool
    idx = TorchNN(mb, n_neighbors=N_NEIGHBORS)

    val_scores_ = []
    for img_t, _ in cat_val_ds:
        pf, _ = ext.extract(img_t.unsqueeze(0))
        s, _ = score_image(pf.squeeze(0).numpy(), idx, grid)
        val_scores_.append(s)
    val_scores_ = np.array(val_scores_)
    thr = float(val_scores_.mean() + N_STD_FOR_THRESHOLD * val_scores_.std())

    scores_, labels_, subtypes_ = [], [], []
    for img_t, lab, sub, path in cat_test_ds:
        pf, _ = ext.extract(img_t.unsqueeze(0))
        s, _ = score_image(pf.squeeze(0).numpy(), idx, grid)
        scores_.append(s); labels_.append(lab); subtypes_.append(sub)
    scores_ = np.array(scores_)
    labels_ = np.array(labels_)

    auc_ = roc_auc_score(labels_, scores_)
    preds_ = (scores_ > thr).astype(int)
    p_, r_, f1_, _ = precision_recall_fscore_support(labels_, preds_, average="binary", zero_division=0)

    subtype_detection = {}
    for subtype in ["good", "logical_anomalies", "structural_anomalies"]:
        sub_idx = [i for i, s in enumerate(subtypes_) if s == subtype]
        if not sub_idx:
            continue
        sub_preds = preds_[sub_idx]
        rate = (1 - sub_preds.mean()) if subtype == "good" else sub_preds.mean()
        subtype_detection[subtype] = rate

    print(f"Train/Val/Test sizes: {len(cat_train_ds)}/{len(cat_val_ds)}/{len(cat_test_ds)}")
    print(f"ROC-AUC: {auc_:.3f}  Precision: {p_:.3f}  Recall: {r_:.3f}  F1: {f1_:.3f}")
    print(f"Subtype detection rates: {subtype_detection}")

    return {
        "category": category, "n_train": len(cat_train_ds), "n_val": len(cat_val_ds),
        "n_test": len(cat_test_ds), "threshold": thr, "auc": auc_, "precision": p_,
        "recall": r_, "f1": f1_, "subtype_detection": subtype_detection,
    }


ALL_CATEGORIES = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]

multi_category_results = []
for cat in ALL_CATEGORIES:
    cat_path = DATA_ROOT / cat
    if not cat_path.exists():
        print(f"Skipping {cat} -- folder not found at {cat_path}")
        continue
    try:
        result = run_category_pipeline(cat)
        multi_category_results.append(result)
    except Exception as e:
        print(f"Skipping {cat} due to error: {type(e).__name__}: {e}")


# %% [code] Cell 42
import pandas as pd

summary_rows = []
for r in multi_category_results:
    row = {
        "Category": r["category"],
        "Train/Val/Test": f"{r['n_train']}/{r['n_val']}/{r['n_test']}",
        "ROC-AUC": round(r["auc"], 3),
        "Precision": round(r["precision"], 3),
        "Recall": round(r["recall"], 3),
        "F1": round(r["f1"], 3),
        "Logical detection": round(r["subtype_detection"].get("logical_anomalies", float("nan")), 3),
        "Structural detection": round(r["subtype_detection"].get("structural_anomalies", float("nan")), 3),
    }
    summary_rows.append(row)

multi_category_df = pd.DataFrame(summary_rows)
if multi_category_df.empty:
    print('No category finished -- check the messages above (dataset folders or weight download).')
else:
    print(multi_category_df.to_string(index=False))

if len(multi_category_df) > 1:
    print(f"\nMean ROC-AUC across categories: {multi_category_df['ROC-AUC'].mean():.3f}")
    print(f"Mean logical-anomaly detection: {multi_category_df['Logical detection'].mean():.3f}")
    print(f"Mean structural-anomaly detection: {multi_category_df['Structural detection'].mean():.3f}")
    print("\nThis is the key evidence for your report's 'generalization' claim -- cite the "
          "per-category spread, not just the mean, since consistency across categories matters "
          "as much as the average.")


# %% [markdown] Cell 43
# ## 19. Trained statistical model: per-location Gaussian anomaly detector (PaDiM-style)
# 
# The main model (Sections 5-9) stores raw normal patch features and scores new patches
# by nearest-neighbor distance -- this is a non-parametric, "instance memorization"
# approach. This section fits an explicit statistical model instead: for every patch
# *location* in the feature grid, it estimates a mean vector and a variance vector from
# the training images (i.e. it learns a per-location Gaussian distribution of what
# "normal" looks like there), then scores test patches by how many standard deviations
# they fall from that learned distribution (a diagonal Mahalanobis distance).
# 
# This directly answers "what machine learning did you build": these mean/variance
# parameters are *fitted* from data (maximum-likelihood estimation of a Gaussian), which
# is a genuine trained model, as distinct from the memory-bank/kNN approach which performs
# no parameter estimation at all.

# %% [code] Cell 44
def fit_gaussian_model(dataset, extractor, eps: float = 1e-6):
    """
    Fits a per-patch-location diagonal Gaussian: for each of the h*w grid positions,
    estimates mean and variance across all training images at that position.
    Returns: mean (h*w, C), var (h*w, C), grid_size
    """
    # Streaming sums in float64: the naive version stacks every training patch (~2 GB for 335 images)
    # plus several temporary copies in .var(), which can exhaust RAM; this needs only (N, C) arrays.
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    total, total_sq, count, grid = None, None, 0, None
    for images, _ in loader:
        pf, grid = extractor.extract(images)         # (B, N, C) float32 tensor on CPU
        pf = pf.double()
        s, sq = pf.sum(dim=0), (pf ** 2).sum(dim=0)
        total = s if total is None else total + s
        total_sq = sq if total_sq is None else total_sq + sq
        count += pf.shape[0]

    mean_t = total / count                                       # (N, C)
    var_t = (total_sq / count - mean_t ** 2).clamp_min(0.0)       # population variance, never negative
    mean = mean_t.float().numpy()
    var = var_t.float().numpy() + eps                             # regularized to avoid division by zero
    return mean, var, grid


def gaussian_anomaly_score(patch_features: np.ndarray, mean: np.ndarray, var: np.ndarray,
                            grid_size, out_size: int = IMAGE_SIZE, topk: int = 1):
    """
    Diagonal Mahalanobis distance per patch: sum over channels of (x-mu)^2 / var.
    Same image-level aggregation (top-k mean) and upsampling as the kNN scorer,
    so results are directly comparable.
    """
    sq_z = ((patch_features - mean) ** 2) / var          # (N, C)
    patch_scores = sq_z.sum(axis=1)                       # (N,)

    h, w = grid_size
    score_map = torch.tensor(patch_scores, dtype=torch.float32).reshape(1, 1, h, w)
    upsampled = F.interpolate(score_map, size=(out_size, out_size), mode="bilinear",
                               align_corners=False).squeeze().numpy()

    top_scores = np.sort(patch_scores)[::-1][:topk]
    image_score = float(top_scores.mean())
    return image_score, upsampled


print("Fitting Gaussian model on training data...")
gauss_mean, gauss_var, gauss_grid = fit_gaussian_model(train_ds, extractor)
print(f"Fitted Gaussian parameters: mean {gauss_mean.shape}, var {gauss_var.shape}")


# %% [code] Cell 45
# Threshold from validation (same data-driven approach as the kNN model)
gauss_val_scores = []
for img_t, _ in val_ds:
    pf, _ = extractor.extract(img_t.unsqueeze(0))
    s, _ = gaussian_anomaly_score(pf.squeeze(0).numpy(), gauss_mean, gauss_var, gauss_grid)
    gauss_val_scores.append(s)
gauss_val_scores = np.array(gauss_val_scores)
gauss_threshold = float(gauss_val_scores.mean() + N_STD_FOR_THRESHOLD * gauss_val_scores.std())

# Score the test set
gauss_scores, gauss_labels_check = [], []
for img_t, lab, sub, path in test_ds:
    pf, _ = extractor.extract(img_t.unsqueeze(0))
    s, _ = gaussian_anomaly_score(pf.squeeze(0).numpy(), gauss_mean, gauss_var, gauss_grid)
    gauss_scores.append(s)
    gauss_labels_check.append(lab)
gauss_scores = np.array(gauss_scores)
gauss_labels_check = np.array(gauss_labels_check)

gauss_auc = roc_auc_score(gauss_labels_check, gauss_scores)
gauss_preds = (gauss_scores > gauss_threshold).astype(int)
gauss_p, gauss_r, gauss_f1, _ = precision_recall_fscore_support(gauss_labels_check, gauss_preds, average="binary", zero_division=0)

print(f"Gaussian model -- Threshold: {gauss_threshold:.3f}")
print(f"Gaussian model -- ROC-AUC: {gauss_auc:.3f}  Precision: {gauss_p:.3f}  Recall: {gauss_r:.3f}  F1: {gauss_f1:.3f}")

print("\n--- Three-way method comparison ---")
method_rows = []
if baseline_results is not None:
    method_rows.append({"Method": "Baseline (ResNet18, full memory bank, kNN)", "ROC-AUC": baseline_results["auc"],
                        "Precision": baseline_results["precision"], "Recall": baseline_results["recall"],
                        "F1": baseline_results["f1"]})
method_rows.append({"Method": "Main (WideResNet50, coreset memory bank, kNN)", "ROC-AUC": auc,
                    "Precision": precision, "Recall": recall, "F1": f1})
method_rows.append({"Method": "Gaussian (WideResNet50, fitted per-location Gaussian)", "ROC-AUC": gauss_auc,
                    "Precision": gauss_p, "Recall": gauss_r, "F1": gauss_f1})
three_way_df = pd.DataFrame(method_rows)
print(three_way_df.to_string(index=False))


# %% [markdown] Cell 46
# ## 20. Precision-recall operating point trade-off
# 
# A single ROC-AUC number (0.896) hides an important detail: at the current
# validation-derived threshold, recall is only ~0.47 -- more than half of all real
# anomalies are missed, even though precision is very high (~0.99, almost no false
# alarms). This section makes that trade-off explicit by evaluating several candidate
# thresholds side by side, so the choice of operating point can be justified rather than
# presented as a single unexplained number.

# %% [code] Cell 47
# Build candidate thresholds from validation-score statistics
val_mean, val_std = val_scores.mean(), val_scores.std()
candidate_points = {
    "Very conservative (mean + 4*std)": val_mean + 4 * val_std,
    "Current / conservative (mean + 3*std)": val_mean + 3 * val_std,
    "Balanced (mean + 2*std)": val_mean + 2 * val_std,
    "High recall (mean + 1*std)": val_mean + 1 * val_std,
}

rows = []
for label_name, t in candidate_points.items():
    preds_t = (image_scores > t).astype(int)
    p_t, r_t, f1_t, _ = precision_recall_fscore_support(labels, preds_t, average="binary", zero_division=0)
    fp = int(((preds_t == 1) & (labels == 0)).sum())
    fn = int(((preds_t == 0) & (labels == 1)).sum())
    rows.append({
        "Operating point": label_name, "Threshold": round(t, 3),
        "Precision": round(p_t, 3), "Recall": round(r_t, 3), "F1": round(f1_t, 3),
        "False Positives": fp, "False Negatives": fn,
    })

operating_point_df = pd.DataFrame(rows)
print(operating_point_df.to_string(index=False))

print("""
Interpretation for the report:
- The current threshold (mean + 3*std) is a CONSERVATIVE operating point: it minimizes
  false alarms (good for avoiding unnecessary line stoppages) at the cost of missing
  many real defects.
- A quality-control system where missed defects are costly (e.g. safety-critical
  products) would justify moving to the 'High recall' or 'Balanced' operating point
  instead, accepting more false alarms in exchange for catching more real anomalies.
- This is a deployment decision, not a modeling one -- the same trained model supports
  any of these operating points simply by changing the threshold, so this table should
  be presented as a tunable design choice in the report rather than a limitation of the
  method itself.
""")


# %% [markdown] Cell 48
# ## 21. Notes for the report
# 
# - **Multi-category generalization**: Section 18 runs the full pipeline independently
#   across all 5 categories and reports per-category metrics plus mean detection rates --
#   use this table (not the single Juice Bottle result) as your headline generalization
#   evidence, and discuss the spread across categories, not just the average.
# - **What ML was actually built**: Section 19 fits an explicit statistical model (a
#   per-patch-location Gaussian, mean/variance estimated from training data) as a second,
#   genuinely *trained* anomaly scorer, alongside the memory-bank/kNN approach. Use this to
#   answer directly: "we estimated Gaussian distribution parameters via maximum likelihood
#   from the pretrained feature space of normal images" -- a real modeling contribution,
#   not just off-the-shelf inference.
# - **Precision-recall trade-off**: Section 20 makes explicit that the headline ROC-AUC
#   (0.896) sits behind a threshold choice with real consequences (recall ~0.47 at the
#   current conservative threshold). Present this as a tunable deployment decision, with
#   the operating-point table as evidence you understand the trade-off rather than just
#   reporting one number.
# - **Baseline vs main model**: Sections 11-12 provide the ResNet18-vs-WideResNet50
#   comparison; Section 19 adds a third method (Gaussian) for a fuller comparison.
# - **Pixel-level localization metrics**: Section 13 computes pixel-level ROC-AUC against
#   the ground-truth masks. Section 17 adds qualitative best/worst localization examples
#   with an IoU score.
# - **Note on localization metric choice**: this notebook uses pixel-level ROC-AUC and a
#   simple top-k IoU, which are standard and interpretable. The original MVTec LOCO AD
#   paper's official benchmark uses a different metric (saturated per-region overlap,
#   sPRO) -- worth mentioning this choice explicitly in your report's methodology section.
# - **Error analysis (decision-level)**: Section 14 pulls out false-positive/false-negative
#   detection examples with images saved under `outputs/error_analysis/`.
# - **Standalone inference**: Section 16's `predict_on_image(path)` runs the full trained
#   pipeline on any single image -- the Gradio web app in Section 22 calls it directly,
#   and it is what you would show in your demo video.
# - **Logical vs structural gap**: check the subtype breakdowns across categories in
#   Section 18, and the IoU scores in Section 17 -- feature-distance methods are typically
#   weaker on logical anomalies than structural ones. This pattern showing up consistently
#   across multiple categories (not just Juice Bottle) would be strong evidence for your
#   discussion section.

# %% [markdown] Cell 49
# ## 22. Web application — manufacturing image inspection (Gradio)
# 
# This interface **reuses `predict_on_image()`** and its fitted `extractor`, `nn_index`, `grid_size` and `THRESHOLD`; it does not train a separate model. Run the preceding model-preparation cells (at least Sections 0–9 and 16) before launching.
# 
# * If Gradio is missing, run `%pip install gradio` in a new cell, restart the kernel, then rerun the notebook.
# * The app is configured for the selected `CATEGORY` (initially `juice_bottle`) — upload images of that product only.
# * The server starts in the background and opens in your browser (the address is printed below the launch cell). Stop it with `inspection_app.close()`.
# * The heatmap is a relative visualization, not a calibrated defect probability.

# %% [code] Cell 50
# If needed, run this in a separate notebook cell: %pip install gradio
try:
    import gradio as gr
except ImportError as exc:
    raise ImportError("Install Gradio in this kernel with `%pip install gradio`, then restart the kernel and rerun.") from exc

from PIL import Image as PILImage

_required = ["predict_on_image", "extractor", "nn_index", "grid_size", "THRESHOLD", "CATEGORY"]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise RuntimeError("Run the model preparation and single-image inference cells first. Missing: " + ", ".join(_missing))

_BILINEAR = getattr(getattr(PILImage, "Resampling", PILImage), "BILINEAR")


def inspect_uploaded_image(uploaded_image):
    """Use the notebook's tested single-image inference API and return Gradio outputs."""
    if uploaded_image is None:
        raise gr.Error("Please upload a product image first.")

    try:
        original = uploaded_image.convert("RGB")
        result = predict_on_image(original, show=False)      # accepts a PIL image directly -- no temp file needed

        decision = str(result["decision"])
        score = float(result["score"])
        threshold = float(result["threshold"])
        amap = np.asarray(result["anomaly_map"], dtype=np.float32)
        if amap.ndim != 2 or amap.size == 0 or not np.isfinite(amap).all():
            raise ValueError("The anomaly map must be a finite, non-empty 2D array.")

        low, high = float(amap.min()), float(amap.max())
        normalized = np.clip((amap - low) / (high - low), 0, 1) if high > low else np.zeros_like(amap)
        gray = PILImage.fromarray((normalized * 255).astype(np.uint8)).resize(original.size, _BILINEAR)
        heat = PILImage.fromarray((plt.get_cmap("jet")(np.asarray(gray) / 255.0)[..., :3] * 255).astype(np.uint8))
        overlay = PILImage.blend(original, heat, alpha=0.40)

        summary = (f"### Inspection result: **{decision}**\n"
                   f"**Category:** `{CATEGORY}`  \n"
                   f"**Anomaly score:** `{score:.4f}`  \n"
                   f"**Decision threshold:** `{threshold:.4f}`  \n\n"
                   "Scores measure deviation from defect-free reference features. "
                   "Heatmap colors show relative spatial anomalies, not defect probabilities.")
        return summary, original, heat, overlay
    except Exception as exc:
        raise gr.Error(f"Inspection failed: {type(exc).__name__}: {exc}") from exc


print(f"Gradio inference ready: category={CATEGORY}, threshold={float(THRESHOLD):.4f}")


# %% [code] Cell 51
# Close any earlier Gradio instance before re-running this cell (frees the port).
if "inspection_app" in globals():
    try:
        inspection_app.close()
    except Exception:
        pass


def _upload_widget():
    try:                                                    # Gradio 4 / 5
        return gr.Image(label="Upload product image", type="pil", sources=["upload"])
    except TypeError:                                       # Gradio 3.x
        return gr.Image(label="Upload product image", type="pil", source="upload")


with gr.Blocks(title="Manufacturing Visual Anomaly Inspector") as inspection_app:
    gr.Markdown("# Manufacturing Visual Anomaly Inspector")
    gr.Markdown(
        f"**Model category:** `{CATEGORY}` | Upload an image from this product category. "
        "The detector compares its visual features with defect-free reference images. "
        "It does not explicitly enforce component-count or assembly rules."
    )
    with gr.Row():
        upload = _upload_widget()
        with gr.Column():
            run_button = gr.Button("Inspect image", variant="primary")
            result_text = gr.Markdown("Upload an image and click **Inspect image**.")
    with gr.Row():
        original_out = gr.Image(label="Original image", interactive=False)
        heatmap_out = gr.Image(label="Anomaly heatmap", interactive=False)
        overlay_out = gr.Image(label="Suspected defect overlay", interactive=False)
    run_button.click(
        fn=inspect_uploaded_image,
        inputs=upload,
        outputs=[result_text, original_out, heatmap_out, overlay_out],
    )

# prevent_thread_lock=True returns control to the notebook (debug=True would block every later cell);
# inline=False + inbrowser=True because the inline iframe often stays blank in VS Code notebooks.
inspection_app.launch(server_name="127.0.0.1", share=False, inline=False, inbrowser=True,
                      prevent_thread_lock=True, show_error=True)
_url = getattr(inspection_app, "local_url", None)
print(f"Web app running at: {_url or 'http://127.0.0.1:7860'}   (stop it with inspection_app.close())")

