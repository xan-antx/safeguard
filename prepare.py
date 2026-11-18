"""
prepare.py - ONE command that builds everything the demo needs.

    python prepare.py            full run (downloads data, trains ~5-10 min on CPU, evaluates)
    python prepare.py --quick    tiny subset + 1 epoch, to check the pipeline works (~2 min);
                                 writes to outputs/quick/ so it never replaces the real model
    python prepare.py --retrain  train again even if outputs/model.pt already exists

Steps:
  1. Download GTSRB (43 sign classes) and CIFAR-10 into ./data
  2. Build the training set: GTSRB train + 8,000 CIFAR-10 images as class 43 "not a sign"
     (outlier exposure, Hendrycks et al. 2019). CLEAN images only, light augmentation.
  3. Train the small CNN, save outputs/model.pt
  4. Calibrate tau_conf and tau_agree on a validation mix of clean + corrupted images,
     save outputs/config.json
  5. Evaluate: clean accuracy, confidence-vs-accuracy under fog/noise, risk-coverage
     ("price of safety") curve, CIFAR-10 SAFE STOP rate -> outputs/results.json + 2 charts
  6. Find the three demo presets -> outputs/presets.json + outputs/demo_images/
  7. Print a summary with timings
"""

import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

import matplotlib
matplotlib.use("Agg")                  # draw charts to files, no window needed
import matplotlib.pyplot as plt

import safeguard as sg

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

# ---- settings (full run / --quick run) --------------------------------------------------
FULL = dict(epochs=12, gtsrb_train=None, cifar_train=8000, test_subset=2000, mix_val=1500,
            mix_test=2000, cifar_test=10000, sev_step=5)
QUICK = dict(epochs=1, gtsrb_train=3000, cifar_train=800, test_subset=300, mix_val=200,
             mix_test=200, cifar_test=300, sev_step=10)
TARGET_ACCURACY = 0.97     # accepted answers must be at least 97% correct
BATCH_SIZE = 128
LEARNING_RATE = 2e-3

# ---- chart colours (match the app and the slides) ---------------------------------------
BG, FG = "#0E1217", "#F2EEE6"
AMBER, BLUE, RED, GREY = "#F2A900", "#5AA9E6", "#E5484D", "#5B6270"

TIMINGS = {}


@contextmanager
def timed(name):
    """Print a header, run the block, remember how long it took."""
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    yield
    TIMINGS[name] = time.perf_counter() - t0
    print(f"    ({TIMINGS[name]:.1f} s)", flush=True)


# =========================================================================================
# 1. Data
# =========================================================================================

# The archives torchvision expects. We download them ourselves first, because torchvision's
# downloader has no timeout (a stalled server hangs forever) and restarts from zero.
# Ours gives up on a silent connection after 30 s and RESUMES where it stopped.
ARCHIVES = [
    # (url, file in ./data, folder that exists once torchvision has extracted it)
    ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB-Training_fixed.zip",
     "gtsrb/GTSRB-Training_fixed.zip", "gtsrb/GTSRB/Training"),
    ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
     "gtsrb/GTSRB_Final_Test_Images.zip", "gtsrb/GTSRB/Final_Test"),
    ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",
     "gtsrb/GTSRB_Final_Test_GT.zip", "gtsrb/GT-final_test.csv"),
    ("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
     "cifar-10-python.tar.gz", "cifar-10-batches-py"),
]


def download_resumable(url, dest, retries=20, timeout=30):
    """Download url to dest. If dest is partly there, ask the server for the rest only."""
    import urllib.error
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        timeout=timeout) as r:
                if have and r.status != 206:
                    raise RuntimeError(f"server cannot resume {dest.name}; please delete the "
                                       f"partial file yourself and run again")
                total = have + int(r.headers.get("Content-Length", 0))
                with open(dest, "ab") as f:                 # "ab" = append, never overwrite
                    last_print = 0
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                        have += len(chunk)
                        if total and have - last_print > total / 20:
                            print(f"    {dest.name}: {have / 1e6:.0f} / {total / 1e6:.0f} MB",
                                  flush=True)
                            last_print = have
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:                                # we already have every byte
                return
            print(f"    download problem ({e}), retry {attempt}/{retries}...", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            print(f"    download stalled ({e}), resuming, retry {attempt}/{retries}...", flush=True)
        time.sleep(2)
    raise RuntimeError(f"could not download {url} - check the internet connection and rerun")


def download_all():
    for url, file, extracted in ARCHIVES:
        if not (DATA_DIR / extracted).exists():
            download_resumable(url, DATA_DIR / file)

def load_gtsrb(split):
    """
    Returns images (N, 64, 64, 3) uint8, labels (N,), tracks (N,).
    GTSRB training images come in "tracks": ~30 photos of the SAME physical sign. We keep
    the track id so the validation set can hold out whole tracks (otherwise validation would
    contain near-copies of training images and look far too good).
    The resized images are cached in ./data so later runs start faster.
    """
    cache = DATA_DIR / f"gtsrb_{split}_64.npz"
    if cache.exists():
        d = np.load(cache)
        return d["images"], d["labels"], d["tracks"]
    ds = torchvision.datasets.GTSRB(root=str(DATA_DIR), split=split, download=True)
    samples = ds._samples          # list of (file path, class id) built by torchvision
    n = len(samples)
    images = np.zeros((n, sg.DISPLAY_SIZE, sg.DISPLAY_SIZE, 3), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    tracks = np.full(n, -1, dtype=np.int64)
    for i, (path, label) in enumerate(samples):
        with Image.open(path) as im:
            images[i] = sg.display_uint8(im)
        labels[i] = label
        if split == "train":                      # file name is <track>_<image>.ppm
            tracks[i] = int(Path(path).name.split("_")[0])
    np.savez(cache, images=images, labels=labels, tracks=tracks)
    return images, labels, tracks


def load_cifar(train, indices=None):
    """CIFAR-10 images resized to the same 64x64 display size. Returns images, labels."""
    ds = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=train, download=True)
    data, targets = ds.data, np.array(ds.targets)
    if indices is not None:
        data, targets = data[indices], targets[indices]
    images = np.stack([sg.display_uint8(Image.fromarray(a)) for a in data])
    return images, targets


def to_tensor_batches(images_u8, chunk=2048):
    """uint8 display images -> (N, 3, 32, 32) float tensor, using the same resize as the app."""
    parts = [sg.to_model_input(images_u8[i:i + chunk].astype(np.float32) / 255.0)
             for i in range(0, len(images_u8), chunk)]
    return torch.cat(parts)


# =========================================================================================
# 2. Training
# =========================================================================================

def augment(x):
    """
    Light, CLEAN augmentation only: random rotation (+-10 degrees) and brightness (x0.75-1.25).
    No fog, blur or noise - the model must never see those during training.
    """
    b = x.shape[0]
    angle = (torch.rand(b) * 2 - 1) * math.radians(10)
    cos, sin = torch.cos(angle), torch.sin(angle)
    zeros = torch.zeros(b)
    theta = torch.stack([torch.stack([cos, -sin, zeros], 1),
                         torch.stack([sin, cos, zeros], 1)], 1)          # (b, 2, 3) rotation
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="border", align_corners=False)
    brightness = 0.75 + 0.5 * torch.rand(b, 1, 1, 1)
    return (x * brightness).clamp(0, 1)


@torch.no_grad()
def accuracy_off(model, x, y, n_classes=sg.N_CLASSES, chunk=1024):
    """Plain accuracy with dropout off. n_classes=43 means 'SafeGuard OFF' (sign outputs only)."""
    model.eval()
    correct = 0
    for i in range(0, len(x), chunk):
        logits = model(x[i:i + chunk])[:, :n_classes]
        correct += (logits.argmax(1) == y[i:i + chunk]).sum().item()
    return correct / len(x)


def train_model(x_train, y_train, x_val, y_val, epochs):
    torch.manual_seed(0)
    model = sg.SignNet()
    # The sampler hands the dataset a whole list of 128 indices at once, so each batch is one
    # fast tensor slice instead of 128 separate look-ups (this halves the training time).
    sampler = BatchSampler(RandomSampler(x_train, generator=torch.Generator().manual_seed(0)),
                           batch_size=BATCH_SIZE, drop_last=True)
    loader = DataLoader(TensorDataset(x_train, y_train), sampler=sampler, batch_size=None,
                        num_workers=0)                          # num_workers=0: safe on Windows
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # one-cycle: learning rate warms up, then slowly decays -> good accuracy in few epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=LEARNING_RATE,
                                                    total_steps=epochs * len(loader))
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        t0 = time.perf_counter()
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(augment(xb)), yb)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            scheduler.step()
            total_loss += loss.item()
        val_acc = accuracy_off(model, x_val, y_val)
        print(f"    epoch {epoch + 1:2d}/{epochs}  loss {total_loss / len(loader):.3f}  "
              f"val acc {val_acc:.3%}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    model.eval()
    return model


# =========================================================================================
# 3. Evaluation helpers
# =========================================================================================

@torch.no_grad()
def off_predict_many(model, images_float, chunk=1024):
    preds, confs = [], []
    for i in range(0, len(images_float), chunk):
        p, c, _ = sg.predict_off(model, sg.to_model_input(images_float[i:i + chunk]))
        preds.append(p)
        confs.append(c)
    return np.concatenate(preds), np.concatenate(confs)


def corrupt_many(images_u8, kind, severity):
    return np.stack([sg.corrupt(im.astype(np.float32) / 255.0, kind, severity) for im in images_u8])


def mc_one(model, img):
    """MC dropout stats for one display image - exactly what the app computes."""
    return sg.mc_summary(sg.mc_dropout_probs(model, sg.to_model_input(img)))


def evaluate_mix(model, images_u8, labels, seed, name):
    """
    A mix of clean and corrupted images: every image appears clean and once with each of
    the 4 corruptions at a random severity 0-100. For each item we store the SafeGuard ON
    statistics (MC dropout) and the SafeGuard OFF prediction.
    """
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ["label", "pred", "conf", "agreement", "p_not_sign", "mi",
                            "off_pred", "off_conf", "kind", "severity"]}
    t0 = time.perf_counter()
    for i, im in enumerate(images_u8):
        base = im.astype(np.float32) / 255.0
        variants = [("clean", 0)] + [(k, int(rng.integers(0, 101))) for k in sg.CORRUPTIONS]
        for kind, sev in variants:
            img = base if kind == "clean" else sg.corrupt(base, kind, sev)
            s = mc_one(model, img)
            off_pred, off_conf, _ = sg.predict_off(model, sg.to_model_input(img))
            for k, v in [("label", labels[i]), ("pred", s["pred"]), ("conf", s["conf"]),
                         ("agreement", s["agreement"]), ("p_not_sign", s["p_not_sign"]),
                         ("mi", s["mutual_info"]), ("off_pred", off_pred[0]),
                         ("off_conf", off_conf[0]), ("kind", kind), ("severity", sev)]:
                rows[k].append(v)
        if (i + 1) % 500 == 0:
            print(f"    {name}: {i + 1}/{len(images_u8)} images ({time.perf_counter() - t0:.0f} s)",
                  flush=True)
    return {k: np.array(v) for k, v in rows.items()}


def coverage_accuracy(stats, tau_conf, tau_agree):
    """coverage = share of images ANSWERED, accuracy = share of those answers that are right."""
    status = sg.decide(stats["pred"], stats["conf"], stats["agreement"], stats["p_not_sign"],
                       tau_conf, tau_agree)
    answered = status == sg.ANSWER
    coverage = float(answered.mean())
    accuracy = float((stats["pred"][answered] == stats["label"][answered]).mean()) if answered.any() else float("nan")
    return coverage, accuracy


def calibrate(stats):
    """
    Grid search over tau_conf (0.00..1.00 in 0.01 steps) and tau_agree (0.00..1.00 in 0.05
    steps - agreement can only be k/20). Keep the pair with the HIGHEST coverage whose
    accepted answers are still at least 97% correct.
    Many pairs give exactly the same coverage. Then we take the STRICTEST of them (highest
    tau_agree, then highest tau_conf): same number of answers, but a bigger safety margin,
    and the agreement check really does something.
    """
    best, best_key = None, None
    for tau_conf in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        for tau_agree in np.round(np.arange(0.0, 1.0001, 0.05), 2):
            cov, acc = coverage_accuracy(stats, tau_conf, tau_agree)
            if cov == 0 or not acc >= TARGET_ACCURACY:
                continue
            key = (cov, tau_agree, tau_conf)          # compare coverage first, then strictness
            if best_key is None or key > best_key:
                best, best_key = (float(tau_conf), float(tau_agree), cov, acc), key
    if best is None:
        print("    WARNING: no thresholds reach 97% - using the strictest useful setting")
        best = (0.99, 1.0) + coverage_accuracy(stats, 0.99, 1.0)
    return best


# =========================================================================================
# 4. Charts (1920x1080, dark, big fonts - ready for the slides)
# =========================================================================================

def chart_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
        "axes.edgecolor": GREY, "grid.color": "#2A303A", "font.size": 26,
        "axes.titlesize": 32, "axes.labelsize": 28, "legend.fontsize": 24,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def chart_confidence_vs_accuracy(curves, path):
    chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100, sharey=True)
    for ax, kind in zip(axes, ["fog", "noise"]):
        c = curves[kind]
        sev = np.array(c["severity"])
        acc, conf = 100 * np.array(c["accuracy"]), 100 * np.array(c["mean_confidence"])
        ax.fill_between(sev, acc, conf, where=conf > acc, color=RED, alpha=0.18,
                        label="overconfidence gap")
        ax.plot(sev, conf, color=BLUE, lw=6, label="average confidence")
        ax.plot(sev, acc, color=AMBER, lw=6, label="accuracy")
        ax.set_title(f"{kind.capitalize()}", pad=16)
        ax.set_xlabel(f"{kind} severity (0-100)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.grid(True, alpha=0.6)
    axes[0].set_ylabel("%  (SafeGuard OFF)")
    axes[0].legend(loc="lower left", frameon=False)
    fig.suptitle("Being sure is not being right: the plain model under fog and noise",
                 fontsize=36, y=0.97)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def chart_price_of_safety(curve, calibrated, off_point, path):
    chart_style()
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    cov, acc = 100 * np.array(curve["coverage"]), 100 * np.array(curve["accuracy"])
    ok = ~np.isnan(acc)
    ax.plot(cov[ok], acc[ok], color=AMBER, lw=6, label="SafeGuard ON (strictness 0 -> 100)")
    ax.axhline(100 * TARGET_ACCURACY, color=GREY, lw=3, ls="--")
    ax.text(2, 100 * TARGET_ACCURACY + 0.6, "target: 97% of answers correct", color=FG,
            fontsize=22)
    c_cov, c_acc = 100 * calibrated[0], 100 * calibrated[1]
    ax.scatter([c_cov], [c_acc], s=700, color=FG, edgecolor=AMBER, lw=4, zorder=5)
    ax.annotate(f"calibrated threshold\nanswers {c_cov:.0f}%, {c_acc:.1f}% correct",
                (c_cov, c_acc), xytext=(-40, -150), textcoords="offset points", ha="right",
                fontsize=24, color=FG, arrowprops=dict(arrowstyle="->", color=FG, lw=2))
    o_cov, o_acc = 100 * off_point[0], 100 * off_point[1]
    ax.scatter([o_cov], [o_acc], s=700, marker="s", color=BLUE, zorder=5,
               label="SafeGuard OFF (answers everything)")
    ax.annotate(f"SafeGuard OFF\nanswers 100%, {o_acc:.1f}% correct", (o_cov, o_acc),
                xytext=(-30, -110), textcoords="offset points", ha="right", fontsize=24,
                color=BLUE, arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    lo = max(0, min(np.nanmin(acc[ok]), o_acc) - 5)
    ax.set_ylim(lo, 101)
    ax.set_xlim(0, 102)
    ax.set_xlabel("coverage: share of images the system answers (%)")
    ax.set_ylabel("accuracy of the answers it gives (%)")
    ax.set_title("The price of safety: answering less often, but being right", pad=20,
                 fontsize=36)
    ax.grid(True, alpha=0.6)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# =========================================================================================
# 5. Demo presets
# =========================================================================================

STOP_CLASS = 14
SPEED_CLASSES = [0, 1, 2, 3, 4, 5, 7, 8]
MODERATE_FOG = [50, 40, 60, 30, 70, 80, 90, 100]      # moderate levels are tried first


def find_p2(system, images_u8, labels, min_off_conf=0.8, kinds=("fog",), classes=None):
    """
    Look for: SafeGuard OFF is WRONG with >= min_off_conf confidence AND SafeGuard ON says
    UNCERTAIN. To be robust on stage we also ask that it stays UNCERTAIN at strictness 40
    (slightly looser than the calibrated 50), so tiny slider bumps don't flip it.
    """
    idx = np.arange(len(labels)) if classes is None else np.where(np.isin(labels, classes))[0]
    for kind in kinds:
        for sev in (MODERATE_FOG if kind == "fog" else [50, 40, 60, 70, 80, 30, 90]):
            if len(idx) == 0:
                continue
            imgs = corrupt_many(images_u8[idx], kind, sev)
            off_pred, off_conf = off_predict_many(system.model, imgs)
            # cheap SafeGuard-OFF filter first, then the 20-pass check only for candidates
            cand = np.where((off_pred != labels[idx]) & (off_conf >= min_off_conf))[0]
            for j in cand[np.argsort(-off_conf[cand])]:
                on = system.predict(imgs[j], safeguard_on=True, strictness=50)
                on40 = system.predict(imgs[j], safeguard_on=True, strictness=40)
                if on["status"] == sg.UNCERTAIN and on40["status"] == sg.UNCERTAIN:
                    return int(idx[j]), kind, sev
    return None


def find_presets(system, test_u8, test_labels, test_ids, cifar_u8, cifar_labels, out_dir):
    img_dir = out_dir / "demo_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    presets, notes = {}, []

    # ---- P2 first (hardest), so P1 can reuse the same stop sign if possible -------------
    searches = [
        ("stop sign, fog, OFF conf >= 80%", dict(classes=[STOP_CLASS])),
        ("speed-limit sign, fog, OFF conf >= 80%", dict(classes=SPEED_CLASSES)),
        ("any sign, fog, OFF conf >= 80%", dict()),
        ("any sign, any corruption, OFF conf >= 80%", dict(kinds=tuple(sg.CORRUPTIONS))),
        ("any sign, any corruption, OFF conf >= 70% (loosened)",
         dict(kinds=tuple(sg.CORRUPTIONS), min_off_conf=0.7)),
    ]
    p2 = None
    for i, (desc, kw) in enumerate(searches):
        p2 = find_p2(system, test_u8, test_labels, **kw)
        if p2:
            if i > 0:
                notes.append(f"P2: preferred search found nothing, used: {desc}")
            print(f"    P2 found with search: {desc}")
            break
        print(f"    P2: nothing found for '{desc}', loosening the search...")
    if p2:
        j, kind, sev = p2
        img = sg.corrupt(test_u8[j] / 255.0, kind, sev)
        off = system.predict(img, safeguard_on=False)
        on = system.predict(img, safeguard_on=True)
        Image.fromarray(test_u8[j]).save(img_dir / "p2.png")
        presets["P2"] = dict(title="Foggy sign fools the AI" if kind == "fog" else
                             f"{kind.capitalize()} fools the AI",
                             image="demo_images/p2.png", corruption=kind, severity=sev,
                             true_label=int(test_labels[j]),
                             true_name=sg.CLASS_NAMES[test_labels[j]],
                             source=f"GTSRB test image #{test_ids[j]}",
                             off_label=off["label"], off_conf=off["conf"],
                             on_status=on["status"], on_agreement=on["agreement"],
                             on_conf=on["conf"])
    else:
        notes.append("P2: NOT FOUND even with loosened search")

    # ---- P1: clean stop sign that SafeGuard ON answers correctly ------------------------
    stop_idx = list(np.where(test_labels == STOP_CLASS)[0])
    if p2 and test_labels[p2[0]] == STOP_CLASS:          # same sign as P2 tells a nicer story
        stop_idx.insert(0, p2[0])
    p1 = None
    for j in stop_idx:
        img = test_u8[j] / 255.0
        r = system.predict(img, safeguard_on=True, strictness=50)
        r60 = system.predict(img, safeguard_on=True, strictness=60)   # margin for stage
        if r["status"] == sg.ANSWER and r["pred"] == STOP_CLASS and r60["status"] == sg.ANSWER:
            p1 = (j, r)
            break
    if p1 is None:                                        # loosen: no margin required
        notes.append("P1: no stop sign with a safety margin; accepted one without margin")
        for j in stop_idx:
            r = system.predict(test_u8[j] / 255.0, safeguard_on=True)
            if r["status"] == sg.ANSWER and r["pred"] == STOP_CLASS:
                p1 = (j, r)
                break
    if p1:
        j, r = p1
        Image.fromarray(test_u8[j]).save(img_dir / "p1.png")
        presets["P1"] = dict(title="Clean stop sign", image="demo_images/p1.png",
                             corruption="fog", severity=0, true_label=STOP_CLASS,
                             true_name="Stop", source=f"GTSRB test image #{test_ids[j]}",
                             on_status=r["status"], on_conf=r["conf"],
                             same_sign_as_p2=bool(p2 and j == p2[0]))
    else:
        notes.append("P1: NOT FOUND")

    # ---- P3: a cat or dog that SafeGuard ON sends to SAFE STOP --------------------------
    # Among the cats/dogs that give SAFE STOP, pick the one the plain model is MOST sure is a
    # traffic sign: that makes the contrast between OFF and ON clearest.
    best = None
    for animal in (3, 5):                                  # CIFAR-10: 3 = cat, 5 = dog
        for j in np.where(cifar_labels == animal)[0][:300]:
            img = cifar_u8[j] / 255.0
            on = system.predict(img, safeguard_on=True)
            if on["status"] == sg.SAFE_STOP and on["p_not_sign"] >= 0.6:
                off = system.predict(img, safeguard_on=False)
                if best is None or off["conf"] > best[2]["conf"]:
                    best = (j, animal, off, on)
    if best:
        j, animal, off, on = best
        Image.fromarray(cifar_u8[j]).save(img_dir / "p3.png")
        presets["P3"] = dict(title="Not a sign", image="demo_images/p3.png", corruption="fog",
                             severity=0, true_label=sg.NOT_SIGN, true_name="not a sign",
                             source=f"CIFAR-10 test image #{j} ({'cat' if animal == 3 else 'dog'})",
                             off_label=off["label"], off_conf=off["conf"],
                             on_status=on["status"], on_p_not_sign=on["p_not_sign"])
    else:
        notes.append("P3: NOT FOUND")

    presets["_notes"] = notes
    (out_dir / "presets.json").write_text(json.dumps(presets, indent=2))
    return presets


# =========================================================================================
# Main
# =========================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small subset, 1 epoch")
    parser.add_argument("--retrain", action="store_true", help="train even if model.pt exists")
    args = parser.parse_args()
    cfg = QUICK if args.quick else FULL
    out_dir = ROOT / "outputs" / ("quick" if args.quick else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    sg.seed_everything(0)
    rng = np.random.default_rng(0)

    # ---------------- 1. data -------------------------------------------------------------
    with timed("1. Download + load data"):
        download_all()          # torchvision then only checks the files and unpacks them
        g_x, g_y, g_track = load_gtsrb("train")
        t_x, t_y, _ = load_gtsrb("test")
        # Validation = 10% of the TRACKS (whole physical signs held out), per class
        groups = g_y * 10000 + g_track
        unique_groups = np.unique(groups)
        val_groups = rng.choice(unique_groups, size=len(unique_groups) // 10, replace=False)
        is_val = np.isin(groups, val_groups)
        tr_x, tr_y = g_x[~is_val], g_y[~is_val]
        va_x, va_y = g_x[is_val], g_y[is_val]
        if cfg["gtsrb_train"]:
            keep = rng.choice(len(tr_y), cfg["gtsrb_train"], replace=False)
            tr_x, tr_y = tr_x[keep], tr_y[keep]
        cifar_idx = rng.choice(50000, cfg["cifar_train"], replace=False)
        c_x, _ = load_cifar(train=True, indices=cifar_idx)
        ct_idx = np.arange(cfg["cifar_test"])
        ct_x, ct_y = load_cifar(train=False, indices=ct_idx)
        print(f"    GTSRB train {len(tr_y)}, val {len(va_y)} (held-out tracks), test {len(t_y)}")
        print(f"    CIFAR-10 'not a sign' train {len(c_x)}, CIFAR-10 test {len(ct_x)}")

    # ---------------- 2+3. train (or reuse) ------------------------------------------------
    model_path = out_dir / "model.pt"
    info_path = out_dir / "train_info.json"
    if model_path.exists() and not args.retrain:
        with timed("2. Train model (skipped: reusing existing model.pt, use --retrain)"):
            model = sg.load_model(model_path)
            train_info = json.loads(info_path.read_text()) if info_path.exists() else {}
    else:
        with timed("2. Train model"):
            x_train = to_tensor_batches(np.concatenate([tr_x, c_x]))
            y_train = torch.from_numpy(np.concatenate(
                [tr_y, np.full(len(c_x), sg.NOT_SIGN)])).long()
            x_val = to_tensor_batches(va_x)
            y_val = torch.from_numpy(va_y).long()
            t0 = time.perf_counter()
            model = train_model(x_train, y_train, x_val, y_val, cfg["epochs"])
            train_info = {"train_seconds": time.perf_counter() - t0, "epochs": cfg["epochs"],
                          "train_images": len(y_train)}
            torch.save(model.state_dict(), model_path)
            info_path.write_text(json.dumps(train_info, indent=2))
            del x_train, y_train

    # From here on we only predict: use the same thread count as the app (see safeguard.py)
    sg.use_inference_threads()
    results = {"train": train_info}

    # ---------------- clean accuracy --------------------------------------------------------
    with timed("3. Clean test accuracy (all 12,630 GTSRB test images)"):
        x_test = to_tensor_batches(t_x)
        y_test = torch.from_numpy(t_y).long()
        results["clean_accuracy_off_43"] = accuracy_off(model, x_test, y_test, sg.NOT_SIGN)
        results["clean_accuracy_44"] = accuracy_off(model, x_test, y_test, sg.N_CLASSES)
        print(f"    SafeGuard OFF clean accuracy: {results['clean_accuracy_off_43']:.2%}")
        del x_test

    # ---------------- 4. calibration ---------------------------------------------------------
    with timed("4. Calibrate thresholds on validation mix (clean + corrupted)"):
        vsel = rng.choice(len(va_y), min(cfg["mix_val"], len(va_y)), replace=False)
        val_stats = evaluate_mix(model, va_x[vsel], va_y[vsel], seed=1, name="validation")
        tau_conf, tau_agree, v_cov, v_acc = calibrate(val_stats)
        config = {"tau_conf": tau_conf, "tau_agree": tau_agree, "T": sg.T_PASSES,
                  "target_accuracy": TARGET_ACCURACY,
                  "val_coverage": v_cov, "val_accuracy": v_acc, "val_items": len(val_stats["label"])}
        (out_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(f"    tau_conf = {tau_conf:.2f}, tau_agree = {tau_agree:.2f} -> "
              f"validation coverage {v_cov:.1%}, accuracy {v_acc:.2%}")
        results["calibration"] = config
    system = sg.SafeGuard(model, tau_conf, tau_agree)

    # ---------------- 5. evaluation ----------------------------------------------------------
    test_subset = np.sort(np.random.default_rng(123).choice(len(t_y), cfg["test_subset"], replace=False))
    sub_x, sub_y = t_x[test_subset], t_y[test_subset]

    with timed("5a. Confidence vs accuracy (SafeGuard OFF, fixed test subset)"):
        curves = {}
        for kind in sg.CORRUPTIONS:
            sevs = list(range(0, 101, cfg["sev_step"]))
            accs, confs = [], []
            for sev in sevs:
                pred, conf = off_predict_many(model, corrupt_many(sub_x, kind, sev))
                accs.append(float((pred == sub_y).mean()))
                confs.append(float(conf.mean()))
            curves[kind] = {"severity": sevs, "accuracy": accs, "mean_confidence": confs}
            print(f"    {kind:8s}: accuracy {accs[0]:.0%} -> {accs[-1]:.0%}, "
                  f"confidence {confs[0]:.0%} -> {confs[-1]:.0%}")
        results["confidence_vs_accuracy"] = curves
        chart_confidence_vs_accuracy(curves, out_dir / "confidence_vs_accuracy.png")

    with timed("5b. Price of safety (risk-coverage on test mix, SafeGuard ON)"):
        msel = test_subset[:cfg["mix_test"]]
        test_stats = evaluate_mix(model, t_x[msel], t_y[msel], seed=2, name="test mix")
        rc = {"strictness": [], "coverage": [], "accuracy": [], "tau_conf": [], "tau_agree": []}
        for s in range(0, 101):
            tc, ta = sg.thresholds_for_strictness(s, tau_conf, tau_agree)
            cov, acc = coverage_accuracy(test_stats, tc, ta)
            for k, v in zip(rc, (s, cov, acc, tc, ta)):
                rc[k].append(None if isinstance(v, float) and np.isnan(v) else v)
        cal_cov, cal_acc = coverage_accuracy(test_stats, tau_conf, tau_agree)
        off_acc = float((test_stats["off_pred"] == test_stats["label"]).mean())
        clean = test_stats["kind"] == "clean"
        clean_cov, clean_acc = coverage_accuracy({k: v[clean] for k, v in test_stats.items()},
                                                 tau_conf, tau_agree)
        results["price_of_safety"] = {
            "items": int(len(test_stats["label"])), "curve": rc,
            "calibrated": {"coverage": cal_cov, "accuracy": cal_acc},
            "off": {"coverage": 1.0, "accuracy": off_acc},
            "clean_only_calibrated": {"coverage": clean_cov, "accuracy": clean_acc},
        }
        curve_for_chart = {"coverage": rc["coverage"],
                           "accuracy": [np.nan if a is None else a for a in rc["accuracy"]]}
        chart_price_of_safety(curve_for_chart, (cal_cov, cal_acc), (1.0, off_acc),
                              out_dir / "price_of_safety.png")
        print(f"    test mix: SafeGuard ON answers {cal_cov:.1%}, {cal_acc:.2%} correct; "
              f"SafeGuard OFF answers 100%, {off_acc:.2%} correct")
        print(f"    clean test images only: SafeGuard ON answers {clean_cov:.1%}, {clean_acc:.2%} correct")

    with timed("5c. CIFAR-10 test images (not traffic signs)"):
        off_pred, off_conf = off_predict_many(model, ct_x.astype(np.float32) / 255.0)
        statuses = []
        for im in ct_x:
            s = mc_one(model, im.astype(np.float32) / 255.0)
            statuses.append(sg.decide(s["pred"], s["conf"], s["agreement"], s["p_not_sign"],
                                      tau_conf, tau_agree))
        statuses = np.array(statuses)
        results["cifar10_test"] = {
            "images": int(len(ct_x)),
            "off_mean_confidence": float(off_conf.mean()),
            "off_share_confidence_over_80": float((off_conf >= 0.8).mean()),
            "on_safe_stop_rate": float((statuses == sg.SAFE_STOP).mean()),
            "on_uncertain_rate": float((statuses == sg.UNCERTAIN).mean()),
            "on_answer_rate": float((statuses == sg.ANSWER).mean()),
        }
        r = results["cifar10_test"]
        print(f"    SafeGuard OFF calls them traffic signs with {r['off_mean_confidence']:.1%} "
              f"average confidence")
        print(f"    SafeGuard ON: SAFE STOP {r['on_safe_stop_rate']:.1%}, UNCERTAIN "
              f"{r['on_uncertain_rate']:.1%}, ANSWER (wrongly) {r['on_answer_rate']:.1%}")

    # ---------------- 6. presets -------------------------------------------------------------
    with timed("6. Find demo presets"):
        presets = find_presets(system, sub_x, sub_y, test_subset, ct_x, ct_y, out_dir)
        for key in ("P1", "P2", "P3"):
            p = presets.get(key)
            print(f"    {key}: " + (f"{p['title']} - {p['source']}, {p['corruption']} {p['severity']}"
                                    if p else "NOT FOUND"))
        for note in presets["_notes"]:
            print(f"    NOTE: {note}")
        results["presets_found"] = [k for k in ("P1", "P2", "P3") if k in presets]

    results["timings_seconds"] = TIMINGS
    results["total_seconds"] = time.perf_counter() - t_start
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # ---------------- 7. summary -------------------------------------------------------------
    pos = results["price_of_safety"]
    print("\n" + "=" * 70)
    print("SUMMARY" + ("  (QUICK MODE - numbers are not meaningful)" if args.quick else ""))
    print("=" * 70)
    print(f"Clean GTSRB test accuracy (SafeGuard OFF): {results['clean_accuracy_off_43']:.2%}")
    print(f"Calibrated thresholds: tau_conf = {tau_conf:.2f}, tau_agree = {tau_agree:.2f}")
    print(f"  validation mix: coverage {v_cov:.1%}, accuracy {v_acc:.2%}")
    print(f"  test mix:       coverage {pos['calibrated']['coverage']:.1%}, "
          f"accuracy {pos['calibrated']['accuracy']:.2%}  (OFF: {pos['off']['accuracy']:.2%})")
    print(f"CIFAR-10: OFF mean confidence {results['cifar10_test']['off_mean_confidence']:.1%}, "
          f"ON SAFE STOP rate {results['cifar10_test']['on_safe_stop_rate']:.1%}")
    print(f"Presets found: {', '.join(results['presets_found']) or 'none'}")
    if train_info.get("train_seconds"):
        print(f"Training time: {train_info['train_seconds'] / 60:.1f} min "
              f"({train_info['epochs']} epochs, {train_info['train_images']} images)")
    for name, sec in TIMINGS.items():
        print(f"  {sec:7.1f} s  {name}")
    print(f"Total: {results['total_seconds'] / 60:.1f} min. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
