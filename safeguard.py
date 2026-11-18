"""
safeguard.py - the core of the SafeGuard demo (no Streamlit code in here).

What lives in this file:
  1. CLASS_NAMES      - the 43 GTSRB traffic-sign names + our extra class "not a sign"
  2. Image helpers    - one single path from "any picture" to "what the model sees"
  3. Corruptions      - fog, blur, noise, darkness, controlled by one 0-100 severity
  4. SignNet          - a small CNN (3 conv blocks with BatchNorm and dropout 0.3)
  5. Predictions      - SafeGuard OFF (one plain pass) and SafeGuard ON (MC dropout, T=20)
  6. Decision rule    - ANSWER / UNCERTAIN / SAFE STOP
  7. SafeGuard class  - wraps the model with a real STOP flag and an incident log

prepare.py, check_demo.py and app.py all import from here, so the numbers you see
in the charts, the automated rehearsal and the live app come from the same code.
"""

import datetime
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter

# --------------------------------------------------------------------------------------
# 1. Constants and class names
# --------------------------------------------------------------------------------------

DISPLAY_SIZE = 64   # every picture is first resized to 64x64: this is the "display image"
MODEL_SIZE = 32     # ...and the model only ever sees a 32x32 version of it
T_PASSES = 20       # number of Monte Carlo dropout passes
DROPOUT_P = 0.3     # dropout probability after each block and in the classifier head
NOT_SIGN = 43       # index of our 44th class, "not a sign"
N_CLASSES = 44
CORRUPTIONS = ["fog", "blur", "noise", "darkness"]

# The 43 official GTSRB class names (index = class id), plus our extra class 43.
CLASS_NAMES = [
    "Speed limit (20km/h)",                          # 0
    "Speed limit (30km/h)",                          # 1
    "Speed limit (50km/h)",                          # 2
    "Speed limit (60km/h)",                          # 3
    "Speed limit (70km/h)",                          # 4
    "Speed limit (80km/h)",                          # 5
    "End of speed limit (80km/h)",                   # 6
    "Speed limit (100km/h)",                         # 7
    "Speed limit (120km/h)",                         # 8
    "No passing",                                    # 9
    "No passing for vehicles over 3.5 tons",         # 10
    "Right-of-way at the next intersection",         # 11
    "Priority road",                                 # 12
    "Yield",                                         # 13
    "Stop",                                          # 14
    "No vehicles",                                   # 15
    "Vehicles over 3.5 tons prohibited",             # 16
    "No entry",                                      # 17
    "General caution",                               # 18
    "Dangerous curve to the left",                   # 19
    "Dangerous curve to the right",                  # 20
    "Double curve",                                  # 21
    "Bumpy road",                                    # 22
    "Slippery road",                                 # 23
    "Road narrows on the right",                     # 24
    "Road work",                                     # 25
    "Traffic signals",                               # 26
    "Pedestrians",                                   # 27
    "Children crossing",                             # 28
    "Bicycles crossing",                             # 29
    "Beware of ice/snow",                            # 30
    "Wild animals crossing",                         # 31
    "End of all speed and passing limits",           # 32
    "Turn right ahead",                              # 33
    "Turn left ahead",                               # 34
    "Ahead only",                                    # 35
    "Go straight or right",                          # 36
    "Go straight or left",                           # 37
    "Keep right",                                    # 38
    "Keep left",                                     # 39
    "Roundabout mandatory",                          # 40
    "End of no passing",                             # 41
    "End of no passing by vehicles over 3.5 tons",   # 42
    "not a sign",                                    # 43  <- our outlier class (CIFAR-10 images)
]
assert len(CLASS_NAMES) == N_CLASSES


def use_inference_threads():
    """
    Use the same small number of CPU threads everywhere we predict (prepare.py, check_demo.py,
    app.py). The order in which threads add up numbers can change the last decimal places,
    so a fixed thread count keeps the presets giving exactly the same result in every script.
    """
    import os
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def seed_everything(seed=0):
    """Make runs repeatable: same seed -> same data split, same training, same results."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------------------
# 2. Image helpers: picture -> display image (64x64) -> model input (32x32)
# --------------------------------------------------------------------------------------

def display_uint8(pil_image):
    """Any PIL picture -> 64x64 RGB uint8 array. This is the image we show and corrupt."""
    pil_image = pil_image.convert("RGB")
    if pil_image.size != (DISPLAY_SIZE, DISPLAY_SIZE):
        pil_image = pil_image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


def to_display(pil_image):
    """Any PIL picture -> 64x64 float32 array with values in [0, 1]."""
    return display_uint8(pil_image).astype(np.float32) / 255.0


def to_model_input(images):
    """
    Display image(s) -> model input tensor.
    images: float array (H, W, 3) or (N, H, W, 3) with values in [0, 1].
    returns: torch tensor (N, 3, 32, 32). Uses an anti-aliased resize so that
             downscaling averages pixels (like a real low-resolution camera would).
    """
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 3:
        images = images[None]
    x = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)  # NHWC -> NCHW
    if x.shape[-1] != MODEL_SIZE or x.shape[-2] != MODEL_SIZE:
        x = F.interpolate(x, size=(MODEL_SIZE, MODEL_SIZE), mode="bilinear",
                          align_corners=False, antialias=True)
    return x.contiguous()


def to_uint8(img):
    """float image in [0, 1] -> uint8 image (for showing or saving)."""
    return (np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# --------------------------------------------------------------------------------------
# 3. Corruptions. severity 0 = untouched, 100 = as bad as it gets.
#    They are applied to the 64x64 display image, which is then resized to 32x32.
#    The model NEVER sees these during training, so failures under them are real.
# --------------------------------------------------------------------------------------

def corrupt(img, kind, severity):
    """img: float array (H, W, 3) in [0, 1]. Returns a new corrupted array."""
    f = float(np.clip(severity, 0, 100)) / 100.0      # 0.0 ... 1.0
    img = np.asarray(img, dtype=np.float32)
    if f == 0.0 or kind in (None, "none"):
        return img.copy()

    if kind == "fog":
        # reduce contrast (pull pixels towards the image mean) and blend towards light grey
        mean = img.mean()
        low_contrast = mean + (img - mean) * (1.0 - 0.5 * f)
        alpha = 0.8 * f                               # how much grey "fog" we mix in
        out = (1.0 - alpha) * low_contrast + alpha * 0.82
    elif kind == "blur":
        # Gaussian blur; radius grows with severity (up to 4 px on the 64 px image)
        radius = 4.0 * f
        pil = Image.fromarray(to_uint8(img)).filter(ImageFilter.GaussianBlur(radius))
        out = np.asarray(pil, dtype=np.float32) / 255.0
    elif kind == "noise":
        # Gaussian pixel noise. A FIXED seed makes the same severity always give the same
        # noise pattern, so the demo is repeatable on stage.
        rng = np.random.default_rng(0)
        out = img + rng.normal(0.0, 0.5 * f, size=img.shape).astype(np.float32)
    elif kind == "darkness":
        # lower the brightness (at 100 only 8% of the light is left)
        out = img * (1.0 - 0.92 * f)
    else:
        raise ValueError(f"unknown corruption: {kind}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------------------
# 4. The model: a small CNN
# --------------------------------------------------------------------------------------

class SignNet(nn.Module):
    """
    3 conv blocks, each: Conv 3x3 -> BatchNorm -> ReLU -> MaxPool(2) -> Dropout(0.3)
        32x32x3 -> 16x16xW -> 8x8x2W -> 4x4x4W        (W = width = 48 channels)
    Head: Flatten -> Linear(4*4*4W, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 44)
    """

    def __init__(self, n_classes=N_CLASSES, p_drop=DROPOUT_P, width=48):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout(p_drop),
            )

        w = width
        self.features = nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * w * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x has values in [0, 1]; shift to [-1, 1] so the first layer sees centred data
        return self.head(self.features(x * 2.0 - 1.0))


def load_model(path):
    model = SignNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# 5. Predictions
# --------------------------------------------------------------------------------------

@torch.no_grad()
def predict_off(model, x):
    """
    SafeGuard OFF: the plain classifier. model.eval() -> dropout OFF, one forward pass.
    A plain traffic-sign classifier has no "not a sign" exit (that output belongs to the
    SafeGuard layer), so we take the softmax over the 43 sign outputs only. It therefore
    ALWAYS names a traffic sign - even for a cat.
    x: tensor (N, 3, 32, 32). Returns numpy arrays: pred (N,), conf (N,), probs (N, 43).
    """
    model.eval()
    probs = F.softmax(model(x)[:, :NOT_SIGN], dim=1)
    conf, pred = probs.max(dim=1)
    return pred.numpy(), conf.numpy(), probs.numpy()


@torch.no_grad()
def mc_dropout_probs(model, x1, T=T_PASSES):
    """
    SafeGuard ON: Monte Carlo dropout for ONE image.
    We keep BatchNorm in eval mode but switch the Dropout layers back ON, then look at the
    same image T=20 times. Each pass drops a different random 30% of the neurons, so the
    passes disagree when the model is unsure.
    For speed the 20 passes run as one batch of 20 copies: every copy gets its own random
    dropout mask, which is exactly the same as running 20 separate passes.
    torch.manual_seed(0) right before the passes -> same image + same severity always gives
    the same answer on stage.
    x1: tensor (1, 3, 32, 32). Returns numpy array (T, 44) of softmax probabilities.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()                      # dropout ON at test time
    try:
        torch.manual_seed(0)
        probs = F.softmax(model(x1.repeat(T, 1, 1, 1)), dim=1)
    finally:
        model.eval()                       # always put the model back to normal
    return probs.numpy()


def entropy(p, axis=-1):
    """Shannon entropy in nats. The tiny 1e-12 avoids log(0)."""
    return -(p * np.log(p + 1e-12)).sum(axis=axis)


def mc_summary(probs):
    """
    Turn the T x 44 matrix of MC-dropout probabilities into the numbers we decide on.
      p_mean     = mean softmax over the passes
      pred       = argmax(p_mean),  conf = p_mean[pred]
      agreement  = fraction of passes whose own argmax equals pred
      mutual_info= H(p_mean) - mean_t H(p_t)   (model uncertainty: high when passes disagree)
    """
    p_mean = probs.mean(axis=0)
    pred = int(p_mean.argmax())
    votes = probs.argmax(axis=1)                        # each pass's own answer
    return {
        "p_mean": p_mean,
        "pred": pred,
        "conf": float(p_mean[pred]),
        "agreement": float((votes == pred).mean()),
        "mutual_info": float(entropy(p_mean) - entropy(probs, axis=1).mean()),
        "p_not_sign": float(p_mean[NOT_SIGN]),
        "votes": np.bincount(votes, minlength=N_CLASSES),
    }


# --------------------------------------------------------------------------------------
# 6. Decision rule (SafeGuard ON)
# --------------------------------------------------------------------------------------

ANSWER, UNCERTAIN, SAFE_STOP, HALTED = "ANSWER", "UNCERTAIN", "SAFE STOP", "HALTED"


def decide(pred, conf, agreement, p_not_sign, tau_conf, tau_agree):
    """Works on single numbers or on numpy arrays (used for calibration)."""
    if np.ndim(pred) == 0:
        if pred == NOT_SIGN or p_not_sign >= 0.5:
            return SAFE_STOP
        if conf >= tau_conf and agreement >= tau_agree:
            return ANSWER
        return UNCERTAIN
    pred, conf, agreement, p_not_sign = map(np.asarray, (pred, conf, agreement, p_not_sign))
    out = np.full(pred.shape, UNCERTAIN, dtype=object)
    out[(conf >= tau_conf) & (agreement >= tau_agree)] = ANSWER
    out[(pred == NOT_SIGN) | (p_not_sign >= 0.5)] = SAFE_STOP
    return out


def thresholds_for_strictness(strictness, tau_conf_cal, tau_agree_cal):
    """
    The "Strictness" slider (0-100). 50 = the calibrated thresholds.
    0   -> thresholds 0 (answer anything that is not a SAFE STOP)
    100 -> thresholds 1 (only answer when every pass agrees and confidence is ~100%)
    In between we move in a straight line.
    """
    s = float(np.clip(strictness, 0, 100))
    if s <= 50:
        t = s / 50.0
        return t * tau_conf_cal, t * tau_agree_cal
    t = (s - 50.0) / 50.0
    return (tau_conf_cal + t * (1.0 - tau_conf_cal),
            tau_agree_cal + t * (1.0 - tau_agree_cal))


def top_k(probs, k=3, names=CLASS_NAMES):
    idx = np.argsort(probs)[::-1][:k]
    return [(names[i], float(probs[i])) for i in idx]


# --------------------------------------------------------------------------------------
# 7. The SafeGuard system: model + decision rule + STOP flag + incident log
# --------------------------------------------------------------------------------------

class SafeGuard:
    def __init__(self, model, tau_conf, tau_agree):
        self.model = model
        self.tau_conf_cal = float(tau_conf)
        self.tau_agree_cal = float(tau_agree)
        self.halted = False          # the EU AI Act Art. 14 "stop button" flag
        self.incidents = []          # list of dicts, newest last

    # ---- human oversight ---------------------------------------------------------
    def stop(self, note="STOP button pressed"):
        self.halted = True
        self.log_incident("STOP", note=note)

    def reset(self, note="Reset pressed"):
        self.halted = False
        self.log_incident("RESET", note=note)

    # ---- prediction ------------------------------------------------------------------
    def predict(self, display_img, safeguard_on=True, strictness=50):
        """
        display_img: float array (64, 64, 3) in [0, 1] (already corrupted if wanted).
        Returns a dict with status, prediction and all scores. If the system is halted,
        the model is NOT called at all.
        """
        if self.halted:
            return {"status": HALTED, "pred": None, "label": None, "conf": None,
                    "safeguard_on": safeguard_on, "latency_ms": 0.0}

        t0 = time.perf_counter()
        x = to_model_input(display_img)
        if not safeguard_on:
            pred, conf, probs = predict_off(self.model, x)
            pred, conf, probs = int(pred[0]), float(conf[0]), probs[0]
            result = {"status": ANSWER, "pred": pred, "label": CLASS_NAMES[pred],
                      "conf": conf, "top3": top_k(probs), "safeguard_on": False,
                      "agreement": None, "mutual_info": None, "p_not_sign": None,
                      "votes": None}
        else:
            tau_conf, tau_agree = thresholds_for_strictness(
                strictness, self.tau_conf_cal, self.tau_agree_cal)
            s = mc_summary(mc_dropout_probs(self.model, x))
            status = decide(s["pred"], s["conf"], s["agreement"], s["p_not_sign"],
                            tau_conf, tau_agree)
            result = {"status": status, "pred": s["pred"], "label": CLASS_NAMES[s["pred"]],
                      "conf": s["conf"], "top3": top_k(s["p_mean"]), "safeguard_on": True,
                      "agreement": s["agreement"], "mutual_info": s["mutual_info"],
                      "p_not_sign": s["p_not_sign"], "votes": s["votes"],
                      "tau_conf": tau_conf, "tau_agree": tau_agree}
        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return result

    # ---- honest incident reporting ---------------------------------------------------
    def log_incident(self, event, image="", corruption="", severity=None, result=None, note=""):
        r = result or {}
        self.incidents.append({
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "image": image,
            "corruption": corruption,
            "severity": severity,
            "best_guess": r.get("label"),
            "confidence": None if r.get("conf") is None else round(r["conf"], 4),
            "agreement": None if r.get("agreement") is None else round(r["agreement"], 4),
            "mutual_info": None if r.get("mutual_info") is None else round(r["mutual_info"], 4),
            "p_not_sign": None if r.get("p_not_sign") is None else round(r["p_not_sign"], 4),
            "note": note,
        })


def load_system(out_dir="outputs"):
    """Load outputs/model.pt + outputs/config.json and build a SafeGuard system."""
    out_dir = Path(out_dir)
    cfg = json.loads((out_dir / "config.json").read_text())
    use_inference_threads()
    return SafeGuard(load_model(out_dir / "model.pt"), cfg["tau_conf"], cfg["tau_agree"])
