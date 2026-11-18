# SafeGuard: an AI that knows when it doesn't know

A live classroom demo for a 10-minute ethics talk on **fail-safe AI**. A small traffic-sign
classifier is shown twice: once plain (**SafeGuard OFF**), and once wrapped in a fail-safe
layer (**SafeGuard ON**) that measures its own uncertainty, hands control to a human when
unsure, refuses to act on things that are not traffic signs, and has a real **STOP button**
(EU AI Act Article 14) plus an honest **incident log**.

Everything runs **offline on a normal laptop CPU** (Windows or macOS, Python 3.10-3.12).
Internet is only needed once, for `pip install` and the first `python prepare.py` (data download).

---

## 1. Setup (once, at home, with internet)

Open a terminal **in this folder**.

**Windows (PowerShell)**

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\Activate.ps1
```

**macOS**

```bash
python3 -m venv .venv
```

```bash
source .venv/bin/activate
```

(If PowerShell refuses to run `Activate.ps1`, skip activation and start every command with
`.venv\Scripts\python.exe` instead of `python`, e.g. `.venv\Scripts\python.exe prepare.py`
and `.venv\Scripts\python.exe -m streamlit run app.py`.)

**Both** - install the packages (the first line gets the small CPU-only PyTorch):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

```bash
pip install -r requirements.txt
```

## 2. Build everything (one command, ~15-20 min)

```bash
python prepare.py
```

This downloads GTSRB (~280 MB) and CIFAR-10 (~170 MB) into `./data`, trains the model
(~7-12 min on a laptop CPU), calibrates the thresholds, runs the evaluation (~8 min), finds the
demo presets, and prints a summary with timings. Everything it makes goes into `./outputs`.

- Downloads time out after 30 s of silence and **resume** where they stopped, so a slow or
  dropped connection just means running the command again.
- If `outputs/model.pt` already exists it is **reused** (add `--retrain` to train again).
  Calibration, charts and presets are always recomputed.
- `python prepare.py --quick` is a ~2-minute pipeline check (small subset, 1 epoch). It writes
  to `outputs/quick/` and never touches the real model.

### Results of our run (Windows laptop, 10-core CPU)

| | |
|---|---|
| Clean GTSRB test accuracy (SafeGuard OFF, 12,630 images) | **95.67%** |
| Calibrated thresholds | **τ_conf = 0.60, τ_agree = 0.55** (11 of 20 passes) |
| Validation mix (7,500 images): coverage / accuracy of answers | 68.5% / 97.08% |
| Test mix (10,000 images: clean + 4 corruptions) SafeGuard ON | answers **72.1%**, **98.31%** correct |
| Same test mix, SafeGuard OFF | answers 100%, 81.93% correct |
| Clean test images only, SafeGuard ON | answers 92.1%, 99.13% correct |
| CIFAR-10 test (10,000 non-signs): SafeGuard OFF average confidence | 72.0% (it calls them signs) |
| CIFAR-10 test: SafeGuard ON → SAFE STOP | **100.0%** |
| Training time | 12.0 min, 12 epochs, 32,000 images (≈7 min when the laptop is otherwise idle) |

Presets found automatically (all three, no loosening needed):
- **P1** GTSRB test #11477, a Stop sign: SafeGuard ON → ANSWER: Stop, 100%, 20/20 agree.
- **P2** the *same* stop sign at fog 70: SafeGuard OFF → "Speed limit (30km/h)", 83% sure.
  SafeGuard ON → UNCERTAIN (38% confidence, 9/20 passes agree).
- **P3** CIFAR-10 test #899, a dog: SafeGuard OFF → "Yield", 99.99% sure. SafeGuard ON →
  SAFE STOP (P(not a sign) = 96%).

Your numbers may differ slightly on another machine; `check_demo.py` tells you if the demo
still works.

## 3. Stage rehearsal (run it before every talk)

```bash
python check_demo.py
```

It loads the model without the UI and checks every moment of the demo (P1, P2 OFF/ON, P3,
STOP button, determinism). It prints a PASS/FAIL table and exits with an error if anything fails.

## 4. Start the app

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in the browser (the app does not open it by itself, so
no first-run prompt can block you). Press **F11** for full screen on the projector.
The app only listens on `localhost` and makes no internet calls (no CDN, no telemetry), so
Wi-Fi can be off. The sidebar (Strictness) starts collapsed; open it with the `>` at the top left.

---

## On stage: click-by-click script (about 6 minutes)

Before you start: run `python check_demo.py` (all PASS), start the app, full screen,
the **SafeGuard layer** switch shows **ON** (blue), Strictness at **50**, Incident log empty (restart the app to
clear it). If the screen says HALTED, press **Reset**.

| # | Click | What the audience sees | What you say |
|---|---|---|---|
| 1 | **P1 · Clean stop** | Blue banner **✔ ANSWER: Stop**, 20/20 passes agree | "On a clear day it works. And with SafeGuard ON it also tells us it is sure: all 20 looks agree." |
| 2 | SafeGuard layer → **OFF** (turns grey) | Blue **ANSWER: Stop**, kicker "SafeGuard OFF · no safety checks", all three checks greyed out as *not checked* | "This is the plain model. Same answer. So far, no difference." |
| 3 | **P2 · Foggy sign** (still OFF) | The same stop sign in fog; blue **ANSWER: Speed limit (30km/h)**, 83% | "Now fog. The plain model is 83% sure this stop sign is a speed limit. It never trained on fog, and it has no way to say 'I don't know'." |
| 4 | SafeGuard layer → **ON** (turns blue) | Amber **UNCERTAIN → handing control to a human**; the checklist shows ✗ confidence 38% (needs 60%) and ✗ 9/20 agree (needs 11/20); the 20-square vote grid lights up in mixed colours | "SafeGuard looks 20 times with different neurons switched off. Only 9 of 20 agree, so it hands control to a human instead of guessing." |
| 5 | Drag **Severity** to 0, then back to 70 | Blue ANSWER: Stop at 0, amber UNCERTAIN again at 70 | "It answers when it can, and steps back when it can't." |
| 6 | **P3 · Not a sign** | Red **⏹ SAFE STOP: this is not a traffic sign** | "This is a dog. (Optional: switch OFF - the plain model says 'Yield', 99.99% sure. Switch back ON.) SafeGuard refuses to act." |
| 7 | Big red **STOP** | Pulsing red **SYSTEM HALTED — SAFE STATE**, the header chip turns to *System halted*, the camera image greys out; clicking presets changes nothing | "Article 14 of the EU AI Act: a human must be able to stop the system. This really halts it - the model is not even called." |
| 8 | **Reset** | Normal again | "Only a human can bring it back." |
| 9 | Tab **Incident log** → **Download incident log (CSV)** | Every UNCERTAIN, SAFE STOP, STOP and RESET with time and scores | "Honest incident reporting: every time it stepped back, it wrote it down." |
| 10 | Tab **Price of safety**, move the slider | The white dot slides along the curve: coverage goes down as accuracy goes up (optional: *Use this strictness in the live demo*) | "Safety has a price: it answers fewer images, but the answers it gives are right." |
| 11 | Tab **Confidence vs reality** (switch Fog / Noise / Blur / Darkness) | Blue confidence line stays above the amber accuracy line; the red band is the gap | "The gap between being sure and being right. That gap is why fail-safes exist." |

If something looks wrong: press **Reset**, check Strictness is **50**, press the preset again.
The same image and severity always give exactly the same result.

---

## How it works (for the viva)

| File | What it does |
|---|---|
| `safeguard.py` | Class names, image pipeline, corruptions, the CNN, MC dropout, decision rule, STOP flag and incident log. No Streamlit code. |
| `prepare.py` | Download → train → calibrate → evaluate → charts → presets → summary. |
| `app.py` | The Streamlit screen only. Calls `safeguard.py`. One CSS block (design system) + small HTML helpers for the verdict banner, decision checklist and 20-pass vote grid; interactive charts use Streamlit's bundled Altair/Vega (offline). |
| `check_demo.py` | Automated stage rehearsal. |
| `.streamlit/config.toml` | Dark theme, no telemetry, no file watcher. |

**Data.** GTSRB (43 sign classes) + 8,000 CIFAR-10 training images as a 44th class
"not a sign" (*outlier exposure*, Hendrycks et al. 2019). Validation holds out whole GTSRB
*tracks* (all ~30 photos of one physical sign), so validation never contains near-copies of
training images. Every picture follows one path: resize to 64×64 (the "display image" that we
show and corrupt) → resize to 32×32 for the model.

**Model.** 3 blocks of Conv 3×3 → BatchNorm → ReLU → MaxPool → Dropout(0.3) with 48, 96 and
192 channels, then Linear(3072→256) → ReLU → Dropout(0.3) → Linear(256→44). Trained for 12
epochs with AdamW and a one-cycle learning rate, on **clean images only** (±10° rotation and
×0.75-1.25 brightness augmentation), so its failure under fog is real. (A 32/64/128-channel
version trained faster but stayed at 94.6%, below the 95% target.) Batches come from a
`DataLoader` with `num_workers=0`, which is safe on Windows.

**SafeGuard OFF.** `model.eval()` (dropout off), one forward pass, softmax over the 43 sign
outputs. A plain sign classifier has no "not a sign" exit, so OFF always names a sign.

**SafeGuard ON (MC dropout).** Dropout stays ON at test time and the image goes through the
network T = 20 times (as one batch of 20 copies - each copy gets its own random mask, which is
the same as 20 separate passes). `torch.manual_seed(0)` right before the passes makes the
result repeatable on stage.
- `p_mean` = mean softmax of the 20 passes; `pred = argmax(p_mean)`; `conf = p_mean[pred]`
- `agreement` = share of passes whose own argmax equals `pred`
- `mutual information` = H(p_mean) − mean H(p_t): high when the passes disagree
- `pred` is "not a sign" or P(not a sign) ≥ 0.5 → **SAFE STOP**;
  else `conf ≥ τ_conf` and `agreement ≥ τ_agree` → **ANSWER**; else → **UNCERTAIN**

**Calibration.** On a validation mix (1,500 validation images, each clean + once with each
corruption at a random severity = 7,500 items) we try every τ_conf in 0.00…1.00 (step 0.01)
and τ_agree in 0.00…1.00 (step 0.05) and keep the pair that answers the most images while the
answers stay ≥ 97% correct. Many pairs tie on coverage; we then take the strictest one
(highest τ_agree, then τ_conf): same number of answers, bigger safety margin. A SAFE STOP on a
sign counts as "not answered". Saved in `outputs/config.json`.

**Presets** are searched with exactly the same code the app runs. To survive a nudge of the
slider, P2 must also be UNCERTAIN at strictness 40, and P1 must also be ANSWER at strictness 60.
All predicting scripts use the same CPU thread count, so the last decimal places match too.

**Strictness slider.** 50 = calibrated thresholds; 0 = both thresholds 0; 100 = both 1.
In between the thresholds move in a straight line. The "price of safety" curve is this slider
swept from 0 to 100 on a test mix.

**Corruptions** (on the 64×64 image, severity 0-100): fog = lower contrast + blend toward light
grey; blur = Gaussian blur up to radius 4 px; noise = Gaussian noise up to σ = 0.5 with a fixed
seed (same severity → same noise); darkness = brightness down to 8%.

**STOP button.** `SafeGuard.stop()` sets a flag; while it is set `predict()` returns HALTED
without calling the model (`check_demo.py` checks this with a forward hook). Only `reset()`
clears it. The flag lives in the cached SafeGuard object, so it even survives a page refresh.

## Outputs

| File | Content |
|---|---|
| `outputs/model.pt` | trained weights |
| `outputs/config.json` | τ_conf, τ_agree and validation coverage/accuracy |
| `outputs/results.json` | all evaluation numbers and curves, timings |
| `outputs/confidence_vs_accuracy.png` | slide chart, 1920×1080 |
| `outputs/price_of_safety.png` | slide chart, 1920×1080 |
| `outputs/presets.json`, `outputs/demo_images/` | the three demo presets |

`data/`, `outputs/` and `.venv/` are in `.gitignore`.
