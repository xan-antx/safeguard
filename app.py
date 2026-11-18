"""
app.py - the SafeGuard live demo (Streamlit).   Run:  streamlit run app.py

All the AI logic lives in safeguard.py; this file only draws the screen.
Streamlit re-runs this whole script top to bottom every time someone clicks something.
Things that must survive a re-run live in st.session_state (per browser tab) or in the
cached SafeGuard object (shared, so the STOP flag survives even a page refresh).
Nothing here touches the internet: the model, images and charts are all local files.
"""

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import safeguard as sg

OUT = Path(__file__).resolve().parent / "outputs"
BLUE, AMBER, RED, BG, FG = "#5AA9E6", "#F2A900", "#E5484D", "#0E1217", "#F2EEE6"

st.set_page_config(page_title="SafeGuard", page_icon="🛑", layout="wide",
                   initial_sidebar_state="collapsed")     # Strictness is backstage: open with >

# ---- big fonts and colours for the projector -------------------------------------------
st.markdown(f"""
<style>
  html, body {{ font-size: 19px; }}
  .block-container {{ padding-top: 1.6rem; max-width: 1700px; }}
  h1 {{ font-size: 2.5rem !important; margin-bottom: 0 !important; }}
  .stButton button p, .stDownloadButton button p {{ font-size: 1.15rem; font-weight: 700;
      white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.2; }}
  .st-key-on label p {{ white-space: nowrap; overflow: visible; }}
  /* make the ON/OFF switch itself twice as big, so it is easy to hit on stage */
  .st-key-on label > div:first-of-type {{ transform: scale(1.9); transform-origin: left center;
      margin-right: 2.4rem; }}
  .tiles {{ display: flex; flex-wrap: wrap; gap: .7rem; margin: .6rem 0 1rem; }}
  .tile {{ background: #1B212B; border-radius: 10px; padding: .55rem 1rem; min-width: 8.5rem; }}
  .tile .k {{ font-size: 1rem; opacity: .85; }}
  .tile .v {{ font-size: 2rem; font-weight: 800; line-height: 1.15; }}
  .tile .n {{ font-size: .85rem; opacity: .75; }}
  .stButton button {{ min-height: 3.2rem; }}
  /* the STOP button: big, red, impossible to miss */
  .st-key-stop_btn button {{ background: {RED} !important; border: 3px solid #FFFFFF !important;
      min-height: 4rem; }}
  .st-key-stop_btn button p {{ color: #FFFFFF !important; font-size: 1.7rem !important;
      font-weight: 900 !important; letter-spacing: .08em; }}
  .st-key-reset_btn button {{ min-height: 4rem; border: 2px solid {FG} !important; }}
  .st-key-on label p {{ font-size: 1.6rem !important; font-weight: 800; }}
  [data-testid="stMetricValue"] {{ font-size: 2.1rem; }}
  [data-testid="stMetricLabel"] p {{ font-size: 1.05rem; }}
  [data-testid="stTab"] p {{ font-size: 1.4rem !important; font-weight: 700; }}
  [data-testid="stTab"] {{ padding: .3rem 1.2rem; }}
  section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li {{ font-size: 1.05rem; }}
  .banner {{ border-radius: 14px; padding: 1.1rem 1.4rem; margin-bottom: 1rem; }}
  .banner .status {{ font-size: 2.3rem; font-weight: 900; line-height: 1.15; }}
  .banner .sub {{ font-size: 1.2rem; margin-top: .35rem; font-weight: 600; }}
  .bar-row {{ display: flex; align-items: center; margin: .3rem 0; font-size: 1.15rem; }}
  .bar-name {{ width: 46%; }}
  .bar-track {{ flex: 1; background: #232A35; border-radius: 6px; height: 1.3rem; margin: 0 .6rem; }}
  .bar-fill {{ height: 100%; border-radius: 6px; }}
  .bar-pct {{ width: 4.2rem; text-align: right; font-weight: 700; }}
  .pill {{ display: inline-block; padding: .3rem .9rem; border-radius: 999px; font-weight: 800;
      font-size: 1.2rem; }}
</style>
""", unsafe_allow_html=True)


# ---- load everything once (cached across re-runs) ---------------------------------------
@st.cache_resource
def get_system():
    """The SafeGuard system (model + STOP flag + incident log). One shared copy."""
    return sg.load_system(OUT)


@st.cache_data
def load_json(name):
    import json
    path = OUT / name
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_preset_image(rel_path):
    with Image.open(OUT / rel_path) as im:
        return sg.to_display(im)


if not (OUT / "model.pt").exists() or not (OUT / "config.json").exists():
    st.error("No trained model found. Run `python prepare.py` first, then restart the app.")
    st.stop()

system = get_system()
presets = load_json("presets.json")
results = load_json("results.json")
ss = st.session_state
ss.setdefault("source", "P1" if "P1" in presets else None)
ss.setdefault("corruption", "fog")
ss.setdefault("severity", 0)
ss.setdefault("strictness", 50)
ss.setdefault("on", True)
ss.setdefault("last_logged", None)


# ---- button callbacks (run BEFORE the script re-runs) -----------------------------------
def apply_preset(key):
    p = presets[key]
    ss.source = key
    ss.corruption = p["corruption"]
    ss.severity = int(p["severity"])
    ss.last_logged = None


def on_upload():
    if ss.get("upload") is not None:
        ss.source = "upload"
        ss.severity = 0
        ss.last_logged = None


def press_stop():
    system.stop(note="Human pressed the STOP button (EU AI Act Art. 14)")


def press_reset():
    system.reset(note="Human pressed Reset")
    ss.last_logged = None


# =========================================================================================
# Sidebar: strictness
# =========================================================================================
with st.sidebar:
    st.markdown("## Strictness")
    st.slider("How careful should SafeGuard be?", 0, 100, key="strictness",
              help="50 = thresholds calibrated in prepare.py. Higher = answers less often.")
    tau_conf, tau_agree = sg.thresholds_for_strictness(ss.strictness, system.tau_conf_cal,
                                                       system.tau_agree_cal)
    st.markdown(f"**Confidence needed:** {tau_conf:.0%}  \n**Passes that must agree:** "
                f"{tau_agree:.0%} ({int(np.ceil(tau_agree * sg.T_PASSES - 1e-9))}/{sg.T_PASSES})")
    if ss.strictness != 50:
        st.button("Back to calibrated (50)", on_click=lambda: ss.update(strictness=50))
    st.markdown("---")
    st.markdown(
        "**How SafeGuard decides**  \n"
        "The image goes through the network **20 times** with dropout ON (MC dropout).  \n"
        "• says *not a sign*, or P(not a sign) ≥ 50% → **SAFE STOP**  \n"
        "• confident **and** passes agree → **ANSWER**  \n"
        "• otherwise → **UNCERTAIN**, a human decides")

# =========================================================================================
# Top bar: title, ON/OFF, presets, STOP + Reset (always visible above the tabs)
# =========================================================================================
st.title("SafeGuard: an AI that knows when it doesn't know")
top = st.columns([1.95, 1.15, 1.15, 1.15, 0.1, 1.3, 0.85], vertical_alignment="center")
with top[0]:
    # the label always says the CURRENT state in words (not just the switch colour)
    st.toggle("SafeGuard ON" if ss.on else "SafeGuard OFF", key="on")
SHORT = {"P1": "Clean stop", "P2": "Foggy sign", "P3": "Not a sign"}   # fits on the button
for col, key in zip(top[1:4], ["P1", "P2", "P3"]):
    with col:
        p = presets.get(key)
        st.button(f"{key} · {SHORT[key]}" if p else f"{key}: not found", key=f"btn_{key}",
                  disabled=p is None, on_click=apply_preset, args=(key,), width="stretch",
                  help=f"{p['title']} - {p['source']}" if p else "prepare.py could not find this preset")
with top[5]:
    st.button("STOP", key="stop_btn", on_click=press_stop, width="stretch",
              help="Halts the system: no more predictions until Reset")
with top[6]:
    st.button("Reset", key="reset_btn", on_click=press_reset, width="stretch")

tab_live, tab_price, tab_conf, tab_log = st.tabs(
    ["Live demo", "Price of safety", "Confidence vs reality", "Incident log"])


# =========================================================================================
# Helpers for drawing
# =========================================================================================
def banner(color, status, sub, text_color="#0E1217"):
    st.markdown(f'<div class="banner" style="background:{color};color:{text_color}">'
                f'<div class="status">{status}</div><div class="sub">{sub}</div></div>',
                unsafe_allow_html=True)


def prob_bars(top3, color):
    rows = "".join(
        f'<div class="bar-row"><div class="bar-name">{name}</div><div class="bar-track">'
        f'<div class="bar-fill" style="width:{p * 100:.1f}%;background:{color}"></div></div>'
        f'<div class="bar-pct">{p:.0%}</div></div>' for name, p in top3)
    st.markdown(rows, unsafe_allow_html=True)


def tiles(items):
    """A row of big number tiles: items = [(label, value, small note), ...]. Wraps if narrow."""
    html = "".join(f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div>'
                   f'<div class="n">{n}</div></div>' for k, v, n in items)
    st.markdown(f'<div class="tiles">{html}</div>', unsafe_allow_html=True)


def vote_chart(votes, pred):
    """Horizontal bar chart: how many of the 20 passes voted for each class."""
    idx = [i for i in np.argsort(votes)[::-1] if votes[i] > 0][:5]
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=100)          # fixed size: same look every time
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    names = [sg.CLASS_NAMES[i] if len(sg.CLASS_NAMES[i]) < 28 else sg.CLASS_NAMES[i][:26] + "…"
             for i in idx]
    ax.barh(names[::-1], [votes[i] for i in idx][::-1],
            color=[AMBER if i == pred else BLUE for i in idx][::-1])
    ax.set_xlim(0, sg.T_PASSES)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_ylim(-0.6, max(len(idx), 3) - 0.4)                   # keep bars thin when few classes
    ax.set_xlabel("votes (out of 20 passes)", color=FG, fontsize=15)
    ax.tick_params(colors=FG, labelsize=15)
    for s in ax.spines.values():
        s.set_color("#5B6270")
    for i, v in enumerate([votes[i] for i in idx][::-1]):
        ax.text(v + 0.3, i, str(v), va="center", color=FG, fontsize=15, fontweight="bold")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG)
    plt.close(fig)
    return buf.getvalue()


def upscale(img, size):
    return Image.fromarray(sg.to_uint8(img)).resize((size, size), Image.LANCZOS)


# =========================================================================================
# Tab 1: Live demo
# =========================================================================================
with tab_live:
    left, right = st.columns([1, 1.35], gap="large")

    # ---------------- left: image + controls --------------------------------------------
    with left:
        clean, image_name = None, ""
        if ss.source == "upload" and ss.get("upload") is not None:
            try:
                with Image.open(ss.upload) as im:
                    clean = sg.to_display(im)
                image_name = f"upload: {ss.upload.name}"
            except Exception:                          # noqa: BLE001 - never crash on stage
                st.error("Could not read that file as an image. Showing P1 instead.")
                ss.source = "P1" if "P1" in presets else None
        if clean is None and ss.source in presets:
            clean = load_preset_image(presets[ss.source]["image"])
            image_name = f"{ss.source} ({presets[ss.source]['source']})"
        if clean is None:
            st.warning("No image yet: upload one below.")

        if clean is not None:
            shown = sg.corrupt(clean, ss.corruption, ss.severity)
            st.image(upscale(shown, 360), width=360)
            st.caption(f"{image_name} · {ss.corruption} {ss.severity}")
            with st.expander("What the AI actually sees (32×32 pixels)"):
                tiny = sg.to_model_input(shown)[0].permute(1, 2, 0).numpy()
                st.image(Image.fromarray(sg.to_uint8(tiny)).resize((160, 160), Image.NEAREST))

        c1, c2 = st.columns([1, 1.6])
        with c1:
            st.selectbox("Corruption", sg.CORRUPTIONS, key="corruption")
        with c2:
            st.slider("Severity", 0, 100, key="severity")
        st.file_uploader("Or upload your own image", type=["png", "jpg", "jpeg", "bmp", "ppm", "webp"],
                         key="upload", on_change=on_upload)

    # ---------------- right: decision ----------------------------------------------------
    with right:
        result = None
        if clean is not None:
            try:
                result = system.predict(shown, safeguard_on=ss.on, strictness=ss.strictness)
            except Exception as e:                     # noqa: BLE001 - show, don't crash
                banner(RED, "ERROR → handing control to a human", str(e)[:120], "#FFFFFF")

        if result is None:
            pass
        elif result["status"] == sg.HALTED:
            banner(RED, "⏹ SYSTEM HALTED — SAFE STATE",
                   "A human pressed STOP. The model is not being called. Press Reset to resume.",
                   "#FFFFFF")
        elif not result["safeguard_on"]:
            banner(BLUE, f"ANSWER: {result['label']}",
                   f"SafeGuard OFF · one pass, dropout off, no safety checks · "
                   f"confidence {result['conf']:.0%}")
        elif result["status"] == sg.ANSWER:
            banner(BLUE, f"✔ ANSWER: {result['label']}",
                   f"Confident ({result['conf']:.0%}) and "
                   f"{round(result['agreement'] * sg.T_PASSES)}/20 passes agree")
        elif result["status"] == sg.UNCERTAIN:
            banner(AMBER, "⚠ UNCERTAIN → handing control to a human",
                   f"Best guess: {result['label']} ({result['conf']:.0%}), "
                   f"{round(result['agreement'] * sg.T_PASSES)}/20 passes agree — not enough to act")
        else:
            banner(RED, "⏹ SAFE STOP: this is not a traffic sign",
                   f"P(not a sign) = {result['p_not_sign']:.0%} · the system refuses to act",
                   "#FFFFFF")

        # incident log: record each new UNCERTAIN / SAFE STOP once (not on every re-run)
        if result and result["status"] in (sg.UNCERTAIN, sg.SAFE_STOP):
            key = (image_name, ss.corruption, ss.severity, ss.on, ss.strictness)
            if ss.last_logged != key:
                system.log_incident(result["status"], image=image_name, corruption=ss.corruption,
                                    severity=ss.severity, result=result,
                                    note=f"strictness {ss.strictness}")
                ss.last_logged = key

        if result and result["status"] != sg.HALTED:
            color = BLUE if result["status"] == sg.ANSWER else AMBER if result["status"] == sg.UNCERTAIN else RED
            st.markdown("#### Top 3 classes" + ("" if result["safeguard_on"] else
                                                " (43 sign classes, no 'not a sign' exit)"))
            prob_bars(result["top3"], color)
            if result["safeguard_on"]:
                tiles([("Confidence", f"{result['conf']:.0%}", f"needs ≥ {result['tau_conf']:.0%}"),
                       ("Agreement", f"{result['agreement']:.0%}", f"needs ≥ {result['tau_agree']:.0%}"),
                       ("Mutual info", f"{result['mutual_info']:.3f}", "high = passes disagree"),
                       ("Latency", f"{result['latency_ms']:.0f} ms", "20 passes")])
                st.markdown("#### How the 20 passes voted")
                st.image(vote_chart(result["votes"], result["pred"]), width=640)
            else:
                tiles([("Confidence", f"{result['conf']:.0%}", "no check"),
                       ("Latency", f"{result['latency_ms']:.0f} ms", "1 pass")])
                st.info("SafeGuard OFF: one pass, no uncertainty check. It always names a sign.")


# =========================================================================================
# Tab 2: Price of safety
# =========================================================================================
with tab_price:
    pos = results.get("price_of_safety")
    if (OUT / "price_of_safety.png").exists():
        st.image(str(OUT / "price_of_safety.png"), width="stretch")
    if pos:
        s = st.slider("Strictness (50 = calibrated)", 0, 100, 50, key="pos_strictness")
        curve = pos["curve"]
        c = st.columns(4)
        c[0].metric("Answers given (coverage)", f"{curve['coverage'][s]:.1%}")
        acc = curve["accuracy"][s]
        c[1].metric("Answers correct", "—" if acc is None else f"{acc:.2%}")
        c[2].metric("Confidence needed", f"{curve['tau_conf'][s]:.0%}")
        c[3].metric("Passes that must agree", f"{curve['tau_agree'][s]:.0%}")
        st.markdown(f"Measured on {pos['items']:,} test images (clean + fog, blur, noise, darkness). "
                    f"SafeGuard OFF answers 100% of them and gets **{pos['off']['accuracy']:.1%}** right.")


# =========================================================================================
# Tab 3: Confidence vs reality
# =========================================================================================
with tab_conf:
    if (OUT / "confidence_vs_accuracy.png").exists():
        st.image(str(OUT / "confidence_vs_accuracy.png"), width="stretch")
    curves = results.get("confidence_vs_accuracy")
    if curves:
        rows = []
        for kind in ("fog", "noise"):
            for sev, acc, conf in zip(curves[kind]["severity"], curves[kind]["accuracy"],
                                      curves[kind]["mean_confidence"]):
                if sev % 25 == 0:
                    rows.append({"corruption": kind, "severity": sev, "accuracy": f"{acc:.0%}",
                                 "average confidence": f"{conf:.0%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    cif = results.get("cifar10_test")
    if cif:
        st.markdown(f"On **{cif['images']:,} CIFAR-10 photos** (cats, cars, planes… no signs), "
                    f"SafeGuard OFF still names a traffic sign with **{cif['off_mean_confidence']:.0%}** "
                    f"average confidence. SafeGuard ON gives SAFE STOP **{cif['on_safe_stop_rate']:.1%}** "
                    f"of the time.")


# =========================================================================================
# Tab 4: Incident log
# =========================================================================================
with tab_log:
    log = pd.DataFrame(system.incidents)
    if log.empty:
        st.info("No incidents yet. Every UNCERTAIN, SAFE STOP and STOP event will appear here.")
    else:
        log["severity"] = log["severity"].astype("Int64")      # 70, not 70.0 (blank for STOP)
        counts = log["event"].value_counts()
        c = st.columns(4)
        for col, ev in zip(c, ["UNCERTAIN", "SAFE STOP", "STOP", "RESET"]):
            col.metric(ev, int(counts.get(ev, 0)))
        st.dataframe(log.iloc[::-1], hide_index=True, width="stretch")
        st.download_button("Download incident log (CSV)", log.to_csv(index=False).encode("utf-8"),
                           file_name="safeguard_incident_log.csv", mime="text/csv",
                           on_click="ignore")
