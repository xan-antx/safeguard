"""
app.py - the SafeGuard live demo (Streamlit).   Run:  streamlit run app.py

All the AI logic lives in safeguard.py; this file only draws the screen.

How a Streamlit app works: the whole script re-runs top to bottom on every click.
  * st.session_state  remembers things per browser tab (chosen image, slider values ...)
  * the cached SafeGuard object is shared, so the STOP flag even survives a page refresh
Look and feel: one CSS block (the "design system") + small helper functions that return HTML
for the parts Streamlit has no widget for (verdict banner, decision checklist, vote grid).
Nothing here touches the internet: system fonts, local images, charts drawn by Streamlit's
own bundled chart library.
"""

import base64
import io
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import safeguard as sg

OUT = Path(__file__).resolve().parent / "outputs"

# ---- colour tokens (same as the slides) --------------------------------------------------
BG, SURFACE, FG, MUTED = "#0E1217", "#151C25", "#F2EEE6", "#9AA5B4"
BLUE, AMBER, RED, GRID = "#5AA9E6", "#F2A900", "#E5484D", "#232C38"
OTHER_VOTES = ["#7C8796", "#A78BFA", "#4FD1C5", "#F472B6"]   # minority classes in the vote grid
STATUS_COLOR = {sg.ANSWER: BLUE, sg.UNCERTAIN: AMBER, sg.SAFE_STOP: RED, sg.HALTED: RED}

st.set_page_config(page_title="SafeGuard", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="collapsed")

# =========================================================================================
# Design system (CSS). Large type and strong contrast because it runs on a projector.
# =========================================================================================
CSS = """
<style>
:root { --bg:#0E1217; --surface:#151C25; --surface2:#1B2430; --line:#263040; --fg:#F2EEE6;
        --muted:#9AA5B4; --blue:#5AA9E6; --amber:#F2A900; --red:#E5484D; }
html { font-size: 18px; }
.stApp { background:
    radial-gradient(1200px 500px at 10% -10%, rgba(90,169,230,.10), transparent 60%),
    radial-gradient(900px 420px at 95% -5%, rgba(242,169,0,.08), transparent 60%), #0E1217; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.2rem !important; padding-bottom: 3rem; max-width: 1760px; }
[data-testid="stMarkdownContainer"] p { line-height: 1.5; }

/* ---------- hero header ---------- */
.hero { display:flex; align-items:center; justify-content:space-between; gap:1.5rem;
        flex-wrap:wrap; margin: .2rem 0 1.1rem; }
.brand { display:flex; align-items:center; gap:1.1rem; }
.logo { width:64px; height:64px; border-radius:18px; display:grid; place-items:center;
        background: linear-gradient(135deg, #F2A900 0%, #E5484D 100%);
        box-shadow: 0 10px 30px rgba(242,169,0,.25); }
.logo svg { width:36px; height:36px; stroke:#0E1217; }
.hero h1 { font-size: 2.35rem !important; font-weight: 800 !important; margin:0 !important;
           padding:0 !important; letter-spacing:-.02em; line-height:1.1 !important; color:var(--fg); }
.hero h1 .grad { background: linear-gradient(90deg, #F2A900, #FFD36B);
                 -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero .tag { color: var(--muted); font-size: 1.02rem; margin-top:.35rem; }
.chips { display:flex; gap:.5rem; flex-wrap:wrap; }
.chip { display:inline-flex; align-items:center; gap:.45rem; padding:.42rem .85rem;
        border-radius:999px; background: rgba(255,255,255,.04); border:1px solid var(--line);
        color: var(--fg); font-size:.9rem; font-weight:600; white-space:nowrap; }
.chip .dot { width:.6rem; height:.6rem; border-radius:50%; background:#3DD68C;
             box-shadow:0 0 0 4px rgba(61,214,140,.18); }
.chip.halt { border-color: rgba(229,72,77,.6); background: rgba(229,72,77,.12); }
.chip.halt .dot { background: var(--red); box-shadow:0 0 0 4px rgba(229,72,77,.25);
                  animation: blink 1s infinite; }

/* ---------- cards (st.container(key="card_...")) ---------- */
[class*="st-key-card_"] { background: linear-gradient(180deg, rgba(27,36,48,.92), rgba(21,28,37,.92));
    border: 1px solid var(--line); border-radius: 22px; padding: 1.25rem 1.4rem 1.35rem;
    box-shadow: 0 18px 50px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04); }
.st-key-card_controls { padding: .8rem 1rem !important; border-radius: 20px !important; }
.eyebrow { font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase;
           color: var(--muted); margin-bottom:.55rem; display:flex; align-items:center; gap:.5rem; }
.eyebrow .num { width:1.45rem; height:1.45rem; border-radius:7px; display:grid; place-items:center;
                background: rgba(242,169,0,.14); color: var(--amber); font-size:.78rem; }
.section-title { font-size:1.35rem; font-weight:800; margin: 0 0 .2rem; }
.lead { color: var(--muted); font-size: 1rem; margin-bottom: .8rem; }

/* ---------- buttons ---------- */
.stButton button, .stDownloadButton button { border-radius: 14px !important; min-height: 3.4rem;
    font-weight: 700; transition: transform .08s ease, box-shadow .2s ease, border-color .2s; }
.stButton button p, .stDownloadButton button p { font-size: 1.05rem !important; font-weight: 750; }
.stButton button:hover { transform: translateY(-1px); }
.stButton button:active { transform: translateY(1px) scale(.99); }
[class*="st-key-btn_P"] button { background: rgba(255,255,255,.035); border: 1px solid var(--line); }
[class*="st-key-btn_P"] button:hover { border-color: var(--amber); color: var(--fg); }
.st-key-stop_btn button { background: linear-gradient(180deg, #F0575C, #C9343A) !important;
    border: 2px solid rgba(255,255,255,.85) !important; min-height: 4.1rem;
    box-shadow: 0 10px 28px rgba(229,72,77,.45), inset 0 1px 0 rgba(255,255,255,.35); }
.st-key-stop_btn button p { color:#fff !important; font-size: 1.55rem !important; font-weight: 900 !important;
    letter-spacing: .16em; }
.st-key-stop_btn button:hover { box-shadow: 0 14px 36px rgba(229,72,77,.6); }
.st-key-reset_btn button { min-height: 4.1rem; background: transparent; border: 2px solid #3A4656 !important; }
.st-key-reset_btn button:hover { border-color: var(--fg) !important; }

/* ---------- SafeGuard ON / OFF switch (segmented control) ---------- */
.st-key-mode button { min-height: 4.1rem; padding: 0 1rem; flex: 1 1 0; }
.st-key-mode button p { font-size: 1.25rem !important; font-weight: 850 !important; letter-spacing:.04em; }
.st-key-card_controls .eyebrow { margin-bottom: .35rem; }
[class*="st-key-btn_P"] button { min-height: 4.1rem; }

/* ---------- tabs as a pill bar ---------- */
.stTabs [role="tablist"] { gap: .4rem; background: rgba(255,255,255,.03); padding: .35rem;
    border-radius: 16px; border: 1px solid var(--line); width: fit-content; margin-bottom: 1rem; }
[data-testid="stTab"] { border-radius: 12px !important; padding: .55rem 1.2rem !important; height:auto !important; }
[data-testid="stTab"] p { font-size: 1.12rem !important; font-weight: 750; }
[data-testid="stTab"][aria-selected="true"] { background: rgba(242,169,0,.14); }
[data-testid="stTab"][aria-selected="true"] p { color: var(--amber); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }

/* ---------- input image figure ---------- */
.fig { position: relative; width: 100%; max-width: 440px; aspect-ratio: 1/1; margin: .2rem auto .6rem;
       border-radius: 20px; overflow: hidden; border: 1px solid var(--line);
       box-shadow: 0 20px 50px rgba(0,0,0,.45); background:#000; }
.fig .main { width:100%; height:100%; object-fit:cover; display:block; }
.fig .fchip { position:absolute; top:12px; left:12px; padding:.35rem .75rem; border-radius:999px;
    background: rgba(14,18,23,.78); backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,.14);
    font-size:.85rem; font-weight:700; color: var(--fg); }
.fig .inset { position:absolute; right:12px; bottom:12px; padding:6px 6px 4px; border-radius:12px;
    background: rgba(14,18,23,.82); backdrop-filter: blur(8px); border:1px solid rgba(255,255,255,.14);
    text-align:center; }
.fig .inset img { width:78px; height:78px; image-rendering: pixelated; border-radius:6px; display:block; }
.fig .inset span { font-size:.66rem; font-weight:700; color: var(--muted); letter-spacing:.05em; }
.figcap { text-align:center; color: var(--muted); font-size:.88rem; margin-bottom: .8rem; }

/* ---------- verdict banner ---------- */
.verdict { display:flex; align-items:center; gap:1.1rem; padding: 1.15rem 1.35rem; border-radius: 20px;
    margin-bottom: 1rem; animation: pop .35s ease-out; position: relative; overflow: hidden; }
.verdict::after { content:""; position:absolute; inset:0; pointer-events:none;
    background: linear-gradient(120deg, rgba(255,255,255,.18), transparent 40%); }
.verdict .vicon { flex: 0 0 64px; height:64px; border-radius:18px; display:grid; place-items:center;
    background: rgba(0,0,0,.16); }
.verdict .vicon svg { width:38px; height:38px; }
.verdict .kicker { font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; opacity:.8; }
.verdict .title { font-size: 2.05rem; font-weight: 900; line-height:1.1; letter-spacing:-.01em; margin:.1rem 0 .25rem; }
.verdict .sub { font-size: 1.02rem; font-weight: 600; opacity:.92; }
.v-answer { background: linear-gradient(135deg, #73B8EE, #4A97D6); color:#0B1220;
            box-shadow: 0 16px 40px rgba(90,169,230,.30); }
.v-answer svg { stroke:#0B1220; }
.v-uncertain { background: linear-gradient(135deg, #FFC233, #E99F00); color:#1A1200;
               box-shadow: 0 16px 40px rgba(242,169,0,.30); }
.v-uncertain svg { stroke:#1A1200; }
.v-stop, .v-halted { background: linear-gradient(135deg, #EE5A5F, #C7363C); color:#fff;
                     box-shadow: 0 16px 40px rgba(229,72,77,.35); }
.v-stop svg, .v-halted svg { stroke:#fff; }
.v-halted { animation: pop .35s ease-out, glow 1.6s ease-in-out infinite; }

/* ---------- decision checklist ---------- */
.checks { display:flex; flex-direction:column; gap:.55rem; margin-bottom: 1rem; }
.chk { display:grid; grid-template-columns: 2.3rem 1fr 6.6rem; align-items:center; gap:.85rem;
       padding:.7rem .9rem; border-radius:14px; background: rgba(255,255,255,.03); border:1px solid var(--line); }
.chk .ico { width:2.3rem; height:2.3rem; border-radius:10px; display:grid; place-items:center; }
.chk .ico svg { width:1.25rem; height:1.25rem; stroke-width:3; }
.chk.ok .ico { background: rgba(61,214,140,.15); } .chk.ok .ico svg { stroke:#3DD68C; }
.chk.fail .ico { background: rgba(229,72,77,.16); } .chk.fail .ico svg { stroke:#FF6B70; }
.chk.skip { opacity:.5; } .chk.skip .ico { background: rgba(255,255,255,.06); } .chk.skip .ico svg { stroke: var(--muted); }
.chk .q { font-weight:750; font-size:1.02rem; margin-bottom:.4rem; }
.chk .q small { color: var(--muted); font-weight:600; font-size:.85rem; margin-left:.4rem; }
.meter { position:relative; height:.62rem; border-radius:999px; background:#0E141B; border:1px solid #222B37; }
.meter .fill { position:absolute; left:0; top:0; bottom:0; border-radius:999px; }
.meter .tick { position:absolute; top:-5px; bottom:-5px; width:3px; border-radius:2px; background: var(--fg); }
.chk .val { text-align:right; font-size:1.45rem; font-weight:850; font-variant-numeric: tabular-nums; }
.chk .val small { display:block; font-size:.74rem; color:var(--muted); font-weight:650; }

/* ---------- top-3 bars ---------- */
.bars { display:flex; flex-direction:column; gap:.6rem; }
.bar .lbl { display:flex; justify-content:space-between; font-size:1rem; font-weight:650; margin-bottom:.28rem; }
.bar .lbl b { font-variant-numeric: tabular-nums; }
.bar .trk { height:.8rem; border-radius:999px; background:#0E141B; border:1px solid #222B37; overflow:hidden; }
.bar .fil { height:100%; border-radius:999px; }

/* ---------- 20-pass vote grid ---------- */
.passes { display:grid; grid-template-columns: repeat(10, 1fr); gap:.38rem; margin:.2rem 0 .8rem; }
.pass { aspect-ratio: 1/1; border-radius:9px; display:grid; place-items:center; font-size:.74rem;
        font-weight:800; color: rgba(0,0,0,.6); box-shadow: inset 0 -2px 0 rgba(0,0,0,.18);
        animation: pop .3s ease-out both; }
.legend { display:flex; flex-direction:column; gap:.35rem; }
.legend div { display:flex; align-items:center; gap:.55rem; font-size:.95rem; }
.legend i { width:.85rem; height:.85rem; border-radius:4px; display:inline-block; flex:0 0 auto; }
.legend b { margin-left:auto; font-variant-numeric: tabular-nums; }
.single { padding:1rem; border-radius:14px; border:1px dashed #3A4656; color: var(--muted);
          text-align:center; font-weight:650; }

/* ---------- stat tiles ---------- */
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(9.5rem, 1fr)); gap:.6rem; margin-top:.9rem; }
.tile { background: rgba(255,255,255,.03); border:1px solid var(--line); border-radius:14px; padding:.7rem .9rem; }
.tile .k { font-size:.74rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); font-weight:800; }
.tile .v { font-size:1.7rem; font-weight:850; line-height:1.15; font-variant-numeric: tabular-nums; }
.tile .n { font-size:.8rem; color:var(--muted); }
.kpi { border-left: 4px solid var(--accent, var(--amber)); }
.kpi .v { font-size: 2.3rem; }

/* ---------- callouts ---------- */
.callout { border-radius:16px; padding:1rem 1.1rem; background: rgba(255,255,255,.035);
           border:1px solid var(--line); margin-bottom:.8rem; }
.callout .big { font-size:2.1rem; font-weight:900; line-height:1.05; }
.callout .txt { color: var(--muted); font-size:.98rem; margin-top:.25rem; }
.halted-note { display:flex; flex-direction:column; gap:.6rem; }
.halted-note div { display:flex; gap:.7rem; align-items:center; padding:.75rem .95rem; border-radius:14px;
                   background: rgba(229,72,77,.08); border:1px solid rgba(229,72,77,.3); font-weight:650; }
.halted-note svg { width:1.3rem; height:1.3rem; stroke: #FF6B70; flex: 0 0 auto; }
.empty { text-align:center; padding:2.4rem 1rem; color: var(--muted); }
.empty svg { width:3rem; height:3rem; stroke: var(--muted); margin-bottom:.6rem; }
.foot { margin-top:1.6rem; color:#6E7A8A; font-size:.85rem; text-align:center; }
.foot b { color: var(--muted); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: #121820; border-right:1px solid var(--line); }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] li { font-size: 1rem; }
.rule { display:flex; gap:.6rem; align-items:flex-start; margin:.45rem 0; font-size:.95rem; }
.rule span { flex:0 0 auto; padding:.1rem .5rem; border-radius:7px; font-size:.75rem; font-weight:800; }

@keyframes pop { from { opacity:0; transform: translateY(6px) scale(.985); } to { opacity:1; transform:none; } }
@keyframes glow { 0%,100% { box-shadow: 0 16px 40px rgba(229,72,77,.35); }
                  50% { box-shadow: 0 16px 60px rgba(229,72,77,.75), 0 0 0 6px rgba(229,72,77,.18); } }
@keyframes blink { 50% { opacity:.25; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---- small inline icons (SVG paths, drawn locally, no icon font download) ---------------
def icon(name):
    paths = {
        "shield": '<path d="M12 2 4 5v6c0 5 3.4 9.3 8 11 4.6-1.7 8-6 8-11V5l-8-3z"/><path d="m9 12 2 2 4-4"/>',
        "check": '<path d="M20 6 9 17l-5-5"/>',
        "x": '<path d="M18 6 6 18M6 6l12 12"/>',
        "minus": '<path d="M5 12h14"/>',
        "alert": '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
        "octagon": '<polygon points="7.9 2 16.1 2 22 7.9 22 16.1 16.1 22 7.9 22 2 16.1 2 7.9"/><path d="M15 9l-6 6M9 9l6 6"/>',
        "halt": '<circle cx="12" cy="12" r="10"/><rect x="8.5" y="8.5" width="7" height="7" rx="1"/>',
        "eye_off": '<path d="M3 3l18 18"/><path d="M10.6 10.6a2 2 0 0 0 2.8 2.8"/><path d="M9.9 5.1A10 10 0 0 1 12 5c5 0 9 4.5 10 7-.4 1-1.2 2.3-2.3 3.5M6.1 6.1C3.9 7.6 2.5 9.8 2 12c1 2.5 5 7 10 7 1.8 0 3.4-.5 4.8-1.3"/>',
        "lock": '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
        "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
        "inbox": '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.5 5h13L22 12v6a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2v-6z"/>',
    }
    return (f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
            f'stroke-linecap="round" stroke-linejoin="round">{paths[name]}</svg>')


def html(s):
    st.markdown(s, unsafe_allow_html=True)


def b64_png(pil):
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# =========================================================================================
# Load model + data files once (cached across re-runs)
# =========================================================================================
@st.cache_resource
def get_system():
    """The SafeGuard system (model + STOP flag + incident log). One shared copy."""
    return sg.load_system(OUT)


@st.cache_data
def load_json(name):
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
ss.setdefault("mode", "ON")
ss.setdefault("last_logged", None)


# ---- button callbacks (they run BEFORE the script re-runs) ------------------------------
def apply_preset(key):
    p = presets[key]
    ss.source, ss.corruption, ss.severity = key, p["corruption"], int(p["severity"])
    ss.last_logged = None


def on_upload():
    if ss.get("upload") is not None:
        ss.source, ss.severity, ss.last_logged = "upload", 0, None


def press_stop():
    system.stop(note="Human pressed the STOP button (EU AI Act Art. 14)")


def press_reset():
    system.reset(note="Human pressed Reset")
    ss.last_logged = None


def use_strictness(value):
    ss.strictness = value


safeguard_on = ss.mode == "ON"

# Styles that depend on the current state: highlight the chosen preset and colour the
# ON/OFF switch (blue = protected, grey = unprotected).
active = ss.source if ss.source in ("P1", "P2", "P3") else None
html(f"""<style>
{'.st-key-btn_' + active + ' button { border: 2px solid #F2A900 !important; background: rgba(242,169,0,.12) !important; box-shadow: 0 0 0 4px rgba(242,169,0,.12); }' if active else ''}
.st-key-mode button[aria-checked="true"], .st-key-mode button[aria-pressed="true"] {{
   background: {'linear-gradient(135deg,#73B8EE,#4A97D6)' if safeguard_on else 'linear-gradient(135deg,#566275,#3A4656)'} !important;
   border-color: {'#9CCDF3' if safeguard_on else '#7A8699'} !important;
   box-shadow: 0 8px 24px {'rgba(90,169,230,.35)' if safeguard_on else 'rgba(0,0,0,.35)'}; }}
.st-key-mode button[aria-checked="true"] *, .st-key-mode button[aria-pressed="true"] * {{
   color: {'#0B1220' if safeguard_on else '#F2EEE6'} !important; }}
{'.st-key-stop_btn button { animation: glow 1.6s ease-in-out infinite; }' if system.halted else ''}
{'.fig .main { filter: grayscale(1) brightness(.45); } .fig .fchip { background: rgba(229,72,77,.9) !important; color:#fff !important; }' if system.halted else ''}
</style>""")

# =========================================================================================
# Sidebar: strictness (backstage control)
# =========================================================================================
with st.sidebar:
    html('<div class="eyebrow">Control room</div><div class="section-title">Strictness</div>'
         '<div class="lead">How careful should SafeGuard be? 50 = calibrated in prepare.py.</div>')
    st.slider("Strictness", 0, 100, key="strictness", label_visibility="collapsed")
    tau_conf, tau_agree = sg.thresholds_for_strictness(ss.strictness, system.tau_conf_cal,
                                                       system.tau_agree_cal)
    need = int(np.ceil(tau_agree * sg.T_PASSES - 1e-9))
    html(f'<div class="tiles" style="margin-top:.2rem">'
         f'<div class="tile"><div class="k">Confidence</div><div class="v">≥ {tau_conf:.0%}</div></div>'
         f'<div class="tile"><div class="k">Agreement</div><div class="v">≥ {need}/20</div></div></div>')
    if ss.strictness != 50:
        st.button("Back to calibrated (50)", on_click=use_strictness, args=(50,), width="stretch")
    html(f"""<div style="margin-top:1.4rem" class="eyebrow">How SafeGuard decides</div>
    <div class="lead">The image goes through the network <b>20 times</b> with dropout ON
    (Monte Carlo dropout). Then, in this order:</div>
    <div class="rule"><span style="background:{RED};color:#fff">SAFE STOP</span><div>says "not a sign", or P(not a sign) ≥ 50%</div></div>
    <div class="rule"><span style="background:{BLUE};color:#0B1220">ANSWER</span><div>confident <b>and</b> the passes agree</div></div>
    <div class="rule"><span style="background:{AMBER};color:#1A1200">UNCERTAIN</span><div>otherwise → a human decides</div></div>
    <div style="margin-top:1.4rem" class="eyebrow">About</div>
    <div class="lead">Small CNN trained on clean GTSRB signs + CIFAR-10 photos as "not a sign"
    (outlier exposure). Never trained on fog, blur or noise. Runs fully offline on the CPU.</div>""")

# =========================================================================================
# Header + control bar (always visible above the tabs)
# =========================================================================================
acc = results.get("clean_accuracy_off_43")
status_chip = ('<span class="chip halt"><span class="dot"></span>System halted</span>' if system.halted
               else '<span class="chip"><span class="dot"></span>System live</span>')
html(f"""<div class="hero"><div class="brand"><div class="logo">{icon('shield')}</div><div>
<h1><span class="grad">SafeGuard</span>: an AI that knows when it doesn't know</h1>
<div class="tag">Fail-safe traffic-sign recognition · Monte Carlo dropout · human oversight (EU AI Act Art. 14)</div>
</div></div><div class="chips">{status_chip}
<span class="chip">⚡ CPU · offline</span>
{f'<span class="chip">🎯 {acc:.1%} clean accuracy</span>' if acc else ''}
<span class="chip">🔁 T = 20 passes</span></div></div>""")

with st.container(key="card_controls"):
    g_mode, g_presets, g_human = st.columns([1.05, 2.35, 1.45], vertical_alignment="bottom", gap="large")
    with g_mode:
        html('<div class="eyebrow">SafeGuard layer</div>')
        st.segmented_control("SafeGuard", ["OFF", "ON"], key="mode", required=True, width="stretch",
                             label_visibility="collapsed",
                             format_func=lambda m: (":material/shield: ON" if m == "ON"
                                                    else ":material/remove_moderator: OFF"))
    with g_presets:
        html('<div class="eyebrow">Demo scenarios</div>')
        labels = {"P1": ("Clean stop", ":material/traffic:"), "P2": ("Foggy sign", ":material/foggy:"),
                  "P3": ("Not a sign", ":material/pets:")}
        for col, key in zip(st.columns(3, gap="small"), ["P1", "P2", "P3"]):
            with col:
                p = presets.get(key)
                st.button(f"{key} · {labels[key][0]}" if p else f"{key} missing", key=f"btn_{key}",
                          icon=labels[key][1], disabled=p is None, on_click=apply_preset, args=(key,),
                          width="stretch",
                          help=f"{p['title']} — {p['source']}" if p else "prepare.py could not find this preset")
    with g_human:
        html('<div class="eyebrow">Human oversight</div>')
        h1, h2 = st.columns([1.55, 1], gap="small")
        with h1:
            st.button("STOP", key="stop_btn", icon=":material/front_hand:", on_click=press_stop,
                      width="stretch", help="Halts the system: no more predictions until Reset")
        with h2:
            st.button("Reset", key="reset_btn", icon=":material/restart_alt:", on_click=press_reset,
                      width="stretch")

st.write("")
tab_live, tab_price, tab_conf, tab_log = st.tabs(
    [":material/visibility: Live demo", ":material/balance: Price of safety",
     ":material/insights: Confidence vs reality", ":material/receipt_long: Incident log"])


# =========================================================================================
# HTML building blocks for the decision panel
# =========================================================================================
def verdict(kind, kicker, title, sub, ico):
    html(f'<div class="verdict v-{kind}"><div class="vicon">{icon(ico)}</div><div>'
         f'<div class="kicker">{kicker}</div><div class="title">{title}</div>'
         f'<div class="sub">{sub}</div></div></div>')


def check_row(state, question, hint, value, value_note, fill, tick, color):
    """One line of the decision checklist: icon, question, meter with threshold tick, value."""
    ico = {"ok": "check", "fail": "x", "skip": "minus"}[state]
    tick_html = f'<div class="tick" style="left:calc({tick * 100:.1f}% - 1px)"></div>' if tick is not None else ""
    return (f'<div class="chk {state}"><div class="ico">{icon(ico)}</div><div>'
            f'<div class="q">{question}<small>{hint}</small></div>'
            f'<div class="meter"><div class="fill" style="width:{fill * 100:.1f}%;background:{color}"></div>{tick_html}</div>'
            f'</div><div class="val">{value}<small>{value_note}</small></div></div>')


def checklist(r):
    """The decision rule, step by step, with each measured value against its threshold."""
    if not r["safeguard_on"]:
        rows = [check_row("skip", "Is it a traffic sign?", "not checked", "—", "SafeGuard OFF", 0, None, MUTED),
                check_row("skip", "Is the model confident?", "not checked", f"{r['conf']:.0%}", "no threshold",
                          r["conf"], None, MUTED),
                check_row("skip", "Do 20 looks agree?", "not checked", "1 look", "dropout off", 0, None, MUTED)]
        return '<div class="checks">' + "".join(rows) + "</div>"
    is_sign = r["status"] != sg.SAFE_STOP
    conf_ok = r["conf"] >= r["tau_conf"]
    agree_ok = r["agreement"] >= r["tau_agree"]
    k = round(r["agreement"] * sg.T_PASSES)
    need = int(np.ceil(r["tau_agree"] * sg.T_PASSES - 1e-9))
    later = (lambda ok: "ok" if ok else "fail") if is_sign else (lambda ok: "skip")
    rows = [
        check_row("ok" if is_sign else "fail", "Is it a traffic sign?", "P(not a sign) must stay below 50%",
                  f"{r['p_not_sign']:.0%}", "not a sign", r["p_not_sign"], 0.5, RED if not is_sign else "#3DD68C"),
        check_row(later(conf_ok), "Is the model confident?", f"needs ≥ {r['tau_conf']:.0%}",
                  f"{r['conf']:.0%}", "confidence", r["conf"], r["tau_conf"], BLUE if conf_ok else AMBER),
        check_row(later(agree_ok), "Do the 20 passes agree?", f"needs ≥ {need}/20",
                  f"{k}/20", "agreement", r["agreement"], r["tau_agree"], BLUE if agree_ok else AMBER),
    ]
    return '<div class="checks">' + "".join(rows) + "</div>"


def top3_bars(top3, color):
    rows = "".join(f'<div class="bar"><div class="lbl"><span>{n}</span><b>{p:.0%}</b></div>'
                   f'<div class="trk"><div class="fil" style="width:{max(p, .005) * 100:.1f}%;'
                   f'background:{color if i == 0 else "#4A5566"}"></div></div></div>'
                   for i, (n, p) in enumerate(top3))
    return f'<div class="bars">{rows}</div>'


def vote_grid(r, color):
    """20 squares = 20 forward passes, coloured by what each pass said."""
    order = [i for i in np.argsort(r["votes"])[::-1] if r["votes"][i] > 0]
    palette, k = {}, 0
    for cls in order:
        if cls == r["pred"]:
            palette[cls] = color
        elif cls == sg.NOT_SIGN:
            palette[cls] = RED if color != RED else "#FF9A9D"
        else:
            palette[cls] = OTHER_VOTES[k % len(OTHER_VOTES)]
            k += 1
    cells = "".join(f'<div class="pass" title="Pass {i + 1}: {sg.CLASS_NAMES[v]}" '
                    f'style="background:{palette[v]};animation-delay:{i * 18}ms">{i + 1}</div>'
                    for i, v in enumerate(r["pass_preds"]))
    legend = "".join(f'<div><i style="background:{palette[c]}"></i>{sg.CLASS_NAMES[c]}<b>{r["votes"][c]}</b></div>'
                     for c in order[:5])
    return f'<div class="passes">{cells}</div><div class="legend">{legend}</div>'


def tiles(items):
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div><div class="n">{n}</div></div>'
        for k, v, n in items) + "</div>"


def chart_theme(chart, height=430):
    return (chart.properties(height=height)
            .configure(background="transparent", font="Segoe UI")
            .configure_axis(labelColor=MUTED, titleColor=FG, gridColor=GRID, domainColor="#3A4656",
                            tickColor="#3A4656", labelFontSize=15, titleFontSize=16, titlePadding=12,
                            titleFontWeight=700)
            .configure_legend(labelColor=FG, titleColor=FG, labelFontSize=15, orient="top",
                              symbolStrokeWidth=5, symbolSize=300)
            .configure_view(stroke=None))


# =========================================================================================
# Tab 1: Live demo
# =========================================================================================
with tab_live:
    left, right = st.columns([1, 1.32], gap="large")

    # ---------------- left: the camera image -----------------------------------------------
    with left:
        with st.container(key="card_input"):
            clean, image_name, short_name = None, "", ""
            if ss.source == "upload" and ss.get("upload") is not None:
                try:
                    with Image.open(ss.upload) as im:
                        clean = sg.to_display(im)
                    image_name = short_name = f"upload: {ss.upload.name}"
                except Exception:                      # noqa: BLE001 - never crash on stage
                    st.error("Could not read that file as an image. Showing P1 instead.")
                    ss.source = "P1" if "P1" in presets else None
            if clean is None and ss.source in presets:
                clean = load_preset_image(presets[ss.source]["image"])
                image_name = f"{ss.source} ({presets[ss.source]['source']})"
                short_name = f"{ss.source} · {presets[ss.source]['source']}"

            html('<div class="eyebrow"><span class="num">1</span>Camera input</div>')
            if clean is None:
                html(f'<div class="empty">{icon("inbox")}<div>No image yet: upload one below.</div></div>')
            else:
                shown = sg.corrupt(clean, ss.corruption, ss.severity)
                tiny = sg.to_model_input(shown)[0].permute(1, 2, 0).numpy()
                big = Image.fromarray(sg.to_uint8(shown)).resize((440, 440), Image.LANCZOS)
                chip = "Clean image" if ss.severity == 0 else f"{ss.corruption.capitalize()} · {ss.severity}%"
                if system.halted:
                    chip = "🔒 Input ignored — system halted"
                html(f'<div class="fig"><img class="main" src="data:image/png;base64,{b64_png(big)}">'
                     f'<div class="fchip">{chip}</div><div class="inset">'
                     f'<img src="data:image/png;base64,{b64_png(Image.fromarray(sg.to_uint8(tiny)))}">'
                     f'<span>AI SEES 32×32</span></div></div><div class="figcap">{short_name}</div>')

            st.segmented_control("Corruption", sg.CORRUPTIONS, key="corruption", required=True,
                                 width="stretch",
                                 format_func=lambda k: {"fog": ":material/foggy: Fog",
                                                        "blur": ":material/blur_on: Blur",
                                                        "noise": ":material/grain: Noise",
                                                        "darkness": ":material/dark_mode: Dark"}[k])
            st.slider("Severity", 0, 100, key="severity", format="%d%%")
            with st.expander(":material/upload: Try your own photo"):
                st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "ppm", "webp"],
                                 key="upload", on_change=on_upload, label_visibility="collapsed")

    # ---------------- right: the decision ------------------------------------------------
    with right:
        with st.container(key="card_decision"):
            html('<div class="eyebrow"><span class="num">2</span>Decision</div>')
            result = None
            if clean is not None:
                try:
                    result = system.predict(shown, safeguard_on=safeguard_on, strictness=ss.strictness)
                except Exception as e:                 # noqa: BLE001 - show, don't crash
                    verdict("stop", "Error", "ERROR → handing control to a human", str(e)[:120], "alert")

            if result is None:
                pass
            elif result["status"] == sg.HALTED:
                verdict("halted", "Human oversight · Art. 14", "SYSTEM HALTED — SAFE STATE",
                        "A human pressed STOP. Press Reset to resume.", "halt")
                stop_time = next((i["timestamp"][-8:] for i in reversed(system.incidents)
                                  if i["event"] == "STOP"), "")
                html(f'<div class="halted-note">'
                     f'<div>{icon("lock")}The model is not being called — every input is refused.</div>'
                     f'<div>{icon("eye_off")}Presets, sliders and uploads change nothing until Reset.</div>'
                     f'<div>{icon("file")}STOP was written to the incident log at {stop_time}.</div></div>')
            else:
                st_ = result["status"]
                color = STATUS_COLOR[st_]
                k = round(result["agreement"] * sg.T_PASSES) if safeguard_on else None
                if not safeguard_on:
                    verdict("answer", "SafeGuard OFF · no safety checks", f"ANSWER: {result['label']}",
                            f"One pass, dropout off · {result['conf']:.0%} confident · it always names a sign",
                            "check")
                elif st_ == sg.ANSWER:
                    verdict("answer", "SafeGuard ON · safe to act", f"ANSWER: {result['label']}",
                            f"Confident ({result['conf']:.0%}) and {k}/20 passes agree", "check")
                elif st_ == sg.UNCERTAIN:
                    verdict("uncertain", "SafeGuard ON · not sure enough",
                            "UNCERTAIN → handing control to a human",
                            f"Best guess “{result['label']}” ({result['conf']:.0%}), only {k}/20 passes agree",
                            "alert")
                else:
                    verdict("stop", "SafeGuard ON · out of scope", "SAFE STOP: this is not a traffic sign",
                            f"P(not a sign) = {result['p_not_sign']:.0%} · the system refuses to act",
                            "octagon")

                html(checklist(result))
                a, b = st.columns([1, 1], gap="medium")
                with a:
                    html('<div class="eyebrow">Top 3 classes</div>' + top3_bars(result["top3"], color))
                with b:
                    if safeguard_on:
                        html('<div class="eyebrow">How the 20 passes voted</div>' + vote_grid(result, color))
                    else:
                        html('<div class="eyebrow">How many looks?</div><div class="single">'
                             'Just 1 pass, dropout off.<br>No second opinion, no doubt, no exit.</div>')
                if safeguard_on:
                    html(tiles([("Mutual info", f"{result['mutual_info']:.3f}", "high = passes disagree"),
                                ("Latency", f"{result['latency_ms']:.0f} ms", "20 passes on CPU"),
                                ("Strictness", f"{ss.strictness}", "50 = calibrated")]))
                else:
                    html(tiles([("Latency", f"{result['latency_ms']:.0f} ms", "1 pass on CPU"),
                                ("Safety layer", "Off", "turn ON to see the checks")]))

            # incident log: record each new UNCERTAIN / SAFE STOP once (not on every re-run)
            if result and result["status"] in (sg.UNCERTAIN, sg.SAFE_STOP):
                key = (image_name, ss.corruption, ss.severity, ss.mode, ss.strictness)
                if ss.last_logged != key:
                    system.log_incident(result["status"], image=image_name, corruption=ss.corruption,
                                        severity=ss.severity, result=result,
                                        note=f"strictness {ss.strictness}")
                    ss.last_logged = key


# =========================================================================================
# Tab 2: Price of safety (interactive risk-coverage curve)
# =========================================================================================
with tab_price:
    pos = results.get("price_of_safety")
    if not pos:
        st.info("Run prepare.py to create the evaluation results.")
    else:
        curve = pos["curve"]
        left, right = st.columns([2.1, 1], gap="large")
        with right:
            with st.container(key="card_pos_side"):
                html('<div class="eyebrow">Try it</div><div class="section-title">Strictness</div>'
                     '<div class="lead">Move the slider: stricter means fewer answers, but more of them are right.</div>')
                s = st.slider("Strictness (50 = calibrated)", 0, 100, 50, key="pos_strictness",
                              label_visibility="collapsed")
                acc_s = curve["accuracy"][s]
                html(f'<div class="callout"><div class="big" style="color:{AMBER}">{curve["coverage"][s]:.1%}</div>'
                     f'<div class="txt">of images get an answer (coverage)</div></div>'
                     f'<div class="callout"><div class="big" style="color:{BLUE}">'
                     f'{"—" if acc_s is None else f"{acc_s:.2%}"}</div>'
                     f'<div class="txt">of those answers are correct</div></div>'
                     + tiles([("Confidence ≥", f"{curve['tau_conf'][s]:.0%}", ""),
                              ("Agreement ≥", f"{curve['tau_agree'][s]:.0%}", "")]))
                st.write("")
                st.button("Use this strictness in the live demo", on_click=use_strictness, args=(s,),
                          icon=":material/tune:", width="stretch")
        with left:
            with st.container(key="card_pos_chart"):
                html(f'<div class="eyebrow">Risk–coverage curve · {pos["items"]:,} test images</div>'
                     f'<div class="section-title">The price of safety</div>'
                     f'<div class="lead">SafeGuard OFF answers everything and is right '
                     f'<b style="color:{FG}">{pos["off"]["accuracy"]:.1%}</b> of the time. '
                     f'SafeGuard ON trades some answers for being right.</div>')
                df = pd.DataFrame({"strictness": curve["strictness"],
                                   "coverage": [100 * c for c in curve["coverage"]],
                                   "accuracy": [None if a is None else 100 * a for a in curve["accuracy"]]}).dropna()
                lo = max(0, min(df["accuracy"].min(), 100 * pos["off"]["accuracy"]) - 3)
                x = alt.X("coverage:Q", title="Share of images answered (%)",
                          scale=alt.Scale(domain=[0, 104], nice=False), axis=alt.Axis(values=list(range(0, 101, 10))))
                y = alt.Y("accuracy:Q", title="Answers correct (%)", scale=alt.Scale(domain=[lo, 100.8], nice=False))
                tip = [alt.Tooltip("strictness:Q", title="Strictness"),
                       alt.Tooltip("coverage:Q", title="Answered %", format=".1f"),
                       alt.Tooltip("accuracy:Q", title="Correct %", format=".2f")]
                line = alt.Chart(df).mark_line(color=AMBER, strokeWidth=5).encode(x=x, y=y, tooltip=tip)
                target = alt.Chart(pd.DataFrame({"y": [97]})).mark_rule(
                    color=MUTED, strokeDash=[8, 6], strokeWidth=2).encode(y="y:Q")
                target_lbl = alt.Chart(pd.DataFrame({"x": [3], "y": [97], "t": ["target: 97% correct"]})).mark_text(
                    align="left", dy=-12, color=MUTED, fontSize=15, fontWeight=600).encode(x="x:Q", y="y:Q", text="t:N")
                off = pd.DataFrame({"coverage": [100.0], "accuracy": [100 * pos["off"]["accuracy"]],
                                    "t": [f"SafeGuard OFF · {pos['off']['accuracy']:.1%} right"]})
                off_pt = alt.Chart(off).mark_square(size=520, color=BLUE).encode(x="coverage:Q", y="accuracy:Q")
                off_lbl = alt.Chart(off).mark_text(align="right", dx=-18, dy=-16, color=BLUE, fontSize=16,
                                                   fontWeight=700).encode(x="coverage:Q", y="accuracy:Q", text="t:N")
                now = df[df["strictness"] == s]
                if now.empty:
                    now = df.iloc[[-1]]
                now = now.assign(t=f"you are here · strictness {s}")
                now_pt = alt.Chart(now).mark_circle(size=900, color=FG, stroke=AMBER, strokeWidth=5,
                                                    opacity=1).encode(x="coverage:Q", y="accuracy:Q", tooltip=tip)
                now_lbl = alt.Chart(now).mark_text(align="right", dx=-26, dy=36, color=FG, fontSize=16,
                                                   fontWeight=800).encode(x="coverage:Q", y="accuracy:Q", text="t:N")
                st.altair_chart(chart_theme(line + target + target_lbl + off_pt + off_lbl + now_pt + now_lbl, 470),
                                theme=None, width="stretch")
                if (OUT / "price_of_safety.png").exists():
                    st.download_button("Slide version (1920×1080 PNG)", (OUT / "price_of_safety.png").read_bytes(),
                                       "price_of_safety.png", "image/png", icon=":material/download:",
                                       on_click="ignore")


# =========================================================================================
# Tab 3: Confidence vs reality
# =========================================================================================
with tab_conf:
    curves = results.get("confidence_vs_accuracy")
    if not curves:
        st.info("Run prepare.py to create the evaluation results.")
    else:
        left, right = st.columns([2.1, 1], gap="large")
        with left:
            with st.container(key="card_conf_chart"):
                html('<div class="eyebrow">SafeGuard OFF · 2,000 test images per point</div>'
                     '<div class="section-title">Being sure is not being right</div>'
                     '<div class="lead">The plain model stays confident while its accuracy falls. '
                     'The red band is the overconfidence gap.</div>')
                kind = st.segmented_control("Corruption type", list(curves), default="fog", required=True,
                                            key="conf_kind", label_visibility="collapsed",
                                            format_func=lambda k: k.capitalize())
                c = curves[kind]
                wide = pd.DataFrame({"severity": c["severity"], "Accuracy": [100 * a for a in c["accuracy"]],
                                     "Average confidence": [100 * v for v in c["mean_confidence"]]})
                long = wide.melt("severity", var_name="series", value_name="pct")
                x = alt.X("severity:Q", title=f"{kind.capitalize()} severity", scale=alt.Scale(domain=[0, 100]))
                gap = alt.Chart(wide).mark_area(color=RED, opacity=.22).encode(
                    x=x, y=alt.Y("Accuracy:Q", title="%", scale=alt.Scale(domain=[0, 100])),
                    y2="Average confidence:Q")
                lines = alt.Chart(long).mark_line(strokeWidth=5, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
                    x=x, y=alt.Y("pct:Q", title="%"),
                    color=alt.Color("series:N", title=None,
                                    scale=alt.Scale(domain=["Accuracy", "Average confidence"], range=[AMBER, BLUE])),
                    tooltip=[alt.Tooltip("severity:Q"), alt.Tooltip("series:N"), alt.Tooltip("pct:Q", format=".1f")])
                layers = gap + lines
                if ss.corruption == kind:          # show where the live-demo slider is right now
                    rule = alt.Chart(pd.DataFrame({"s": [ss.severity], "t": ["live slider"]}))
                    layers += rule.mark_rule(color=FG, strokeDash=[4, 4], strokeWidth=2).encode(x="s:Q")
                    layers += rule.mark_text(color=FG, dy=-8, fontSize=14, fontWeight=700, align="left", dx=6
                                             ).encode(x="s:Q", y=alt.value(12), text="t:N")
                st.altair_chart(chart_theme(layers, 430), theme=None, width="stretch")
                if (OUT / "confidence_vs_accuracy.png").exists():
                    st.download_button("Slide version (1920×1080 PNG)",
                                       (OUT / "confidence_vs_accuracy.png").read_bytes(),
                                       "confidence_vs_accuracy.png", "image/png", icon=":material/download:",
                                       on_click="ignore")
        with right:
            with st.container(key="card_conf_side"):
                gaps = np.array(c["mean_confidence"]) - np.array(c["accuracy"])
                i = int(np.argmax(gaps))
                html(f'<div class="eyebrow">Biggest gap · {kind}</div>'
                     f'<div class="callout"><div class="big"><span style="color:{BLUE}">{c["mean_confidence"][i]:.0%}</span>'
                     f' <span style="color:{MUTED};font-size:1.3rem">sure</span> · '
                     f'<span style="color:{AMBER}">{c["accuracy"][i]:.0%}</span> '
                     f'<span style="color:{MUTED};font-size:1.3rem">right</span></div>'
                     f'<div class="txt">at {kind} severity {c["severity"][i]}: on average '
                     f'{gaps[i] * 100:.0f} points more confident than it should be.</div></div>')
                cif = results.get("cifar10_test")
                if cif:
                    html(f'<div class="eyebrow" style="margin-top:1rem">Photos with no sign at all</div>'
                         f'<div class="callout"><div class="big" style="color:{BLUE}">{cif["off_mean_confidence"]:.0%}</div>'
                         f'<div class="txt">average confidence of SafeGuard OFF when it calls {cif["images"]:,} '
                         f'CIFAR-10 photos (cats, cars, planes…) a traffic sign</div></div>'
                         f'<div class="callout"><div class="big" style="color:{RED}">{cif["on_safe_stop_rate"]:.0%}</div>'
                         f'<div class="txt">of the same photos get a SAFE STOP from SafeGuard ON</div></div>')


# =========================================================================================
# Tab 4: Incident log (honest incident reporting)
# =========================================================================================
with tab_log:
    with st.container(key="card_log"):
        html('<div class="eyebrow">Honest incident reporting</div><div class="section-title">Incident log</div>'
             '<div class="lead">Every time SafeGuard stepped back, or a human stepped in, it is written down here.</div>')
        log = pd.DataFrame(system.incidents)
        counts = log["event"].value_counts() if not log.empty else {}
        html('<div class="tiles" style="margin-bottom:1rem">' + "".join(
            f'<div class="tile kpi" style="--accent:{col}"><div class="k">{ev}</div>'
            f'<div class="v">{int(counts.get(ev, 0))}</div><div class="n">{note}</div></div>'
            for ev, col, note in [("UNCERTAIN", AMBER, "handed to a human"), ("SAFE STOP", RED, "not a sign"),
                                  ("STOP", RED, "human pressed STOP"), ("RESET", BLUE, "human resumed")]) + "</div>")
        if log.empty:
            html(f'<div class="empty">{icon("inbox")}<div>No incidents yet. Every UNCERTAIN, SAFE STOP, '
                 f'STOP and RESET will appear here.</div></div>')
        else:
            log["severity"] = log["severity"].astype("Int64")          # 70, not 70.0
            # a copy for the screen only: newest first, a coloured marker per event, blanks not "None"
            shown_log = log.iloc[::-1].reset_index(drop=True).copy()
            marker = {"UNCERTAIN": "🟠 UNCERTAIN", "SAFE STOP": "🔴 SAFE STOP", "STOP": "⛔ STOP",
                      "RESET": "🔵 RESET"}
            shown_log["event"] = shown_log["event"].map(lambda e: marker.get(e, e))
            for col in ["confidence", "agreement", "mutual_info", "p_not_sign"]:
                shown_log[col] = pd.to_numeric(shown_log[col], errors="coerce")
            for col in ["best_guess", "image", "corruption", "note"]:
                shown_log[col] = shown_log[col].fillna("")
            cc = st.column_config
            st.dataframe(shown_log, hide_index=True, width="stretch", row_height=42, placeholder="—",
                         column_config={
                             "timestamp": cc.TextColumn("Time"), "event": cc.TextColumn("Event"),
                             "image": cc.TextColumn("Image", width="medium"),
                             "corruption": cc.TextColumn("Corruption"), "severity": cc.NumberColumn("Severity"),
                             "best_guess": cc.TextColumn("Best guess", width="medium"),
                             "confidence": cc.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent"),
                             "agreement": cc.ProgressColumn("Agreement", min_value=0, max_value=1, format="percent"),
                             "mutual_info": cc.NumberColumn("Mutual info", format="%.3f"),
                             "p_not_sign": cc.NumberColumn("P(not a sign)", format="percent"),
                             "note": cc.TextColumn("Note", width="medium")})
            st.download_button("Download incident log (CSV)", log.to_csv(index=False).encode("utf-8"),
                               file_name="safeguard_incident_log.csv", mime="text/csv",
                               icon=":material/download:", on_click="ignore", type="primary")

html('<div class="foot"><b>SafeGuard</b> · runs 100% offline on a laptop CPU · MC dropout (T = 20) · '
     'outlier exposure (Hendrycks et al., 2019) · human oversight: EU AI Act Article 14</div>')
