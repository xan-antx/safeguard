"""
check_demo.py - automated stage rehearsal. Loads the model WITHOUT the UI and checks that
every moment of the talk will behave exactly as planned.

    python check_demo.py                  checks outputs/
    python check_demo.py --dir outputs/quick

Prints a PASS/FAIL table and exits with code 1 if anything fails.
"""

import argparse
import json
import py_compile
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import safeguard as sg

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="outputs", help="folder made by prepare.py")
    out_dir = ROOT / parser.parse_args().dir

    checks = []                                      # list of (name, passed, details)

    def check(name, passed, details=""):
        checks.append((name, bool(passed), details))

    # ---- files exist ------------------------------------------------------------------
    needed = ["model.pt", "config.json", "presets.json", "results.json",
              "confidence_vs_accuracy.png", "price_of_safety.png"]
    missing = [f for f in needed if not (out_dir / f).exists()]
    check("output files exist", not missing, "missing: " + ", ".join(missing) if missing else "all 6")
    if "model.pt" in missing or "config.json" in missing or "presets.json" in missing:
        return report(checks)

    try:
        py_compile.compile(str(ROOT / "app.py"), doraise=True)
        check("app.py compiles", True)
    except Exception as e:                           # noqa: BLE001 - we want any error
        check("app.py compiles", False, str(e)[:60])

    system = sg.load_system(out_dir)
    presets = json.loads((out_dir / "presets.json").read_text())

    def preset_image(key):
        """Exactly what the app does when a preset button is pressed."""
        p = presets[key]
        with Image.open(out_dir / p["image"]) as im:
            img = sg.to_display(im)
        return sg.corrupt(img, p["corruption"], p["severity"]), p

    def fmt(r):
        if r["status"] == sg.HALTED:
            return "HALTED"
        return f"{r['status']}: {r['label']} ({r['conf']:.0%})"

    # ---- P1: clean stop sign -> ANSWER and correct -------------------------------------
    if "P1" in presets:
        img, p = preset_image("P1")
        r = system.predict(img, safeguard_on=True)
        check("P1 ON -> ANSWER and correct", r["status"] == sg.ANSWER and r["pred"] == p["true_label"],
              fmt(r))
    else:
        check("P1 ON -> ANSWER and correct", False, "preset P1 not found")

    # ---- P2: foggy sign: OFF wrong and confident, ON uncertain --------------------------
    if "P2" in presets:
        img, p = preset_image("P2")
        off = system.predict(img, safeguard_on=False)
        check("P2 OFF -> wrong with >= 80% confidence",
              off["pred"] != p["true_label"] and off["conf"] >= 0.8,
              f"true: {p['true_name']}; said {off['label']} ({off['conf']:.0%})")
        on = system.predict(img, safeguard_on=True)
        check("P2 ON -> UNCERTAIN", on["status"] == sg.UNCERTAIN,
              f"{fmt(on)}, agreement {on['agreement']:.0%}")
    else:
        check("P2 OFF -> wrong with >= 80% confidence", False, "preset P2 not found")
        check("P2 ON -> UNCERTAIN", False, "preset P2 not found")

    # ---- P3: cat/dog -> SAFE STOP -------------------------------------------------------
    if "P3" in presets:
        img, p = preset_image("P3")
        r = system.predict(img, safeguard_on=True)
        check("P3 ON -> SAFE STOP", r["status"] == sg.SAFE_STOP,
              f"P(not a sign) = {r['p_not_sign']:.0%}")
    else:
        check("P3 ON -> SAFE STOP", False, "preset P3 not found")

    # ---- STOP button really blocks the model ---------------------------------------------
    calls = []
    hook = system.model.register_forward_hook(lambda *a: calls.append(1))
    img, _ = preset_image("P1") if "P1" in presets else (np.zeros((64, 64, 3), np.float32), None)
    system.stop()
    halted_on = system.predict(img, safeguard_on=True)
    halted_off = system.predict(img, safeguard_on=False)
    check("STOP blocks prediction (model never called)",
          halted_on["status"] == sg.HALTED and halted_off["status"] == sg.HALTED
          and halted_on["pred"] is None and len(calls) == 0,
          f"status {halted_on['status']}, model calls while halted: {len(calls)}")
    system.reset()
    after = system.predict(img, safeguard_on=True)
    check("Reset restores prediction", after["status"] != sg.HALTED and len(calls) > 0, fmt(after))
    hook.remove()
    events = [i["event"] for i in system.incidents]
    check("STOP and RESET appear in incident log", "STOP" in events and "RESET" in events,
          ", ".join(events))

    # ---- determinism: same input twice -> same output -------------------------------------
    key = "P2" if "P2" in presets else "P1"
    img, _ = preset_image(key)
    a = sg.mc_dropout_probs(system.model, sg.to_model_input(img))
    b = sg.mc_dropout_probs(system.model, sg.to_model_input(img))
    r1, r2 = system.predict(img, True), system.predict(img, True)
    o1, o2 = system.predict(img, False), system.predict(img, False)
    same = (np.array_equal(a, b) and r1["status"] == r2["status"] and r1["conf"] == r2["conf"]
            and o1["conf"] == o2["conf"])
    check("same input twice -> same output", same, f"{key}: {fmt(r1)} both times" if same else "differs")

    # ---- noise corruption is repeatable too ------------------------------------------------
    n1 = sg.corrupt(img, "noise", 60)
    n2 = sg.corrupt(img, "noise", 60)
    check("noise corruption is repeatable", np.array_equal(n1, n2))

    return report(checks)


def report(checks):
    width = max(len(c[0]) for c in checks) + 2
    print("\nSafeGuard stage rehearsal")
    print("-" * (width + 60))
    for name, passed, details in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}} {details}")
    print("-" * (width + 60))
    failed = sum(not c[1] for c in checks)
    print(f"{len(checks) - failed}/{len(checks)} checks passed" + ("" if not failed else " - FIX BEFORE GOING ON STAGE"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
