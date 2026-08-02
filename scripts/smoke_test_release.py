"""
smoke_test_release.py -- run everything the public release ships, once, in order.

Why this exists: the analysis tools existed for weeks and were never run after a
training run, so every finding surfaced at push time instead of at run time.
This script is the thing to run before a release AND after any run worth
trusting. It does not judge results; it checks that each entry point starts,
finishes, and produces what it claims to produce.

What it covers, in order:
  1. a short training run                        -> import closure is complete
  2. the same run with the new learning flags    -> the flags actually take effect
  3. check_flog on both                          -> log is well-formed and real
  4. the four analysis entry points              -> they parse what train_baby wrote
  5. the renderers (video, reel) and sonify      -> the FLOG feeds the video path
  6. flog_server                                 -> the dashboard serves its page
  7. bridge_bittle_wifi --help                   -> imports resolve without hardware
  8. text scan                                   -> no German left in public output
  9. path scan                                   -> no private paths in the output

What it does NOT cover: anything needing the physical robot, and whether the
numbers are good. A PASS here means the release runs, not that the run learned.

Usage:
    py -3.11 scripts/smoke_test_release.py
    py -3.11 scripts/smoke_test_release.py --steps 200      # faster, less realistic
    py -3.11 scripts/smoke_test_release.py --skip-render    # skip the slow video part
    py -3.11 scripts/smoke_test_release.py --reuse <run-dir>  # no training, use this run
"""

__version__ = "0.1.0"

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

# Words that must not appear in the output of a public entry point. Kept short
# and specific: a broad German word list produces false hits on English text.
GERMAN_MARKERS = [
    "Schritt", "Fenster", "Deutung", "Verlauf", "Belohnung", "Gewicht",
    "Schwelle", "Feuerrate", "keine ", "nicht ", "wurde", "waehrend",
    "ueber ", "Laeufe", "Abschnitt", "Zaehl",
]
PRIVATE_MARKERS = ["mhflocke-work", "D:\\claude", "info@", "marc@"]


class Result:
    def __init__(self):
        self.rows = []      # (level, name, detail)
        self.t0 = time.time()

    def add(self, level, name, detail=""):
        self.rows.append((level, name, detail))
        mark = {"PASS": "  ok  ", "WARN": " warn ", "FAIL": " FAIL ", "SKIP": " skip "}[level]
        print(f"[{mark}] {name:32s} {detail}")

    @property
    def failed(self):
        return any(l == "FAIL" for l, _, _ in self.rows)


def run(cmd, timeout=1800, cwd=ROOT):
    """Run a command, capture output. Returns (rc, combined_output).

    PYTHONIOENCODING is forced to utf-8 because on Windows a piped stdout
    defaults to cp1252, and any entry point printing a non-ASCII character then
    dies with UnicodeEncodeError before doing anything. That is a real fault in
    the entry point, but it must not be what this harness measures.
    """
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        p = subprocess.run(cmd, cwd=cwd, timeout=timeout, env=env,
                           capture_output=True, text=True, errors="replace")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return -9, f"timeout after {timeout}s"
    except OSError as e:
        return -1, str(e)


def scan_text(out, ignore=()):
    """Return (german_hits, private_hits) found in a captured output.

    `ignore` removes strings we passed IN ourselves -- a tool echoing back the
    path it was given is not a leak, and flagging it buries the real hits.
    """
    for s in ignore:
        if s:
            out = out.replace(s, "<arg>")
    g = sorted({m for m in GERMAN_MARKERS if m in out})
    p = sorted({m for m in PRIVATE_MARKERS if m in out})
    return g, p


def newest_run(creature="bittle"):
    base = os.path.join(ROOT, "creatures", creature)
    if not os.path.isdir(base):
        return None
    runs = [os.path.join(base, d) for d in os.listdir(base)
            if os.path.isfile(os.path.join(base, d, "training_log.bin"))]
    return max(runs, key=os.path.getmtime) if runs else None


def train(rep, steps, extra=(), label="train"):
    cmd = [PY, "scripts/train_baby.py", "--creature-name", "bittle", "--neural-cpg",
           "--scene", "flat", "--hardware-sensors", "--fresh", "--snn-substeps", "10",
           "--no-balance", "--no-drives", "--steps", str(steps),
           "--log-every", "1", "--record-interval", "1", *extra]
    rc, out = run(cmd)
    if rc != 0:
        rep.add("FAIL", label, f"exit {rc} -- {out.strip().splitlines()[-1] if out.strip() else ''}")
        return None
    run_dir = newest_run()
    if not run_dir:
        rep.add("FAIL", label, "no run directory produced")
        return None
    rep.add("PASS", label, os.path.basename(run_dir))
    return run_dir


def tool(rep, name, args, run_dir, expect=None, timeout=900):
    """Run an entry point, check exit code, optional expected substring, and scan text.

    Paths are passed RELATIVE to the repo root, so an absolute path in the
    output means the tool built it itself -- which is what we want to catch.
    """
    rc, out = run([PY] + args)
    if rc != 0:
        last = out.strip().splitlines()[-1] if out.strip() else ""
        rep.add("FAIL", name, f"exit {rc} -- {last}")
        return out
    if expect and expect not in out:
        rep.add("FAIL", name, f"expected {expect!r} in output")
        return out
    g, p = scan_text(out, ignore=args + [ROOT])
    if p:
        rep.add("FAIL", name, "private path in output: " + ", ".join(p))
    elif g:
        rep.add("WARN", name, "possible German: " + ", ".join(g))
    else:
        rep.add("PASS", name)
    return out


def check_flags_took_effect(rep, out_plain, out_flags):
    """The two runs must differ in the way the flags promise, or the flags are dead."""
    def grab(txt, pat):
        m = re.search(pat, txt)
        return m.group(1) if m else None

    base_on = grab(out_flags, r"baseline active\s*:\s*(\w+)")
    blend = grab(out_flags, r"pe_blend\s*:\s*([\d.]+)")
    pos_plain = grab(out_plain, r"share positive\s*:\s*([\d.]+)")
    pos_flags = grab(out_flags, r"share positive\s*:\s*([\d.]+)")

    if base_on != "YES":
        rep.add("FAIL", "flag:reward-baseline", f"baseline reads {base_on!r} with the flag set")
    elif pos_plain is not None and pos_flags is not None and \
            float(pos_flags) <= float(pos_plain) + 1.0:
        rep.add("FAIL", "flag:reward-baseline",
                f"modulator positive share unchanged ({pos_plain} -> {pos_flags})")
    else:
        rep.add("PASS", "flag:reward-baseline", f"positive share {pos_plain} -> {pos_flags}")

    if blend is None or abs(float(blend) - 0.3) > 1e-6:
        rep.add("FAIL", "flag:pe-blend", f"pe_blend reads {blend!r}, expected 0.30")
    else:
        rep.add("PASS", "flag:pe-blend", "0.30 applied")


def serve_check(rep):
    """Start flog_server, fetch the dashboard page, stop it again."""
    try:
        import urllib.request
    except ImportError:
        rep.add("SKIP", "flog_server", "urllib unavailable")
        return
    proc = subprocess.Popen([PY, "flog_server.py"], cwd=ROOT,
                            env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace")
    try:
        page = None
        for _ in range(30):
            time.sleep(1)
            if proc.poll() is not None:
                out = proc.stdout.read() if proc.stdout else ""
                last = out.strip().splitlines()[-1] if out.strip() else ""
                rep.add("FAIL", "flog_server", f"exited early -- {last}")
                return
            try:
                with urllib.request.urlopen("http://127.0.0.1:5050/", timeout=2) as r:
                    page = r.read().decode("utf-8", "replace")
                break
            except Exception:
                continue
        if page is None:
            rep.add("FAIL", "flog_server", "no response on :5050 within 30s")
        elif "<html" not in page.lower():
            rep.add("FAIL", "flog_server", "response is not an HTML page")
        else:
            rep.add("PASS", "flog_server", f"served {len(page)} bytes")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--skip-render", action="store_true", help="skip video/reel/sonify")
    ap.add_argument("--skip-train", action="store_true", help="use existing runs")
    ap.add_argument("--reuse", default=None, help="run directory to test against")
    args = ap.parse_args()

    rep = Result()
    print(f"\nsmoke test -- repo {ROOT}\n" + "=" * 72)

    # --- 1/2. training runs -------------------------------------------------
    if args.reuse:
        plain = args.reuse
        flags = None
        rep.add("SKIP", "train", "reusing " + os.path.basename(plain))
    elif args.skip_train:
        plain = newest_run()
        flags = None
        rep.add("SKIP", "train", "using newest run")
    else:
        plain = train(rep, args.steps, label="train:defaults")
        flags = train(rep, args.steps,
                      extra=("--reward-baseline", "--pe-blend", "0.3"),
                      label="train:new-flags")
    if not plain:
        print("\nno usable run -- stopping")
        return 1

    log = os.path.relpath(os.path.join(plain, "training_log.bin"), ROOT)
    snn = os.path.relpath(os.path.join(plain, "snn_state.pt"), ROOT)

    # --- 3. log integrity ---------------------------------------------------
    tool(rep, "check_flog", ["scripts/check_flog.py", log], plain, expect="spike_raster")

    # --- 4. analysis entry points -------------------------------------------
    out_plain = tool(rep, "analyze_reward_gradient",
                     ["scripts/analyze_reward_gradient.py", log], plain)
    tool(rep, "analyze_substrate_health",
         ["scripts/analyze_substrate_health.py", log], plain)
    tool(rep, "analyze_locomotion", ["scripts/analyze_locomotion.py", log], plain)
    if os.path.isfile(os.path.join(ROOT, snn)):
        tool(rep, "diagnose_homeostasis", ["scripts/diagnose_homeostasis.py", snn], plain)
    else:
        rep.add("WARN", "diagnose_homeostasis", "no snn_state.pt in the run")

    # --- flags actually took effect ----------------------------------------
    if flags:
        out_flags = tool(rep, "analyze_reward_gradient:flags",
                         ["scripts/analyze_reward_gradient.py",
                          os.path.relpath(os.path.join(flags, "training_log.bin"), ROOT)],
                         flags)
        check_flags_took_effect(rep, out_plain, out_flags)

    # --- 5. video path ------------------------------------------------------
    if args.skip_render:
        rep.add("SKIP", "render", "--skip-render")
    else:
        tool(rep, "render_bittle", ["scripts/render_bittle.py", log], plain, timeout=3600)
        tool(rep, "render_insta_reel_bittle",
             ["scripts/render_insta_reel_bittle.py", log], plain, timeout=3600)
        tool(rep, "sonify_flog", ["scripts/sonify_flog.py", "--flog", log], plain, timeout=1800)

    # --- 6. dashboard -------------------------------------------------------
    serve_check(rep)

    # --- 7. hardware bridge (imports only) ---------------------------------
    rc, out = run([PY, "scripts/bridge_bittle_wifi.py", "--help"], timeout=120)
    if rc != 0:
        rep.add("FAIL", "bridge_bittle_wifi", f"--help exits {rc}")
    else:
        rep.add("PASS", "bridge_bittle_wifi", "imports resolve (no hardware tested)")

    # --- verdict ------------------------------------------------------------
    dt = time.time() - rep.t0
    n_fail = sum(1 for l, _, _ in rep.rows if l == "FAIL")
    n_warn = sum(1 for l, _, _ in rep.rows if l == "WARN")
    print("=" * 72)
    print(f"  {len(rep.rows)} checks, {n_fail} failed, {n_warn} warnings, {dt/60:.1f} min")
    print("  => " + ("FAIL" if n_fail else ("PASS (with warnings)" if n_warn else "PASS")))
    print()
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
