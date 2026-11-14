"""Fills Samsung's 9-slide submission template from results/summary.json.

    python deck/build_deck.py      # writes deck/PathFinder_SRMIST_CaliforniaBurrito.pptx
"""

import json
from copy import deepcopy
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
S = json.loads((ROOT / "results" / "summary.json").read_text())

BG = RGBColor(0x0A, 0x0E, 0x1A)
PANEL = RGBColor(0x12, 0x1A, 0x2E)
LINE = RGBColor(0x23, 0x2E, 0x4A)
INK = RGBColor(0xEE, 0xF1, 0xF6)
MUTED = RGBColor(0x9A, 0xA6, 0xBC)
CYAN = RGBColor(0x00, 0xB4, 0xD8)
GREEN = RGBColor(0x2E, 0xD3, 0x9A)
ORANGE = RGBColor(0xF0, 0x8A, 0x34)
RED = RGBColor(0xF3, 0x6B, 0x5F)
GREY = RGBColor(0x6B, 0x75, 0x8A)
FONT = "Calibri"
TOP = Inches(1.05)          # content starts below the template's header band


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, opts) tuples."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(6)
        parts = [(para, {})] if isinstance(para, str) else para
        for t, o in parts:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = FONT
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
    return tb


def card(slide, x, y, w, h, fill=PANEL, name=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    s.adjustments[0] = 0.06
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = LINE
    s.line.width = Pt(0.75)
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def dot(slide, x, y, d, color, label):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name = FONT
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = BG
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return s


def set_header(slide, new):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            runs = sh.text_frame.paragraphs[0].runs
            runs[0].text = new
            for r in runs[1:]:
                r.text = ""
            return


def sc(name, m):
    return S["scenarios"][name][m]["ate_m"]


def score_line():
    out = {"win": 0, "tie": 0, "loss": 0}
    for v in S["scenarios"].values():
        o, n = v["ours"]["ate_m"], v["naive_fusion"]["ate_m"]
        tol = 0.05 * n + 0.005
        out["win" if o < n - tol else "loss" if o > n + tol else "tie"] += 1
    return (f"{out['win']} better, {out['tie']} level, {out['loss']} worse out of {len(S['scenarios'])} "
            f"scenarios on {len(S['sequences'])} recordings, all in the README")


def style_chart(chart, colors):
    chart.font.name = FONT
    chart.font.size = Pt(10)
    chart.font.color.rgb = MUTED
    for s, c in zip(chart.plots[0].series, colors):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = c
        s.format.line.color.rgb = c


def main():
    prs = Presentation(HERE / "template.pptx")
    sl = prs.slides
    sw = prs.slide_width

    # ---------- 1. Title
    s = sl[0]
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text.strip()
        rep = {"Track No": "Track 3  |  Path Finder",
               "XR Hackathon": "XR Hackathon",
               "Team Name": "Team California Burrito",
               "College Name": "SRM IST, Kattankulathur"}
        if t in rep:
            sh.text_frame.paragraphs[0].runs[0].text = rep[t]
        elif "Team Name" in t and "College" in t:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.text = rep.get(r.text.strip(), r.text)
        elif t == "Members Details":
            sh.text_frame.paragraphs[0].runs[0].text = "Members"
            sh.top = Emu(sh.top)
    text(s, Inches(5.6), Inches(2.86), Inches(3.0), Inches(1.4),
         ["Rehaan Ajani (lead)", "Nikhil Solanki", "Debadrita Haldar", "Mayank Yadav"],
         size=12, color=INK, name="Members list")
    text(s, Inches(0.55), Inches(3.55), Inches(5.0), Inches(1.0),
         [[("Trajectory estimation that finds the failing sensor and keeps tracking", {"size": 15, "bold": True, "color": CYAN})],
          [("B.Tech CSE (AI/ML), 2nd year  |  github.com/rehaanajani/pathfinder-xr", {"size": 11, "color": MUTED})]],
         name="Tagline")

    # ---------- 2. Problem
    s = sl[1]
    imu10 = 1.4
    text(s, Inches(0.5), TOP, Inches(5.2), Inches(1.6),
         [[("A headset only knows where it is while its sensors tell the truth", {"size": 22, "bold": True})],
          [("Inside-out tracking fuses cameras with IMUs. Each fails in its own way, and today's fusion "
            "trusts every reading until the virtual world visibly slides away.", {"size": 13, "color": MUTED})]],
         name="Problem statement")
    rows = [("Camera", "Darkness, motion blur, blank walls, lens covered, locks onto a moving object", ORANGE),
            ("IMU", "Bias, drift with heat, saturation on fast turns, spikes, stuck or dead sensor", RED),
            ("Both", "Common-mode vibration hits every IMU at once while the camera is blind", CYAN)]
    for i, (k, v, c) in enumerate(rows):
        y = Inches(2.75 + i * 0.85)
        card(s, Inches(0.5), y, Inches(5.2), Inches(0.7), name=f"Failure {k}")
        dot(s, Inches(0.65), y + Inches(0.15), Inches(0.4), c, k[0])
        text(s, Inches(1.2), y + Inches(0.05), Inches(4.4), Inches(0.62),
             [[(k + "  ", {"bold": True, "size": 13}), (v, {"size": 12, "color": MUTED})]], anchor=MSO_ANCHOR.MIDDLE)
    card(s, Inches(6.0), TOP + Inches(0.1), Inches(2.5), Inches(3.9), name="Stat card")
    text(s, Inches(6.15), TOP + Inches(0.35), Inches(2.2), Inches(1.0), f"{imu10:.1f} m",
         size=48, bold=True, color=RED)
    text(s, Inches(6.15), TOP + Inches(1.35), Inches(2.2), Inches(1.0),
         "drift in 10 s when a real headset trusts its IMU alone (TLIO test sequence)", size=12, color=MUTED)
    text(s, Inches(6.15), TOP + Inches(2.35), Inches(2.2), Inches(0.5), "5 IMUs, 6 cameras", size=17, bold=True, color=CYAN)
    text(s, Inches(6.15), TOP + Inches(2.95), Inches(2.2), Inches(0.9),
         "on Galaxy XR: redundancy that can outvote a failing sensor", size=12, color=MUTED)
    s.notes_slide.notes_text_frame.text = (
        "Official track text: design robust trajectory-estimation techniques capable of maintaining reliable "
        "tracking and navigation when sensor measurements become degraded or unreliable.")

    # ---------- 3. Solution
    s = sl[2]
    text(s, Inches(0.5), TOP, Inches(8.0), Inches(0.6),
         "Find the sensor that is lying, stop trusting it, keep tracking on the rest", size=17, bold=True)
    steps = [("1", "Camera health", "Each frame scored for light, sharpness and trackable corners before fusion", ORANGE),
             ("2", "IMU voting", "IMUs share one rigid frame, so they must agree; a stuck, biased or spiking IMU is outvoted", RED),
             ("3", "Fault-aware motion model", "Tiny network reads 1 s of IMU, predicts movement and how sure it is", GREEN),
             ("4", "Robust Kalman filter", "Blends sensors by trust; a soft gate caps any reading that disagrees", CYAN)]
    for i, (n, h, d, c) in enumerate(steps):
        x = Inches(0.5 + i * 2.05)
        card(s, x, Inches(1.85), Inches(1.9), Inches(2.6), name=f"Step {n}")
        dot(s, x + Inches(0.15), Inches(2.0), Inches(0.42), c, n)
        text(s, x + Inches(0.1), Inches(2.55), Inches(1.72), Inches(0.6), h, size=14, bold=True)
        text(s, x + Inches(0.1), Inches(3.15), Inches(1.72), Inches(1.3), d, size=11, color=MUTED)
    text(s, Inches(0.5), Inches(4.6), Inches(8.0), Inches(0.6),
         [[("Out: ", {"bold": True, "color": CYAN}),
           ("position + orientation at 200 Hz, per-sensor trust and the filter's own uncertainty, a tracking-quality "
            "signal apps can act on.", {"color": MUTED})]], size=12)

    # ---------- 4. Architecture
    s = sl[3]
    boxes = [("Cameras", 0.5, 1.35, ORANGE), ("IMU 1 ... IMU N", 0.5, 3.15, RED)]
    for label, x, y, c in boxes:
        card(s, Inches(x), Inches(y), Inches(1.6), Inches(0.8), name=label)
        text(s, Inches(x), Inches(y), Inches(1.6), Inches(0.8), label, size=13, bold=True,
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, color=c)
    mids = [("Health score\nlight, blur, corners", 2.55, 1.35), ("Vote\nconsistency check", 2.55, 3.15),
            ("Motion model\n1 s IMU -> move + doubt", 4.6, 3.15)]
    for label, x, y in mids:
        card(s, Inches(x), Inches(y), Inches(1.8), Inches(0.8), name=label.split("\n")[0])
        a, b = label.split("\n")
        text(s, Inches(x), Inches(y), Inches(1.8), Inches(0.8), [[(a, {"bold": True, "size": 13})], [(b, {"size": 10, "color": MUTED})]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    card(s, Inches(6.65), Inches(1.9), Inches(1.85), Inches(1.9), fill=RGBColor(0x0F, 0x2A, 0x2A), name="Filter")
    text(s, Inches(6.65), Inches(1.9), Inches(1.85), Inches(1.9),
         [[("Error-state Kalman filter", {"bold": True, "size": 13})],
          [("15 states, soft chi-square gate, trust-weighted updates", {"size": 10, "color": MUTED})]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    arrows = [(2.1, 1.75, 0.45), (2.1, 3.55, 0.45), (4.35, 3.55, 0.25), (4.35, 1.75, 2.3), (6.4, 3.55, 0.25)]
    for x, y, w in arrows:
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y - 0.07), Inches(w), Inches(0.14))
        a.fill.solid()
        a.fill.fore_color.rgb = GREY
        a.line.fill.background()
    text(s, Inches(0.5), Inches(4.25), Inches(8.0), Inches(1.0),
         [[("Out: ", {"bold": True, "color": CYAN}), ("pose 200 Hz  |  per-sensor trust  |  position uncertainty", {"color": INK})],
          [(f"Model: {S['model'].get('params', 0):,} parameters, {S['model'].get('size_mb', 0):.1f} MB, "
            f"{S['model'].get('ms_per_prediction_cpu', 0):.1f} ms per prediction on one laptop CPU core, 10 per second. "
            "Training data: Meta TLIO headset dataset, 284 sequences.", {"size": 11, "color": MUTED})]], size=13)

    # ---------- 5. Track final output
    s = sl[4]
    set_header(s, "TRACK FINAL OUTPUT")
    sw_ = S["blackout_sweep"]
    cd = CategoryChartData()
    cd.categories = [f"{L} s" for L in sw_["lengths_s"]]
    show = [("naive_fusion", "Standard camera+IMU fusion", ORANGE), ("ours", "Ours", GREEN)]
    for k, lbl, _ in show:
        cd.add_series(lbl, [round(v, 2) for v in sw_[k]])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), TOP, Inches(5.0), Inches(3.4), cd)
    ch = gf.chart
    style_chart(ch, [c for _, _, c in show])
    ch.has_title = True
    ch.chart_title.text_frame.text = "Error when the camera comes back (m)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = INK
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.color.rgb = MUTED
    pl = ch.plots[0]
    pl.has_data_labels = True
    pl.data_labels.font.size = Pt(9)
    pl.data_labels.font.color.rgb = INK
    pl.data_labels.number_format = "0.00"
    pl.data_labels.number_format_is_linked = False
    ch.value_axis.has_major_gridlines = True
    ch.value_axis.major_gridlines.format.line.color.rgb = LINE
    ch.value_axis.tick_labels.font.color.rgb = MUTED
    ch.category_axis.tick_labels.font.color.rgb = MUTED
    ch.category_axis.format.line.color.rgb = LINE
    ch.value_axis.format.line.fill.background()
    text(s, Inches(0.4), Inches(4.5), Inches(5.0), Inches(0.7),
         f"{len(S['sequences'])} held-out headset sequences x {S['clip_s']:.0f} s, camera off from 20 s. "
         "Every number from python evaluate.py.", size=10, color=MUTED)
    picks = [("blackout_plus_saturation_all", "Camera off + every IMU saturates"),
             ("blackout_plus_one_imu_bias", "Camera off + one IMU biased"),
             ("blackout_plus_spikes_all", "Camera off + spikes on every IMU"),
             ("camera_blackout", "Camera off 10 s")]
    for i, (k, lbl) in enumerate(picks):
        y = TOP + Inches(i * 0.9)
        card(s, Inches(5.6), y, Inches(2.95), Inches(0.78), name=lbl)
        n, o = sc(k, "naive_fusion"), sc(k, "ours")
        text(s, Inches(5.7), y + Inches(0.02), Inches(2.8), Inches(0.35), lbl, size=11, color=MUTED)
        text(s, Inches(5.7), y + Inches(0.32), Inches(2.8), Inches(0.42),
             [[(f"{n:.2f} m", {"color": ORANGE, "bold": True, "size": 15}), ("  ->  ", {"color": MUTED, "size": 13}),
               (f"{o:.2f} m", {"color": GREEN, "bold": True, "size": 15}),
               (f"   {n / max(o, 1e-6):.1f}x", {"color": INK, "size": 12})]])

    # ---------- 6. Differentiation
    s = sl[5]
    text(s, Inches(0.5), TOP, Inches(8.0), Inches(0.5), "What exists, and what we add", size=20, bold=True)
    hdr = ["", "TLIO / RoNIN / EqNIO", "Samsung DVI-SLAM", "Ours"]
    rws = [["Uses multiple IMUs", "No (one IMU)", "No", "Yes, votes out a faulty one"],
           ["Checks the camera before trusting it", "No camera", "Learned inside dense BA", "Yes, per frame, before fusion"],
           ["Trained on broken IMU data", "Bias + tilt only", "n/a", "Noise, drift, dropout, spikes, saturation, dead"],
           ["Handles a camera that jumps to the wrong place", "n/a", "Partly", "Soft gate + 10 s hold-out"],
           ["Phone-class compute", "Yes", "Dense, heavy (their next step)", f"{S['model'].get('size_mb', 0):.1f} MB, {S['model'].get('ms_per_prediction_cpu', 0):.1f} ms per step"],
           ["Tracking-quality output", "No", "No", "Per-sensor trust + metres (OpenXR: yes/no flag)"]]
    tbl = s.shapes.add_table(len(rws) + 1, 4, Inches(0.5), Inches(1.6), Inches(8.0), Inches(3.4)).table
    widths = [2.4, 1.7, 1.7, 2.2]
    for j, w in enumerate(widths):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate([hdr] + rws):
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = PANEL if i else LINE
            if j == 3 and i:
                c.fill.fore_color.rgb = RGBColor(0x0F, 0x2A, 0x24)
            tf = c.text_frame
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = v
            r.font.name = FONT
            r.font.size = Pt(11 if i else 12)
            r.font.bold = i == 0 or j == 0
            r.font.color.rgb = GREEN if (j == 3 and i) else (INK if j == 0 or i == 0 else MUTED)
            c.margin_left = c.margin_right = Inches(0.08)

    # ---------- 7. Demo & local setup
    s = sl[6]
    card(s, Inches(0.5), TOP, Inches(4.3), Inches(3.9), name="Setup card")
    text(s, Inches(0.65), TOP + Inches(0.1), Inches(4.0), Inches(0.4), "Run it", size=16, bold=True)
    code = ["git clone https://github.com/", "    rehaanajani/pathfinder-xr", "cd pathfinder-xr", "bash reproduce.sh",
            "", "# no download, ~3 min:", "bash reproduce.sh --synthetic"]
    tb = text(s, Inches(0.65), TOP + Inches(0.55), Inches(4.0), Inches(2.0), code, size=12, color=CYAN)
    for p in tb.text_frame.paragraphs:
        p.space_after = Pt(2)
        for r in p.runs:
            r.font.name = "Courier New"
    text(s, Inches(0.65), TOP + Inches(2.6), Inches(4.0), Inches(1.2),
         ["Sets up Python, downloads the TLIO dataset (3.8 GB), trains or reuses both models, runs 10 tests "
          "and the benchmark.", "Outputs: results/report.html, results/summary.json, demo/index.html"],
         size=11, color=MUTED)
    card(s, Inches(5.0), TOP, Inches(3.5), Inches(3.9), name="Demo card")
    text(s, Inches(5.15), TOP + Inches(0.1), Inches(3.2), Inches(0.4), "Interactive demo", size=16, bold=True)
    text(s, Inches(5.15), TOP + Inches(0.55), Inches(3.2), Inches(3.2),
         ["Open demo/index.html in any browser, no install.",
          "Pick a failure: camera off 2-20 s, dark room, camera tracking the wrong thing, one IMU dead or biased.",
          "Press play: true path vs standard fusion vs ours on a real held-out recording, with live trust bars "
          "for the camera and each IMU."], size=12, color=MUTED)

    # ---------- 8. Brownie
    s = sl[7]
    iso = S["imu_isolation"]
    text(s, Inches(0.5), TOP, Inches(4.2), Inches(0.5), "Finding the faulty IMU", size=18, bold=True)
    cd = CategoryChartData()
    cd.categories = [k.capitalize() for k in iso]
    for kk, col in (("2", GREY), ("3", CYAN), ("5", GREEN)):
        cd.add_series(f"{kk} IMUs", [round(100 * iso[k][kk], 0) for k in iso])
    gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), TOP + Inches(0.5), Inches(4.4), Inches(3.0), cd)
    ch = gf.chart
    style_chart(ch, [GREY, CYAN, GREEN])
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.color.rgb = MUTED
    ch.value_axis.maximum_scale = 100
    ch.value_axis.has_major_gridlines = True
    ch.value_axis.major_gridlines.format.line.color.rgb = LINE
    ch.value_axis.tick_labels.font.color.rgb = MUTED
    ch.category_axis.tick_labels.font.color.rgb = MUTED
    ch.value_axis.format.line.fill.background()
    text(s, Inches(0.5), Inches(4.65), Inches(4.3), Inches(0.6),
         "% of faulty moments where the bad IMU is dropped and every good one kept. With 2 IMUs a smooth bias "
         "can't be pinned on one of them (0%); with 3+ it can.",
         size=10, color=MUTED)
    text(s, Inches(5.1), TOP, Inches(3.4), Inches(0.5), "Also built", size=18, bold=True)
    extra = [("Model self-check", "While the camera is healthy, the filter grades the network and distrusts it on motion it never learned"),
             ("Soft robust gate", "Re-anchors 1 s after an outage; holds out a camera that suddenly jumps"),
             ("Honest scoreboard", score_line()),
             ("Next on device", "TFLite / ExecuTorch on Galaxy, real multi-IMU logs, real camera frames")]
    for i, (h, d) in enumerate(extra):
        y = TOP + Inches(0.55 + i * 0.95)
        card(s, Inches(5.1), y, Inches(3.4), Inches(0.82), name=h)
        text(s, Inches(5.2), y + Inches(0.03), Inches(3.2), Inches(0.78),
             [[(h, {"bold": True, "size": 12})], [(d, {"size": 10, "color": MUTED})]])

    out = HERE / "PathFinder_SRMIST_CaliforniaBurrito.pptx"
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
