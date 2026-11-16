"""Fills Samsung's 9-slide submission template for the scored Track 3 task (IMU-only).

    python deck/make_images.py --work <run folder> --gt <MH_05 ground truth>   # real-data images in deck/img/
    python deck/build_track3_deck.py                                          # writes deck/SRMIST_CaliforniaBurrito.pptx

Style: the template's own frame, our content in black, white and greys with one blue accent (from the template) and
one muted red used only for failure. Every number is in R with where it comes from.
"""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
IMG = HERE / "img"

# ------------------------------------------------------------------------------------------------ numbers
R = {
    # all 17 public EuRoC + TUM-VI recordings damaged by the ORGANISERS' degrade_imu.py (seed 42, accel-first),
    # C++ entry at the final settings: runs/organiser/sw_O.json and sw_S.json, results/track3/coupling_sweep.md
    "all17": {"Standing still": {"ATE": 2.62, "Checkpoints": 2.45, "Blind 5 s": 2.28, "Dropouts": 2.42, "Drift 1 s": 0.77},
              "Path Finder": {"ATE": 2.32, "Checkpoints": 2.13, "Blind 5 s": 2.24, "Dropouts": 2.13, "Drift 1 s": 0.57}},
    "test3": {"still": 3.31, "ours": 3.10},          # MH_05, V2_03, room6 (never trained on), same run
    "naive_km": 72,                                  # plain integration of MH_05 with the organisers' damage, 113 s
    "mac_mb": 2.5, "mac_ms": 1.8, "phone_mb": 4.0, "phone_ms": 5.0, "win_mb": 5.1, "win_ms": 14.0,   # docs/reports
    "tests": 58, "params": 42716, "weights_kb": 171,
    # phone live test, 48 s camera blackout, error vs ARCore when it returned (docs/reports/03_phone.md)
    "phone_std": 227, "phone_ours": 3.5, "phone_freeze": 1.8,
}

INK = RGBColor(0xF2, 0xF3, 0xF5)
SOFT = RGBColor(0xC9, 0xCD, 0xD4)
MUTED = RGBColor(0x8A, 0x91, 0x9E)
LINE = RGBColor(0x2A, 0x30, 0x3C)
PANEL = RGBColor(0x11, 0x15, 0x20)
PANEL2 = RGBColor(0x17, 0x1C, 0x29)
ACCENT = RGBColor(0x5B, 0x8C, 0xFF)
ALERT = RGBColor(0xE5, 0x48, 0x4D)
GREY = RGBColor(0x5A, 0x61, 0x6E)
FONT = "Calibri"
MONO = "Courier New"


def text(slide, x, y, w, h, runs, size=12, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         name=None, font=FONT, space=4):
    """runs: str, or list of paragraphs; a paragraph is str or list of (text, opts) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        for t, o in ([(para, {})] if isinstance(para, str) else para):
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = o.get("font", font)
            f.size = Pt(o.get("size", size))
            f.bold = o.get("bold", bold)
            f.color.rgb = o.get("color", color)
    return tb


def box(slide, x, y, w, h, fill=PANEL, line=LINE, name=None, radius=0.04, weight=0.75):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(weight)
    s.shadow.inherit = False
    if name:
        s.name = name
    return s


def rule(slide, x, y, w, color=LINE, weight=0.75):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x + w), Inches(y))
    c.line.color.rgb = color
    c.line.width = Pt(weight)
    return c


def arrow(slide, x1, y1, x2, y2, color=MUTED, weight=1.25):
    from lxml import etree
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(weight)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, "{http://schemas.openxmlformats.org/drawingml/2006/main}tailEnd")
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def node(slide, x, y, w, h, title, sub, accent=False, name=None, fill=PANEL):
    box(slide, x, y, w, h, fill=fill, line=ACCENT if accent else LINE, name=name, weight=1.25 if accent else 0.75)
    text(slide, x + 0.08, y, w - 0.16, h, [[(title, {"bold": True, "size": 11, "color": INK})],
                                         [(sub, {"size": 8.5, "color": MUTED})]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=1)


def picture(slide, path, x, y, w=None, h=None, name=None, border=True):
    pic = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    if name:
        pic.name = name
    if border:
        pic.line.color.rgb = LINE
        pic.line.width = Pt(0.75)
    return pic


def kicker(slide, x, y, s, color=ACCENT, w=None):
    w = w if w is not None else min(6.0, 9.5 - x)
    text(slide, x, y, w, 0.25, s.upper(), size=8.5, bold=True, color=color)


def headline(slide, s, sub=None, y=1.08):
    text(slide, 0.5, y, 9.0, 0.45, s, size=20, bold=True, name="Headline")
    if sub:
        text(slide, 0.5, y + 0.44, 9.0, 0.3, sub, size=11, color=MUTED, name="Subhead")


def set_header(slide, new):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            runs = sh.text_frame.paragraphs[0].runs
            runs[0].text = new
            for r in runs[1:]:
                r.text = ""
            return


def notes(slide, t):
    slide.notes_slide.notes_text_frame.text = t


def stat(slide, x, y, w, big, label, color=INK, name=None, big_size=26):
    text(slide, x, y, w, 0.5, big, size=big_size, bold=True, color=color, name=name)
    text(slide, x, y + 0.5, w, 0.6, label, size=9.5, color=MUTED)


# ------------------------------------------------------------------------------------------------ slides
def main():
    prs = Presentation(HERE / "template.pptx")
    sl = prs.slides
    a17 = R["all17"]

    # ---------- 1. Title
    s = sl[0]
    rep = {"Track No": "Track 3  ·  Path Finder", "Team Name": "Team California Burrito",
           "College Name": "SRM IST, Kattankulathur"}
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text.strip()
        if t in rep:
            sh.text_frame.paragraphs[0].runs[0].text = rep[t]
        elif "Team Name" in t and "College" in t:
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    r.text = rep.get(r.text.strip(), r.text)
        elif t == "Members Details":
            sh.text_frame.paragraphs[0].runs[0].text = "Members"
    text(s, 5.6, 2.86, 3.2, 1.3, ["Rehaan Ajani  (lead)", "Nikhil Solanki", "Debadrita Haldar", "Mayank Yadav"],
         size=11, color=SOFT, name="Members list", space=3)
    text(s, 0.55, 3.75, 4.9, 1.2,
         [[("Trajectory from a degraded IMU alone", {"size": 16, "bold": True, "color": INK})],
          [(f"{R['mac_mb']} MB of memory  ·  {R['mac_ms']} ms per second of data  ·  one CPU thread",
            {"size": 10.5, "color": MUTED})],
          [("Track 3: Trajectory Estimation Under Sensor Degradation", {"size": 9.5, "color": MUTED})]],
         name="Tagline", space=3)
    notes(s, "Team California Burrito, SRM Institute of Science and Technology, Kattankulathur. Track 3, Path Finder: "
             "trajectory estimation under sensor degradation, IMU only.")

    # ---------- 2. Problem
    s = sl[1]
    headline(s, "One sensor left, and it is being jammed",
             "Only the IMU: noise and drift rise, readings spike, freeze and vanish, then it dies. The path is still due.")
    picture(s, IMG / "signal.png", 0.5, 1.85, w=5.6, name="Real IMU signal")
    text(s, 0.5, 4.5, 5.6, 0.3, "Real data: EuRoC MH_05 accelerometer, before and after the organisers' degradation script.",
         size=8.5, color=MUTED)
    box(s, 6.35, 1.85, 3.15, 3.25, name="Constraints")
    kicker(s, 6.55, 2.0, "Why it is hard")
    stat(s, 6.55, 2.25, 2.8, f"{R['naive_km']} km", "off after 113 s if the IMU is simply integrated: a 1° tilt error "
         "leaks 0.17 m/s² of gravity, and double integration does the rest", color=ALERT)
    rule(s, 6.55, 3.55, 2.75)
    stat(s, 6.55, 3.65, 2.8, "8 MB  ·  20 ms", "the budget: peak memory, and compute per second of data, one thread")
    notes(s, "Input imu0/data.csv (EuRoC format), output estimated_trajectory.txt (TUM, one pose per input row, "
             "including the 5 s after the IMU dies). Scored: memory 20, runtime 20, checkpoints 30, blind 5 s 20, "
             "dropouts 10. The 72 km comes from integrating MH_05 directly; clean data drifts almost the same "
             "(65 km after 107 s): the problem is gravity leaking through tilt error, not only the damage.")

    # ---------- 3. Solution
    s = sl[2]
    headline(s, "Trust each reading only as far as physics and a small network agree",
             "Five stages, each with one job. Every input row gets exactly one pose out.")
    stages = [("Check", "every sample", "NaN, zero, frozen, spikes, gaps, duplicates caught per row"),
              ("Start blind", "gravity & bias", "estimated from the data itself; no ground truth, no calibration"),
              ("Physics", "with brakes", "attitude + integration, still detection, speed limit, pull to start"),
              ("Learned velocity", f"{R['params']:,} parameters", "1 s of IMU in; velocity and its confidence out"),
              ("Coast", "when blind", "turn rate and speed decay from the last trusted state")]
    x0, w, gap = 0.5, 1.62, 0.225
    for i, (h, sub, d) in enumerate(stages):
        x = x0 + i * (w + gap)
        text(s, x, 1.85, w, 0.2, f"0{i + 1}", size=8, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
        node(s, x, 2.08, w, 0.8, h, sub, accent=(i == 3), name=f"Stage {i + 1}")
        text(s, x, 2.98, w, 0.85, d, size=9, color=MUTED, align=PP_ALIGN.CENTER)
        if i < 4:
            arrow(s, x + w + 0.03, 2.48, x + w + gap - 0.03, 2.48)
    box(s, 0.5, 3.9, 9.0, 1.2, fill=PANEL2, name="Equations")
    kicker(s, 0.7, 4.0, "The core equations, as implemented")
    eqs = [("Attitude", "q ← q ⊗ exp(½ ω Δt),   ω ← ω − b + kₚ (â × ĝ)"),
           ("Motion", "v ← v + (R a − g) Δt,   p ← p + v Δt"),
           ("Brakes", "|ω| < ε  ⇒  v → 0,   p ← p · e^(−Δt/τ)"),
           ("Network weight", "w = σ₀² / (σ₀² + σ²)")]
    for i, (k, e) in enumerate(eqs):
        cx = 0.7 + (i % 2) * 4.45
        cy = 4.3 + (i // 2) * 0.36
        text(s, cx, cy, 4.3, 0.3, [[(k + "   ", {"bold": True, "size": 10, "color": SOFT}),
                                     (e, {"size": 10.5, "color": INK, "font": "Cambria Math"})]])
    notes(s, "The network's velocity is a measurement weighted by its own confidence; on bad input it reports more "
             "doubt, so physics takes over. Heading is unobservable with an IMU alone, so position is pulled gently "
             "back toward the start.")

    # ---------- 4. Architecture
    s = sl[3]
    headline(s, "Streamed, bounded, one thread", "C++17 entry with the network inside; no ML runtime, nothing to install.")
    y = 1.95
    flow = [("imu0/data.csv", "degraded input"), ("Layout detect", "columns, units"), ("Sample health", "good / bad / gap"),
            ("Orientation", "gyro + gravity"), ("Position", "integrate + brakes"), ("TUM output", "1 pose per row")]
    w, gap, x0 = 1.33, 0.2, 0.5
    for i, (h, sub) in enumerate(flow):
        x = x0 + i * (w + gap)
        node(s, x, y, w, 0.72, h, sub, name=h)
        if i < len(flow) - 1:
            arrow(s, x + w + 0.02, y + 0.36, x + w + gap - 0.02, y + 0.36)
    xo, xp = x0 + 3 * (w + gap), x0 + 4 * (w + gap)
    node(s, xo + 0.35, 3.05, 2.2, 0.66, "Learned velocity", f"1 s window → v̂, σ  ({R['weights_kb']} KB)", accent=True,
         name="Network node")
    arrow(s, xo + w / 2, y + 0.72, xo + w / 2 + 0.25, 3.05, color=ACCENT)
    arrow(s, xo + 2.25, 3.05, xp + w / 2, y + 0.72, color=ACCENT)
    xs = x0 + 2 * (w + gap)
    node(s, xs - 1.1, 3.05, 1.75, 0.66, "Dead reckoning", "bad rows · 5 s blind", name="Dead reckoning node")
    arrow(s, xs + w / 2, y + 0.72, xs + 0.25, 3.05)
    picture(s, IMG / "terminal_short.png", 0.5, 3.88, h=1.62, name="Real run screenshot")
    text(s, 4.05, 3.95, 5.45, 1.5,
         [[("Real run  ", {"bold": True, "size": 10.5, "color": INK}),
           ("the organisers' damage on MH_05: 22,137 rows in, 22,137 poses out, 0.4 s, 2.5 MB.", {"size": 10, "color": MUTED})],
          [("Stack  ", {"bold": True, "size": 10.5, "color": INK}),
           ("C++17 + zlib (entry)  ·  Python/NumPy (reference, same output to 1 µm)  ·  PyTorch (training only)  ·  "
            "Docker  ·  GitHub Actions", {"size": 10, "color": MUTED})]], space=6)
    notes(s, "The input is streamed: a first look at up to 60 s for gravity and gyro bias, then one pass keeping a 1 s "
             "window, so memory does not grow with the recording. Both column orders and all timestamp units are "
             "detected and reported on stderr.")

    # ---------- 5. Track final output
    s = sl[4]
    set_header(s, "TRACK FINAL OUTPUT")
    metrics = ["ATE", "Checkpoints", "Blind 5 s", "Dropouts", "Drift 1 s"]
    cd = CategoryChartData()
    cd.categories = metrics
    show = [("Standing still", GREY), ("Path Finder", ACCENT)]
    for k, _ in show:
        cd.add_series(k, [a17[k][m] for m in metrics])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), Inches(1.05), Inches(5.0), Inches(3.1), cd).chart
    ch.font.name = FONT
    ch.font.size = Pt(9)
    ch.font.color.rgb = MUTED
    for ser, (_, c) in zip(ch.plots[0].series, show):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = c
        ser.format.line.fill.background()
    ch.plots[0].gap_width = 70
    ch.plots[0].overlap = -10
    ch.has_title = True
    ch.chart_title.text_frame.text = "Organisers' script, all 17 recordings (m, lower is better)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = INK
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.color.rgb = SOFT
    pl = ch.plots[0]
    pl.has_data_labels = True
    pl.data_labels.font.size = Pt(8)
    pl.data_labels.font.color.rgb = SOFT
    pl.data_labels.number_format = "0.00"
    pl.data_labels.number_format_is_linked = False
    ch.value_axis.has_major_gridlines = True
    ch.value_axis.major_gridlines.format.line.color.rgb = LINE
    ch.value_axis.tick_labels.font.color.rgb = MUTED
    ch.value_axis.format.line.fill.background()
    ch.category_axis.tick_labels.font.color.rgb = SOFT
    ch.category_axis.format.line.color.rgb = LINE
    text(s, 0.5, 4.2, 4.9, 0.9,
         [[("−12 % ATE, −13 % checkpoints, −12 % dropouts, −26 % drift; blind 5 s about level. ",
            {"size": 9.5, "color": SOFT}),
           (f"Never trained on (MH_05, V2_03, room6): {R['test3']['ours']:.2f} m vs {R['test3']['still']:.2f} m.",
            {"size": 9.5, "color": MUTED})]])
    picture(s, IMG / "trajectory.png", 5.6, 1.05, h=2.55, name="Trajectory vs truth")
    text(s, 5.6, 3.62, 2.55, 0.3, "MH_05, organisers' damage, SE(3)-aligned", size=8, color=MUTED)
    tiles = [(f"{R['mac_mb']} MB", "peak memory, of 8"), (f"{R['mac_ms']} ms", "per s of data, of 20"),
             (f"{R['tests']}/{R['tests']}", "stress tests pass"), ("1 µm", "C++ vs Python")]
    for i, (big, lab) in enumerate(tiles):
        x = 5.6 + (i % 2) * 1.98
        y = 3.98 + (i // 2) * 0.62
        box(s, x, y, 1.9, 0.55, name=lab)
        text(s, x + 0.12, y + 0.04, 1.7, 0.5, [[(big, {"bold": True, "size": 14, "color": INK})],
                                               [(lab, {"size": 8, "color": MUTED})]], space=0)
    text(s, 8.3, 1.1, 1.3, 2.6,
         [[("Galaxy M36", {"bold": True, "size": 10, "color": INK})],
          [(f"{R['phone_mb']} MB · {R['phone_ms']:.0f} ms on the phone's own CPU", {"size": 9, "color": MUTED})],
          [("Windows laptop", {"bold": True, "size": 10, "color": INK})],
          [(f"{R['win_mb']} MB · {R['win_ms']:.0f} ms under WSL", {"size": 9, "color": MUTED})]], space=4)
    notes(s, "Means over all 11 EuRoC and 6 TUM-VI recordings, damaged by the organisers' own degrade_imu.py with "
             "their seed and documented column order. 11 of the 17 were used to train the network; the three never "
             "used give 3.10 m vs 3.31 m. Memory and CPU: worst of 5 recordings (docs/reports).")

    # ---------- 6. Differentiation
    s = sl[5]
    headline(s, "What else could be submitted, and what we add")
    hdr = ["", "Integrate the IMU", "Stand still", "Large learned model", "Path Finder"]
    rws = [["Error, organisers' damage", f"{R['naive_km']} km (MH_05)", f"{a17['Standing still']['ATE']} m",
            "not run (no clean IMU)", f"{a17['Path Finder']['ATE']} m"],
           ["Bad rows, gaps, IMU death", "drifts away", "ignores them", "garbage in, garbage out", "caught per row; coasts"],
           ["Unknown gravity and mount", "needs a start pose", "n/a", "needs a gravity frame", "estimated; mount guard"],
           ["Knows how sure it is", "no", "no", "sometimes", "yes; weights its own output"],
           ["Fits 8 MB / 20 ms", "yes", "yes", "no (runtime too large)", f"{R['mac_mb']} MB · {R['mac_ms']} ms"]]
    tbl = s.shapes.add_table(len(rws) + 1, 5, Inches(0.5), Inches(1.65), Inches(9.0), Inches(2.6)).table
    for j, wd in enumerate([2.2, 1.6, 1.3, 1.9, 2.0]):
        tbl.columns[j].width = Inches(wd)
    for i, row in enumerate([hdr] + rws):
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = (PANEL2 if i == 0 else PANEL)
            tf = c.text_frame
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = v
            r.font.name = FONT
            r.font.size = Pt(10 if i else 10.5)
            r.font.bold = i == 0 or j == 0 or j == 4
            r.font.color.rgb = ACCENT if (j == 4) else (INK if j == 0 or i == 0 else MUTED)
            c.margin_left = c.margin_right = Inches(0.08)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
    box(s, 0.5, 4.45, 9.0, 0.7, fill=PANEL2, name="Limitations")
    text(s, 0.7, 4.52, 8.6, 0.6,
         [[("Honest limits  ", {"bold": True, "color": INK, "size": 10.5}),
           ("An IMU alone cannot see heading or absolute position, so error grows with distance. The blind 5 s is about "
            "level with standing still. Settings were chosen to be never worse than standing still on any recording, "
            "not to win on a few.", {"size": 10, "color": MUTED})]], anchor=MSO_ANCHOR.MIDDLE)
    notes(s, "Large learned model refers to TLIO / RoNIN-style inertial odometry networks, trained on clean, "
             "gravity-aligned IMU and run in PyTorch.")

    # ---------- 7. Demo & local setup
    s = sl[6]
    headline(s, "Run it in one command, check it in one more")
    box(s, 0.5, 1.6, 4.3, 2.35, fill=RGBColor(0x0D, 0x11, 0x17), name="Commands")
    text(s, 0.68, 1.72, 4.0, 2.2,
         [[("# build and run (Docker)", {"color": MUTED})],
          [("docker build -t pathfinder src/", {})],
          [("docker run --rm -v $PWD/data:/data \\", {})],
          [("  pathfinder /data -o /data/estimated_trajectory.txt", {})],
          [("", {})],
          [("# or natively", {"color": MUTED})],
          [("make -C src && src/pathfinder <input>", {})],
          [("", {})],
          [("# tests (58) and the organisers' script", {"color": MUTED})],
          [("PF_ENTRY=cpp pytest tests/stress", {})],
          [("python tools/eval_organiser.py \\", {})],
          [("  --script degrade_imu.py --data <recordings>", {})]],
         size=9, color=INK, font=MONO, space=0)
    text(s, 0.5, 4.05, 4.3, 1.1,
         [[("Input  ", {"bold": True, "size": 10, "color": INK}),
           ("a folder with imu0/data.csv, a zip, or the CSV; either column order; s/ms/µs/ns; m/s² or g.",
            {"size": 9.5, "color": MUTED})],
          [("Output  ", {"bold": True, "size": 10, "color": INK}),
           ("TUM format, one pose per input row, including the 5 s after the IMU dies.", {"size": 9.5, "color": MUTED})]],
         space=5)
    picture(s, IMG / "localhost_live_blackout.jpg", 5.05, 1.6, w=4.45, name="Live demo, localhost")
    text(s, 5.05, 4.1, 4.45, 1.0,
         [[("Live demo on localhost  ", {"bold": True, "size": 10, "color": INK}),
           ("Galaxy phone streaming its IMU; camera covered. The standard tracker (red) runs away; Path Finder (green) "
            "stays close, draws its uncertainty, and flags itself \"too_uncertain\".", {"size": 9, "color": MUTED})]])
    notes(s, "Live demo: demo_line/tools/live_demo/server.py, replaying the recorded 17 Nov phone session "
             "(http://localhost). CI on every push builds the Docker image and runs the stress tests on Linux.")

    # ---------- 8. Brownie
    s = sl[7]
    headline(s, "Beyond the scored task: the same ideas on real hardware")
    box(s, 0.5, 1.6, 4.4, 3.5, name="Phone test")
    kicker(s, 0.7, 1.75, "Real Galaxy phone · camera covered 48 s")
    bars = [("Standard tracker", R["phone_std"], ALERT, f"{R['phone_std']} m"),
            ("Path Finder", R["phone_ours"], ACCENT, f"{R['phone_ours']} m"),
            ("Freezing in place", R["phone_freeze"], SOFT, f"{R['phone_freeze']} m")]
    for i, (lab, v, col, s_) in enumerate(bars):
        y = 2.15 + i * 0.62
        text(s, 0.7, y, 2.0, 0.25, lab, size=10, color=INK)
        wbar = max(0.04, 2.9 * v / R["phone_std"])
        box(s, 0.7, y + 0.28, wbar, 0.14, fill=col, line=None, radius=0.5)
        text(s, 0.8 + wbar, y + 0.21, 1.0, 0.3, s_, size=11, bold=True, color=col)
    text(s, 0.7, 4.1, 4.0, 0.9, "Error against ARCore when the camera returned, in a room about 6 × 9 m. Freezing "
         "in place was closer still: in small spaces doing nothing is a strong baseline, which is why the scored entry "
         "pulls back toward its start.", size=9, color=MUTED)
    tiles = [("Scored entry on a phone", f"{R['phone_mb']} MB · {R['phone_ms']:.0f} ms",
              "the same C++ cross-compiled, run on the Galaxy M36 CPU; output matched the Mac's"),
             ("Real VR headsets", "IMU clipping", "Samsung Odyssey+ / Valve Index recordings: saturation spotted from "
              "the data itself"),
             ("Several IMUs", "fault voting", "IMUs on one rigid frame must agree; a biased, stuck or spiking one is "
              "dropped"),
             ("Breaking our own code", "600 cases", "fuzz and hostile-input tests; 7 bugs found and fixed, each with a "
              "test")]
    for i, (h, big, d) in enumerate(tiles):
        x = 5.1 + (i % 2) * 2.22
        y = 1.6 + (i // 2) * 1.78
        box(s, x, y, 2.12, 1.7, name=h)
        text(s, x + 0.14, y + 0.12, 1.86, 1.5, [[(h.upper(), {"bold": True, "size": 8, "color": ACCENT})],
                                                 [(big, {"bold": True, "size": 15, "color": INK})],
                                                 [(d, {"size": 8.5, "color": MUTED})]], space=3)
    notes(s, "These parts use cameras and several IMUs, which the scored task does not have. Phone numbers: "
             "docs/reports/03_phone.md (all four blackouts, with the freeze baseline).")

    # ---------- 9. Thank you
    s = sl[8]
    text(s, 2.97, 3.25, 5.5, 0.9,
         [[("github.com/rehaanajani/pathfinder-xr", {"size": 13, "bold": True, "color": INK})],
          [("Team California Burrito  ·  SRM IST  ·  Samsung PRISM XR Hackathon 2026", {"size": 10, "color": MUTED})]],
         space=4)

    out = HERE / "SRMIST_CaliforniaBurrito.pptx"
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
