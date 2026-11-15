"""Fills Samsung's 9-slide submission template for the scored Track 3 task (IMU-only).

    python deck/build_track3_deck.py      # writes deck/SRMIST_CaliforniaBurrito.pptx

Every number is in R below with where it comes from. The older camera + IMU deck is deck/build_deck.py.
"""

from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------ numbers
R = {
    # docs/LEARNED_VELOCITY.md: held-out MH_05, V2_03, room6, seeds 1-3, tools/eval_track3.py (metres)
    "held_out": {
        "Physics only": {"ATE": 4.54, "Checkpoints": 4.19, "Blind 5 s": 4.03, "Dropouts": 4.47, "RPE 1 s": 0.97},
        "Stand still": {"ATE": 3.28, "Checkpoints": 3.03, "Blind 5 s": 2.38, "Dropouts": 3.27, "RPE 1 s": 0.85},
        "Ours": {"ATE": 3.19, "Checkpoints": 2.93, "Blind 5 s": 2.36, "Dropouts": 3.27, "RPE 1 s": 0.83},
    },
    # tools/eval_track3.py, PF_ENTRY=cpp, MH_05 seeds 1-3, no corrections at all (runs/track3/raw_mh05)
    "raw_mh05_ate": 825.0,
    "ours_mh05_ate": 6.72,
    # tools/make_bundle.py / docs/TRACK3_STATUS.md (C++ entry, Apple M5)
    "peak_mb": 2.2, "ms_per_s": 4.0, "contract_tests": 24,
    "params": 27020, "weights_kb": 108,
    # PR #11, #8 and the stress report (demo line, not scored)
    "phone_before": 46.0, "phone_after": 0.6,
    "headset_before": 62.4, "headset_after": 2.06,
    "fuzz_cases": 600, "bugs_fixed": 7,
}

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
DARK_GREEN = RGBColor(0x0F, 0x2A, 0x24)
FONT = "Calibri"
TOP = Inches(1.05)


def text(slide, x, y, w, h, runs, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, name=None, mono=False, space=6):
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
        p.space_after = Pt(space)
        parts = [(para, {})] if isinstance(para, str) else para
        for t, o in parts:
            r = p.add_run()
            r.text = t
            f = r.font
            f.name = "Courier New" if mono else FONT
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
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name = FONT
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = BG
    return s


def arrow(slide, x, y, w, color=GREY, down=False):
    shape = MSO_SHAPE.DOWN_ARROW if down else MSO_SHAPE.RIGHT_ARROW
    a = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(0.14 if down else w), Inches(w if down else 0.14))
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    return a


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


def main():
    prs = Presentation(HERE / "template.pptx")
    sl = prs.slides
    ho = R["held_out"]

    # ---------- 1. Title
    s = sl[0]
    rep = {"Track No": "Track 3  |  Path Finder", "XR Hackathon": "XR Hackathon",
           "Team Name": "Team California Burrito", "College Name": "SRM IST, Kattankulathur"}
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
    text(s, Inches(5.6), Inches(2.86), Inches(3.0), Inches(1.4),
         ["Rehaan Ajani (lead)", "Nikhil Solanki", "Debadrita Haldar", "Mayank Yadav"],
         size=12, name="Members list")
    text(s, Inches(0.55), Inches(3.55), Inches(5.0), Inches(1.0),
         [[("Trajectory from a jammed IMU alone, in 2.2 MB", {"size": 15, "bold": True, "color": CYAN})],
          [("Track 3: Trajectory Estimation Under Sensor Degradation  |  github.com/rehaanajani/pathfinder-xr",
            {"size": 10, "color": MUTED})]], name="Tagline")
    notes(s, "Theme: Track 3, Path Finder, trajectory estimation under sensor degradation. Team California Burrito, "
             "SRM Institute of Science and Technology, Kattankulathur.")

    # ---------- 2. Problem
    s = sl[1]
    text(s, Inches(0.5), TOP, Inches(5.3), Inches(1.4),
         [[("One sensor left, and it is being jammed", {"size": 22, "bold": True})],
          [("An aircraft flies through its checkpoints with only an IMU. Jamming makes the readings worse and "
            "worse until the IMU dies, and the path must still be known 5 s after that.", {"size": 12, "color": MUTED})]],
         name="Problem statement")
    rows = [("Noise and drift", "Noise rises and the gyro and accelerometer biases wander", ORANGE),
            ("Gaps", "2 s signal losses: missing, frozen, zeroed or NaN rows", RED),
            ("Blind start and end", "No ground truth, unknown gravity; then total failure and 5 s blind", CYAN)]
    for i, (k, v, c) in enumerate(rows):
        y = Inches(2.65 + i * 0.85)
        card(s, Inches(0.5), y, Inches(5.3), Inches(0.72), name=f"Failure {i + 1}")
        dot(s, Inches(0.63), y + Inches(0.16), Inches(0.4), c, str(i + 1))
        text(s, Inches(1.15), y + Inches(0.04), Inches(4.55), Inches(0.64),
             [[(k + "  ", {"bold": True, "size": 13}), (v, {"size": 11, "color": MUTED})]], anchor=MSO_ANCHOR.MIDDLE)
    card(s, Inches(6.05), TOP + Inches(0.1), Inches(2.45), Inches(3.95), name="Stat card")
    text(s, Inches(6.2), TOP + Inches(0.3), Inches(2.2), Inches(0.8), f"{R['raw_mh05_ate']:.0f} m",
         size=44, bold=True, color=RED)
    text(s, Inches(6.2), TOP + Inches(1.2), Inches(2.2), Inches(1.0),
         "average error when the degraded IMU is simply integrated (EuRoC MH_05, a recording we never trained on)",
         size=11, color=MUTED)
    text(s, Inches(6.2), TOP + Inches(2.4), Inches(2.2), Inches(0.5), "8 MB  |  20 ms", size=18, bold=True, color=CYAN)
    text(s, Inches(6.2), TOP + Inches(2.95), Inches(2.2), Inches(0.9),
         "peak memory, and compute per second of IMU, on one CPU thread", size=11, color=MUTED)
    notes(s, "Brief: estimate the 6-DoF trajectory through the checkpoints from the degraded IMU stream only, then "
             "dead-reckon 5 s after the IMU fails. Input imu0/data.csv (EuRoC format), output estimated_trajectory.txt "
             "(TUM format, one pose per IMU row). Scored: memory 20, runtime 20, checkpoints 30, blind 5 s 20, "
             "dropouts 10, with evo (ATE, RPE). The 825 m is plain strapdown integration of our degraded copy of MH_05, "
             "seeds 1-3.")

    # ---------- 3. Solution
    s = sl[2]
    text(s, Inches(0.5), TOP, Inches(8.0), Inches(0.6),
         "Trust each reading only as far as physics and a small learned model agree with it", size=16, bold=True)
    steps = [("1", "Check every sample", "NaN, zero, frozen, impossible values and time gaps are caught per row", RED),
             ("2", "Start blind", "Gravity and gyro bias come from the data; no ground truth needed", ORANGE),
             ("3", "Integrate, with brakes", "Gravity correction, still detection, speed limit, drift removal", CYAN),
             ("4", "Learned velocity", f"{R['params']:,}-parameter CNN reads 1 s of IMU: velocity and how sure it is", GREEN),
             ("5", "Coast when blind", "Dropouts and the 5 s after death: turn rate and speed decay from the last trusted state", GREY)]
    for i, (n, h, d, c) in enumerate(steps):
        x = Inches(0.5 + i * 1.62)
        card(s, x, Inches(1.8), Inches(1.5), Inches(2.65), name=f"Step {n}")
        dot(s, x + Inches(0.12), Inches(1.95), Inches(0.38), c, n)
        text(s, x + Inches(0.08), Inches(2.45), Inches(1.36), Inches(0.62), h, size=13, bold=True)
        text(s, x + Inches(0.08), Inches(3.08), Inches(1.36), Inches(1.35), d, size=10, color=MUTED)
    text(s, Inches(0.5), Inches(4.6), Inches(8.0), Inches(0.6),
         [[("Out: ", {"bold": True, "color": CYAN}),
           ("one 6-DoF pose per IMU row (TUM format), through every dropout and the 5 s after the IMU dies.",
            {"color": MUTED})]], size=12)
    notes(s, "The network's velocity is a measurement weighted by its own confidence; on bad input it reports more "
             "doubt, so physics takes over. Heading is unobservable with an IMU alone, so after dropouts the network's "
             "horizontal output is shrunk by the expected heading error.")

    # ---------- 4. Architecture
    s = sl[3]
    flow = [("imu0/data.csv", "degraded IMU", 0.4, RED), ("Sample health", "good / bad / gap", 2.05, ORANGE),
            ("Orientation", "gyro + gravity, bias", 3.7, CYAN), ("Position", "integrate, brakes", 5.35, CYAN),
            ("TUM output", "1 pose per row", 7.0, GREEN)]
    for label, sub, x, c in flow:
        card(s, Inches(x), Inches(1.35), Inches(1.45), Inches(0.85), name=label)
        text(s, Inches(x), Inches(1.35), Inches(1.45), Inches(0.85),
             [[(label, {"bold": True, "size": 12, "color": c})], [(sub, {"size": 9, "color": MUTED})]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=2)
    for x in (1.86, 3.51, 5.16, 6.81):
        arrow(s, x, 1.71, 0.17)
    card(s, Inches(3.7), Inches(2.75), Inches(3.1), Inches(0.85), fill=DARK_GREEN, name="Learned velocity")
    text(s, Inches(3.7), Inches(2.75), Inches(3.1), Inches(0.85),
         [[("Learned velocity", {"bold": True, "size": 12, "color": GREEN})],
          [(f"1 s of IMU -> velocity + confidence, {R['weights_kb']} KB", {"size": 9, "color": MUTED})]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=2)
    card(s, Inches(2.05), Inches(2.75), Inches(1.45), Inches(0.85), name="Dead reckoning")
    text(s, Inches(2.05), Inches(2.75), Inches(1.45), Inches(0.85),
         [[("Dead reckoning", {"bold": True, "size": 12, "color": GREY})], [("bad rows, 5 s blind", {"size": 9, "color": MUTED})]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE, space=2)
    arrow(s, 2.71, 2.27, 0.42, down=True)
    arrow(s, 6.0, 2.27, 0.42, color=GREEN, down=True)
    text(s, Inches(0.5), Inches(3.85), Inches(8.0), Inches(1.3),
         [[("Tech stack  ", {"bold": True, "color": CYAN}),
           ("C++17 + zlib (the entry)  |  Python + NumPy (readable reference, same output)  |  PyTorch (training only)  "
            "|  evo (scoring)  |  Docker  |  GitHub Actions", {"color": INK, "size": 11})],
          [("Single pass, constant memory: the input is streamed row by row and only a 1 s window is kept. "
            f"Peak {R['peak_mb']} MB, {R['ms_per_s']:.0f} ms per second of IMU, one thread, offline.",
            {"size": 11, "color": MUTED})]], size=12)
    notes(s, "Training: EuRoC MAV and TUM-VI recordings degraded on the fly (rising noise, bias drift, dropouts), "
             "with a held-out split never used for any choice. The network runs inside the C++ entry with no ML runtime.")

    # ---------- 5. Track final output
    s = sl[4]
    set_header(s, "TRACK FINAL OUTPUT")
    metrics = ["ATE", "Checkpoints", "Blind 5 s", "Dropouts"]
    show = [("Physics only", ORANGE), ("Stand still", GREY), ("Ours", GREEN)]
    cd = CategoryChartData()
    cd.categories = metrics
    for k, _ in show:
        cd.add_series(k, [ho[k][m] for m in metrics])
    ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.4), TOP, Inches(5.1), Inches(3.35), cd).chart
    ch.font.name = FONT
    ch.font.size = Pt(10)
    ch.font.color.rgb = MUTED
    for ser, (_, c) in zip(ch.plots[0].series, show):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = c
        ser.format.line.color.rgb = c
    ch.has_title = True
    ch.chart_title.text_frame.text = "Error on held-out recordings (m, lower is better)"
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(12)
    ch.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = INK
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.color.rgb = MUTED
    pl = ch.plots[0]
    pl.has_data_labels = True
    pl.data_labels.font.size = Pt(8)
    pl.data_labels.font.color.rgb = INK
    pl.data_labels.number_format = "0.00"
    pl.data_labels.number_format_is_linked = False
    ch.value_axis.has_major_gridlines = True
    ch.value_axis.major_gridlines.format.line.color.rgb = LINE
    ch.value_axis.tick_labels.font.color.rgb = MUTED
    ch.category_axis.tick_labels.font.color.rgb = MUTED
    ch.category_axis.format.line.color.rgb = LINE
    ch.value_axis.format.line.fill.background()
    text(s, Inches(0.4), Inches(4.45), Inches(5.1), Inches(0.8),
         "EuRoC MH_05 + V2_03 + TUM-VI room6, never used for training or tuning; degraded 3 ways each. "
         "SE(3)-aligned like evo. Physics only = our filter without the network.", size=9, color=MUTED)
    budget = [("Peak memory", f"{R['peak_mb']} MB", "of 8 MB", GREEN),
              ("Compute", f"{R['ms_per_s']:.0f} ms", "per s of IMU, of 20", GREEN),
              ("Output contract", f"{R['contract_tests']}/{R['contract_tests']}", "tests: 1 pose per row, bad rows", CYAN),
              ("C++ = Python", "1 µm", "largest difference", CYAN)]
    for i, (h, big, sub, c) in enumerate(budget):
        y = TOP + Inches(i * 0.88)
        card(s, Inches(5.7), y, Inches(2.85), Inches(0.78), name=h)
        text(s, Inches(5.8), y + Inches(0.02), Inches(2.7), Inches(0.3), h, size=10, color=MUTED)
        text(s, Inches(5.8), y + Inches(0.3), Inches(2.7), Inches(0.45),
             [[(big, {"bold": True, "size": 17, "color": c}), ("  " + sub, {"size": 10, "color": MUTED})]])
    notes(s, f"Held-out means: ours ATE {ho['Ours']['ATE']} m vs stand still {ho['Stand still']['ATE']} m vs physics only "
             f"{ho['Physics only']['ATE']} m; RPE 1 s {ho['Ours']['RPE 1 s']} vs {ho['Stand still']['RPE 1 s']} vs "
             f"{ho['Physics only']['RPE 1 s']} m. The degradation is our stand-in for the organisers' script, which "
             "we have not received. Memory and time: C++ entry on an Apple M5, from tools/make_bundle.py.")

    # ---------- 6. Differentiation
    s = sl[5]
    text(s, Inches(0.5), TOP, Inches(8.0), Inches(0.5), "What else could be submitted, and what we add", size=19, bold=True)
    hdr = ["", "Integrate the IMU", "Stand still", "Large learned model", "Ours"]
    rws = [["Error on held-out data", f"{R['raw_mh05_ate']:.0f} m (MH_05)", f"{ho['Stand still']['ATE']} m",
            "not run (no clean IMU)", f"{ho['Ours']['ATE']} m"],
           ["Bad rows, gaps, IMU death", "drifts away", "ignores them", "garbage in, garbage out", "caught per row, coasts"],
           ["Unknown gravity, no ground truth", "needs a start pose", "n/a", "needs a gravity frame", "estimated from the data"],
           ["Knows how sure it is", "no", "no", "sometimes", "yes, weights its own output"],
           ["Fits 8 MB / 20 ms", "yes", "yes", "no (PyTorch at run time)", f"{R['peak_mb']} MB, {R['ms_per_s']:.0f} ms"]]
    tbl = s.shapes.add_table(len(rws) + 1, 5, Inches(0.5), Inches(1.55), Inches(8.0), Inches(2.6)).table
    for j, w in enumerate([2.0, 1.45, 1.25, 1.65, 1.65]):
        tbl.columns[j].width = Inches(w)
    for i, row in enumerate([hdr] + rws):
        for j, v in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = DARK_GREEN if (j == 4 and i) else (PANEL if i else LINE)
            tf = c.text_frame
            tf.paragraphs[0].text = ""
            r = tf.paragraphs[0].add_run()
            r.text = v
            r.font.name = FONT
            r.font.size = Pt(10 if i else 11)
            r.font.bold = i == 0 or j == 0
            r.font.color.rgb = GREEN if (j == 4 and i) else (INK if j == 0 or i == 0 else MUTED)
            c.margin_left = c.margin_right = Inches(0.06)
    card(s, Inches(0.5), Inches(4.35), Inches(8.0), Inches(0.62), name="Limitations")
    text(s, Inches(0.6), Inches(4.38), Inches(7.8), Inches(0.58),
         [[("Honest limits  ", {"bold": True, "color": ORANGE, "size": 11}),
           ("On held-out data we beat standing still by about 3 %: an IMU alone cannot see its heading, so long "
            "flights drift. Our degradation is a stand-in for the organisers' script. Results on their data may differ.",
            {"size": 10, "color": MUTED})]])
    notes(s, "Large learned model column refers to TLIO / RoNIN-style inertial odometry networks, which are trained on "
             "clean, gravity-aligned IMU data and run in PyTorch.")

    # ---------- 7. Demo & local setup
    s = sl[6]
    card(s, Inches(0.5), TOP, Inches(4.4), Inches(3.95), name="Setup card")
    text(s, Inches(0.65), TOP + Inches(0.1), Inches(4.1), Inches(0.4), "Run it", size=16, bold=True)
    tb = text(s, Inches(0.65), TOP + Inches(0.55), Inches(4.15), Inches(2.3),
              ["docker build -t pathfinder src/", "docker run --rm -v $PWD/data:/data \\",
               "  pathfinder /data -o /data/estimated_trajectory.txt", "", "# or without Docker:", "make -C src",
               "src/pathfinder <imu folder|zip|csv> \\", "  -o estimated_trajectory.txt"],
              size=10, color=CYAN, mono=True, space=1)
    text(s, Inches(0.65), TOP + Inches(2.85), Inches(4.1), Inches(1.05),
         ["Input: EuRoC imu0/data.csv (a folder, a zip or the CSV). Output: TUM format, one pose per row.",
          "Python reference, same output: python3 src/main.py <input>"], size=10, color=MUTED, space=3)
    card(s, Inches(5.1), TOP, Inches(3.45), Inches(3.95), name="Check card")
    text(s, Inches(5.25), TOP + Inches(0.1), Inches(3.2), Inches(0.4), "Check it", size=16, bold=True)
    text(s, Inches(5.25), TOP + Inches(0.55), Inches(3.2), Inches(3.3),
         [[("Contract tests  ", {"bold": True, "size": 11}),
           ("PF_ENTRY=cpp pytest tests/stress/test_main_contract.py: one pose per row, NaN, frozen, gaps, cut files, "
            "8 MB, 20 ms", {"size": 10, "color": MUTED})],
          [("Score  ", {"bold": True, "size": 11}),
           ("tools/eval_track3.py degrades EuRoC / TUM-VI and scores ATE, checkpoints, blind 5 s, dropouts, RPE",
            {"size": 10, "color": MUTED})],
          [("Bundle  ", {"bold": True, "size": 11}),
           ("tools/make_bundle.py <input> writes CaliforniaBurrito.zip with metadata.json", {"size": 10, "color": MUTED})],
          [("CI  ", {"bold": True, "size": 11}),
           ("every push builds the Docker image and runs the tests on Linux", {"size": 10, "color": MUTED})]],
         space=8)

    # ---------- 8. Brownie
    s = sl[7]
    text(s, Inches(0.5), TOP, Inches(8.0), Inches(0.5), "Beyond the scored task: the same ideas on real XR hardware",
         size=17, bold=True)
    tiles = [(f"{R['phone_before']:.0f} m -> {R['phone_after']} m", "Real phone, camera covered for 61 s",
              "Galaxy M36 live: position error when vision returns, standard fusion vs ours (stand-still prior in long outages)"),
             (f"{R['headset_before']:.0f} m -> {R['headset_after']:.1f} m", "Real VR headsets, clipped IMUs",
              "Samsung Odyssey+ / Valve Index recordings: the saturation guard spots clipping from the data itself"),
             ("5 IMUs", "Finding the faulty IMU",
              "IMUs on one rigid frame must agree; parity-space checks drop a biased, stuck or spiking one (Galaxy XR has 5)"),
             (f"{R['fuzz_cases']} cases", "Breaking our own code",
              f"Fuzz and hostile-input tests: {R['bugs_fixed']} bugs found by the team and fixed, each with a test")]
    for i, (big, h, d) in enumerate(tiles):
        x = Inches(0.5 + (i % 2) * 4.05)
        y = TOP + Inches(0.6 + (i // 2) * 1.85)
        card(s, x, y, Inches(3.95), Inches(1.7), name=h)
        text(s, x + Inches(0.12), y + Inches(0.08), Inches(3.7), Inches(0.5), big, size=22, bold=True, color=GREEN)
        text(s, x + Inches(0.12), y + Inches(0.62), Inches(3.7), Inches(0.35), h, size=12, bold=True)
        text(s, x + Inches(0.12), y + Inches(0.95), Inches(3.7), Inches(0.72), d, size=10, color=MUTED)
    notes(s, "These parts use cameras and several IMUs, which the scored task does not have. They are in the repository "
             "with their own benchmarks (README) and are the plan for the final demo.")

    out = HERE / "SRMIST_CaliforniaBurrito.pptx"
    prs.save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
