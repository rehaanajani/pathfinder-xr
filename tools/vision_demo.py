"""Self-contained replay page from the frozen real-vision bundle (issue #3 F).

    python tools/vision_demo.py results/vision/<run> --rows results/runs/<run>/rows.jsonl
                                [--out demo/vision_replay.html]

The page replays PRECOMPUTED offline results; nothing is computed live in the browser. Every curve
comes from the source run's rows.jsonl (checked against the SHA-256 recorded in bundle.json).
"""

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path

import numpy as np

SYSTEMS = ("vo_only", "conventional", "pathfinder", "pathfinder_model")
EVERY = 1          # Preserve every recorded 10 Hz sample; do not discard peaks.


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", default="demo/vision_replay.html")
    a = ap.parse_args(argv)
    b = json.loads((Path(a.bundle) / "bundle.json").read_text(encoding="utf-8"))
    got = hashlib.sha256(Path(a.rows).read_bytes()).hexdigest()
    if got != b["source_sha256"]["rows.jsonl"]:
        sys.exit("rows.jsonl does not match the bundle's recorded SHA-256")
    rows = [json.loads(x) for x in Path(a.rows).read_text(encoding="utf-8").splitlines() if x.strip()]
    proto = json.loads((Path(a.bundle) / "source_protocol.json").read_text(encoding="utf-8"))
    data = {}
    for r in rows:
        if r["system"] not in SYSTEMS or not r.get("series"):
            continue
        s = r["series"]
        t = np.array(s["t"])[::EVERY]
        e = np.array(s["err"], float)[::EVERY]
        pa = np.array(s["p_aligned"], float)[::EVERY, :2]
        pr = np.array(s["p_ref"], float)[::EVERY, :2]
        rnd = lambda x: [None if not np.isfinite(v) else round(float(v), 3) for v in np.ravel(x)]  # noqa: E731
        d = data.setdefault(r["scenario"], {"t": rnd(t), "ref": rnd(pr), "faults": r["scenario_def"].get("image_faults", [])
                                            + r["scenario_def"].get("imu_faults", []), "sys": {}})
        d["sys"][r["system"]] = {"err": rnd(e), "xy": rnd(pa),
                                 "ate": r["metrics"].get("start_se3", {}).get("ate_all", {}).get("rmse")}
    payload = json.dumps({"data": data, "run": b["source_run"], "protocol": b["protocol"],
                          "commit": (b.get("code") or {}).get("commit"), "systems": SYSTEMS})
    page = TEMPLATE.replace("__PAYLOAD__", payload).replace("__RUN__", html.escape(b["source_run"])) \
        .replace("__SEQ__", html.escape(proto["sequence"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(page, encoding="utf-8", newline="\n")
    print("wrote", a.out, f"{len(page) / 1e6:.2f} MB")


TEMPLATE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Path Finder: real-vision replay (TUM-VI __SEQ__)</title>
<style>body{font:14px system-ui,sans-serif;margin:20px;max-width:1100px;color:#222}
.banner{background:#fff4d6;border:1px solid #e0b84c;padding:10px 14px;border-radius:6px}
canvas{border:1px solid #ccc;background:#fff} .row{display:flex;gap:16px;flex-wrap:wrap}
.legend span{display:inline-block;margin-right:14px} .sw{display:inline-block;width:14px;height:4px;margin-right:4px;vertical-align:middle}
table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:3px 8px;text-align:right}</style></head><body>
<h1>Path Finder: real-vision replay</h1>
<p class="banner"><b>Prerecorded offline replay. Nothing is computed live in this page.</b> These are real stereo images and the
real IMU of TUM-VI <b>__SEQ__</b>, a <b>development</b> recording that earlier work already used. Faults were injected into
the images or IMU before tracking. No system was given or started from the reference. Error is position error after
start-SE(3) alignment. Source run <code>__RUN__</code>. This is one recording, not evidence of general performance.</p>
<label>Scenario <select id="sc"></select></label>
<div class="legend" id="lg"></div>
<div class="row"><canvas id="err" width="680" height="300" aria-label="position error over time"></canvas>
<canvas id="xy" width="380" height="380" aria-label="top view trajectory"></canvas></div>
<table id="tb"></table>
<p>All fixed and randomized scenarios are included. Curves show every stored 10 Hz sample, not full-rate IMU peaks.
IMU-only trajectories are omitted from this comparison; see the report for their errors.</p>
<p>Grey bands: injected fault intervals. Black: motion-capture reference (used only for scoring).</p>
<script>
const P=__PAYLOAD__, C={vo_only:'#8c8c8c',conventional:'#1f77b4',pathfinder:'#d62728',pathfinder_model:'#2ca02c'};
const sel=document.getElementById('sc'); Object.keys(P.data).forEach(k=>{const o=document.createElement('option');o.text=k;sel.add(o)});
document.getElementById('lg').innerHTML=P.systems.map(s=>`<span><i class="sw" style="background:${C[s]}"></i>${s}</span>`).join('');
function draw(){const d=P.data[sel.value],e=document.getElementById('err').getContext('2d'),W=680,H=300;e.clearRect(0,0,W,H);
 const T=d.t[d.t.length-1]||1;let M=0;for(const s in d.sys)for(const v of d.sys[s].err)if(v!=null)M=Math.max(M,v);M=M||1;
 e.fillStyle='#0001';for(const f of d.faults)e.fillRect(40+(W-50)*f.start/T,0,(W-50)*(f.end-f.start)/T,H-20);
 e.fillStyle='#222';e.fillText('error (m), max shown '+M.toFixed(2),44,12);e.fillText('time (s)',W-50,H-4);
 for(const s in d.sys){e.strokeStyle=C[s];e.beginPath();let up=false;d.t.forEach((t,i)=>{const v=d.sys[s].err[i];
  if(v==null){up=false;return}const x=40+(W-50)*t/T,y=(H-20)*(1-v/M);up?e.lineTo(x,y):e.moveTo(x,y);up=true});e.stroke()}
 const g=document.getElementById('xy').getContext('2d'),S=380;g.clearRect(0,0,S,S);const all=[...d.ref,...Object.values(d.sys).flatMap(s=>s.xy)];
 let lo=[1e9,1e9],hi=[-1e9,-1e9];for(let i=0;i<all.length;i+=2){if(all[i]==null||all[i+1]==null)continue;lo[0]=Math.min(lo[0],all[i]);hi[0]=Math.max(hi[0],all[i]);lo[1]=Math.min(lo[1],all[i+1]);hi[1]=Math.max(hi[1],all[i+1])}
 const sc=(S-20)/Math.max(hi[0]-lo[0],hi[1]-lo[1],0.1)/1.2,cx=(lo[0]+hi[0])/2,cy=(lo[1]+hi[1])/2;
 const line=(xy,col)=>{g.strokeStyle=col;g.beginPath();let up=false;for(let i=0;i<xy.length;i+=2){if(xy[i]==null||xy[i+1]==null){up=false;continue}
  const x=S/2+(xy[i]-cx)*sc,y=S/2-(xy[i+1]-cy)*sc;up?g.lineTo(x,y):g.moveTo(x,y);up=true}g.stroke()};
 line(d.ref,'#000');for(const s in d.sys)line(d.sys[s].xy,C[s]);
 document.getElementById('tb').innerHTML='<tr><th>system</th><th>ATE RMSE (m)</th></tr>'+P.systems.filter(s=>d.sys[s]).map(s=>`<tr><td style="text-align:left">${s}</td><td>${d.sys[s].ate==null?'–':d.sys[s].ate.toFixed(3)}</td></tr>`).join('')}
sel.onchange=draw;draw();
</script></body></html>
"""

if __name__ == "__main__":
    main()
