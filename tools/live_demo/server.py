"""Live Path Finder demo: phone as the headset's sensors, tracking computed live on the laptop.

    python tools/live_demo/server.py [--model] [--replay runs/live/<file>.jsonl]

1. The phone app (android/, "Path Finder bridge") streams raw IMU + ARCore tracking over USB
   (`adb reverse tcp:7777 tcp:7777`, set up by tools/live_demo/run.sh).
2. Three trackers run on the same live data:
     blue  = ARCore alone (Google's camera tracking; freezes when it loses tracking)
     red   = a standard tracker: phone IMU + ARCore positions, trusting every reading
     green = Path Finder: same inputs, with sensor checks, outage logic, speed limit,
             saturation guard; draws its uncertainty circle and says why it's unsure
3. Open http://localhost:8080 for the live top-down view.

Every session is saved to runs/live/<time>.jsonl (raw lines from the phone), so it can be replayed
(--replay) and scored afterwards. Real: the phone's IMU, ARCore, covering the camera, shaking.
Simulated, and labelled on screen: the "Simulated IMU bias" button in the app.
"""

import argparse
import json
import socket
import sys
import threading
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pathfinder.filter import FilterConfig                     # noqa: E402
from pathfinder.pipeline import ours_cfg                      # noqa: E402
from pathfinder.stream import StreamingTracker                # noqa: E402

RATE = 200.0
DT_NS = int(1e9 / RATE)
Y_UP_TO_Z_UP = R.from_matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])    # ARCore world (y up) -> ours (z up)
STATE = {"paths": {"arcore": [], "standard": [], "ours": []}, "info": {}}
LOCK = threading.Lock()


class Live:
    def __init__(self, use_model=False):
        self.g, self.a = [], []                 # (t_ns, xyz) raw, unsorted arrival
        self.next_t = None
        self.trk = {}
        self.last_ar = None
        self.ar_pos = None
        self.sim = False
        self.counts = {"g": 0, "a": 0, "ar": 0}
        self.t0 = time.time()
        self.predictor = None
        if use_model:
            from pathfinder.predictor import Predictor
            self.predictor = Predictor("models/dispnet_faults/best.pt")

    def _start(self, p, q):
        ours = replace(ours_cfg(), trust=None)
        std = FilterConfig(use_health=False, use_gate=False)
        self.trk = {"standard": StreamingTracker(std, p, q),
                    "ours": StreamingTracker(ours, p, q, predictor=self.predictor)}

    def line(self, msg):
        k = msg["k"]
        self.counts[k] = self.counts.get(k, 0) + 1
        if k in ("g", "a"):
            (self.g if k == "g" else self.a).append((msg["t"], np.array(msg["v"], float)))
            self.sim = bool(msg.get("sim"))
            self._drain()
        elif k == "ar":
            if self.last_ar is not None and msg["t"] == self.last_ar["t"]:
                return                                   # same camera frame re-sent by the app
            p = Y_UP_TO_Z_UP.apply(np.array(msg["p"], float))
            q = (Y_UP_TO_Z_UP * R.from_quat(msg["q"])).as_quat()
            tracking = msg["s"] == "TRACKING"
            self.last_ar = {"t": msg["t"], "tracking": tracking, "reason": msg["r"], "n": msg.get("n", 0)}
            if tracking:
                self.ar_pos = p
                if not self.trk and self.g and self.a:
                    self._start(p, q)
                    # Camera frames and IMU samples use different phone clocks (measured: ARCore
                    # ~6.6 s ahead on a Galaxy M36). Fixes are applied on arrival, so the 200 Hz grid
                    # starts on the IMU clock.
                    self.next_t = min(max(t for t, _ in self.g), max(t for t, _ in self.a))
                health = 1.0 if msg["r"] == "NONE" else 0.3
                for t in self.trk.values():
                    t.push_camera_fix(p, health)

    def _drain(self):
        """Resample gyro + accel onto a 200 Hz grid (mean of each 5 ms bin) and feed the trackers."""
        if not self.trk or not self.g or not self.a:
            return
        g = sorted(self.g, key=lambda x: x[0])
        a = sorted(self.a, key=lambda x: x[0])
        ready = min(g[-1][0], a[-1][0])
        while self.next_t + DT_NS <= ready:
            lo, hi = self.next_t, self.next_t + DT_NS
            gv = [v for t, v in g if lo <= t < hi] or [min(g, key=lambda x: abs(x[0] - lo))[1]]
            av = [v for t, v in a if lo <= t < hi] or [min(a, key=lambda x: abs(x[0] - lo))[1]]
            gm, am = np.mean(gv, 0), np.mean(av, 0)
            for t in self.trk.values():
                t.push_imu(gm, am)
            self.next_t = hi
        self.g = [x for x in g if x[0] >= self.next_t - 50 * DT_NS]
        self.a = [x for x in a if x[0] >= self.next_t - 50 * DT_NS]
        self._publish()

    def _publish(self):
        o, s = self.trk["ours"].out, self.trk["standard"].out
        if o is None:
            return
        with LOCK:
            P = STATE["paths"]
            if self.ar_pos is not None:
                P["arcore"].append(self.ar_pos[:2].tolist())
            P["standard"].append(s["position"][:2].tolist())
            P["ours"].append(o["position"][:2].tolist())
            for v in P.values():
                del v[:-6000]
            el = max(time.time() - self.t0, 1e-3)
            STATE["info"] = {
                "ours_std": o["position_std_m"], "ours_tracked": o["tracked"], "ours_valid": o["valid"],
                "ours_reason": o["reason"], "arcore": (self.last_ar or {}).get("reason", "-"),
                "arcore_tracking": bool((self.last_ar or {}).get("tracking")), "sim_bias": self.sim,
                "imu_hz": round(self.counts.get("g", 0) / el), "ar_hz": round(self.counts.get("ar", 0) / el)}


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Path Finder live</title>
<meta name=viewport content="width=device-width,initial-scale=1"><style>
:root{--bg:#0e1116;--fg:#e8eaed;--mut:#9aa0a6;--blue:#4c8df6;--red:#ef5350;--green:#34c46a}
body{margin:0;background:var(--bg);color:var(--fg);font:15px system-ui,sans-serif}
header{display:flex;gap:24px;align-items:center;padding:12px 20px;flex-wrap:wrap}
.k{display:flex;gap:8px;align-items:center}.d{width:12px;height:12px;border-radius:50%}
#st{margin-left:auto;color:var(--mut)}canvas{display:block;width:100vw;height:calc(100vh - 56px)}
.badge{padding:3px 10px;border-radius:12px;font-weight:600}</style></head><body><header>
<div class=k><span class=d style="background:var(--blue)"></span>ARCore alone</div>
<div class=k><span class=d style="background:var(--red)"></span>Standard tracker</div>
<div class=k><span class=d style="background:var(--green)"></span>Path Finder</div>
<span id=ours class=badge></span><span id=st></span></header><canvas id=c></canvas><script>
const c=document.getElementById('c'),x=c.getContext('2d');let S={paths:{},info:{}};
const col={arcore:'#4c8df6',standard:'#ef5350',ours:'#34c46a'};
function fit(){c.width=c.clientWidth*devicePixelRatio;c.height=c.clientHeight*devicePixelRatio}
addEventListener('resize',fit);fit();
const es=new EventSource('/events');es.onmessage=e=>{S=JSON.parse(e.data)};
function draw(){x.fillStyle='#0e1116';x.fillRect(0,0,c.width,c.height);
 const all=Object.values(S.paths).flat();if(all.length){
 const ours=S.paths.ours||[],cx0=ours.length?ours[ours.length-1]:[0,0];
 let sc=Math.min(c.width,c.height)/8;   // 8 m across
 const T=p=>[c.width/2+(p[0]-cx0[0])*sc,c.height/2-(p[1]-cx0[1])*sc];
 x.strokeStyle='#22272e';x.lineWidth=1;for(let i=-20;i<=20;i++){const a=T([cx0[0]+i,cx0[1]-20]),b=T([cx0[0]+i,cx0[1]+20]);
 x.beginPath();x.moveTo(a[0],a[1]);x.lineTo(b[0],b[1]);x.stroke();const e=T([cx0[0]-20,cx0[1]+i]),f=T([cx0[0]+20,cx0[1]+i]);
 x.beginPath();x.moveTo(e[0],e[1]);x.lineTo(f[0],f[1]);x.stroke()}
 for(const k of ['arcore','standard','ours']){const P=S.paths[k]||[];if(!P.length)continue;
  x.strokeStyle=col[k];x.lineWidth=3*devicePixelRatio;x.beginPath();P.forEach((p,i)=>{const q=T(p);i?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1])});x.stroke();
  const q=T(P[P.length-1]);x.fillStyle=col[k];x.beginPath();x.arc(q[0],q[1],8*devicePixelRatio,0,7);x.fill();
  if(k=='ours'&&S.info.ours_std){x.strokeStyle='rgba(52,196,106,.5)';x.beginPath();x.arc(q[0],q[1],Math.max(8,S.info.ours_std*sc),0,7);x.stroke()}}}
 const I=S.info||{},b=document.getElementById('ours');b.textContent='Path Finder: '+(I.ours_reason||'waiting');
 b.style.background=I.ours_reason=='ok'?'#1e5631':'#5c4a10';
 document.getElementById('st').textContent=`ARCore: ${I.arcore_tracking?'tracking':(I.arcore||'-')} · IMU ${I.imu_hz||0} Hz · camera ${I.ar_hz||0} Hz`+(I.sim_bias?' · SIMULATED IMU BIAS ON':'')+(I.ours_std?` · ±${I.ours_std.toFixed(2)} m`:'');
 requestAnimationFrame(draw)}draw();</script></body></html>"""


class Web(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    with LOCK:
                        data = json.dumps(STATE)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(PAGE.encode())


def serve_phone(live, record):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 7777))
    srv.listen(1)
    print("waiting for the phone on port 7777 ...", flush=True)
    while True:
        conn, _ = srv.accept()
        print("phone connected", flush=True)
        buf = b""
        with conn, open(record, "a") as rec:
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    print("phone disconnected", flush=True)
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    if ln.strip():
                        rec.write(ln.decode() + "\n")
                        live.line(json.loads(ln))


def replay(live, path, speed=1.0):
    lines = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    t0, w0 = lines[0]["t"], time.time()
    for m in lines:
        delay = (m["t"] - t0) / 1e9 / speed - (time.time() - w0)
        if delay > 0:
            time.sleep(delay)
        live.line(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="also use the learned motion model")
    ap.add_argument("--replay", help="replay a saved session instead of a live phone")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()
    live = Live(args.model)
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", args.port), Web).serve_forever, daemon=True).start()
    print(f"open http://localhost:{args.port}", flush=True)
    if args.replay:
        replay(live, args.replay)
        print("replay finished; page stays up (Ctrl+C to stop)", flush=True)
        while True:
            time.sleep(1)
    out = Path("runs/live")
    out.mkdir(parents=True, exist_ok=True)
    serve_phone(live, out / time.strftime("%Y%m%d-%H%M%S.jsonl"))


if __name__ == "__main__":
    main()
