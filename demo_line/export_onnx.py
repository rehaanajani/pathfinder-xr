"""Export the displacement network to ONNX (FP32 and dynamic INT8) and time it with ONNX Runtime.

    python export_onnx.py

ONNX is the hand-off format for Android: ONNX Runtime Mobile on the CPU, or Qualcomm's QNN
execution provider / AI Engine Direct for the Hexagon NPU on Snapdragon XR2+ Gen 2. Writes
models/dispnet_faults.onnx, models/dispnet_faults_int8.onnx and results/onnx_benchmark.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from pathfinder.model import WINDOW, DispNet


def bench(sess, x, n=500):
    name = sess.get_inputs()[0].name
    for _ in range(50):
        sess.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(n):
        sess.run(None, {name: x})
    return (time.perf_counter() - t) / n * 1000


def main():
    ck = torch.load("models/dispnet_faults/best.pt", map_location="cpu", weights_only=True)
    net = DispNet(ck["width"]).eval()
    net.load_state_dict(ck["state"])
    x = torch.randn(1, 6, WINDOW)
    fp32 = Path("models/dispnet_faults.onnx")
    torch.onnx.export(net, x, fp32, input_names=["imu"], output_names=["disp", "logstd"], opset_version=17,
                      dynamo=False)
    int8 = Path("models/dispnet_faults_int8.onnx")
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1                      # one core, like a background tracking thread
    s32 = ort.InferenceSession(str(fp32), opts, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(str(int8), opts, providers=["CPUExecutionProvider"])
    xn = x.numpy()
    with torch.no_grad():
        ref = net(x)[0].numpy()
    o32 = s32.run(None, {"imu": xn})[0]
    o8 = s8.run(None, {"imu": xn})[0]

    # Accuracy of the INT8 model on real held-out windows from the TLIO test split.
    from pathfinder import data_tlio as D
    d = np.load(D.ROOT / D.split("test")[0] / "imu0_resampled.npy")
    imu = np.concatenate([d[:, 1:4], d[:, 4:7]], 1).astype(np.float32)
    starts = np.arange(0, len(imu) - WINDOW - 1, 200)[:200]
    win = np.stack([imu[s:s + WINDOW].T for s in starts])
    tgt = d[starts + WINDOW, 11:14] - d[starts, 11:14]
    p32 = np.concatenate([s32.run(None, {"imu": w[None]})[0] for w in win])
    p8 = np.concatenate([s8.run(None, {"imu": w[None]})[0] for w in win])

    res = {
        "fp32_mb": fp32.stat().st_size / 1e6,
        "int8_mb": int8.stat().st_size / 1e6,
        "fp32_ms_1core": bench(s32, xn),
        "int8_ms_1core": bench(s8, xn),
        "max_abs_diff_torch_vs_onnx_m": float(np.abs(o32 - ref).max()),
        "max_abs_diff_int8_vs_fp32_m": float(np.abs(o8 - o32).max()),
        "rmse_fp32_m": float(np.sqrt(np.mean(np.sum((p32 - tgt) ** 2, 1)))),
        "rmse_int8_m": float(np.sqrt(np.mean(np.sum((p8 - tgt) ** 2, 1)))),
        "windows_scored": int(len(win)),
        "device": "Apple M5 laptop, 1 CPU thread, ONNX Runtime " + ort.__version__,
    }
    Path("results/onnx_benchmark.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
