// On-device timing for Path Finder (issue #3 E): the learned motion network via ONNX Runtime, plus
// the filter's per-step arithmetic, on one CPU core. Built with the Android NDK, run over adb:
//   bench <model.onnx> [runs] [threads]
// Prints JSON: latency percentiles (ms) for the network and for one second of filter work
// (200 IMU propagations of a 15-state covariance + 10 learned updates + 20 visual 6-DoF updates).
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
static double pct(std::vector<double> v, double p) { std::sort(v.begin(), v.end()); return v[std::min(v.size() - 1, (size_t)(p / 100.0 * v.size()))]; }

// Dense helpers for the filter cost (n x n row-major).
static void matmul(const double* A, const double* B, double* C, int n, int m, int k) {
  for (int i = 0; i < n; ++i) for (int j = 0; j < k; ++j) { double s = 0; for (int t = 0; t < m; ++t) s += A[i * m + t] * B[t * k + j]; C[i * k + j] = s; }
}

static double filter_second(std::mt19937& g) {
  const int n = 15, nc = 27;               // 15 states + 2 visual clones (6 each)
  std::normal_distribution<double> N(0, 1);
  std::vector<double> P(nc * nc, 0), F(n * n), T1(n * n), T2(n * n), H(6 * nc), S(36), K(nc * 6), tmp(nc * nc);
  for (int i = 0; i < nc; ++i) P[i * nc + i] = 1e-3;
  for (auto& x : F) x = 0.001 * N(g); for (int i = 0; i < n; ++i) F[i * n + i] += 1;
  for (auto& x : H) x = N(g);
  volatile double sink = 0;
  auto t0 = clk::now();
  for (int step = 0; step < 200; ++step) {           // IMU propagation: P = F P F^T + Q (15x15 block)
    std::vector<double> Pb(n * n); for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) Pb[i * n + j] = P[i * nc + j];
    matmul(F.data(), Pb.data(), T1.data(), n, n, n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) T2[j * n + i] = F[i * n + j];
    matmul(T1.data(), T2.data(), Pb.data(), n, n, n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) P[i * nc + j] = Pb[i * n + j] + (i == j ? 1e-6 : 0);
  }
  for (int u = 0; u < 30; ++u) {                     // 10 learned + 20 visual updates: K = P H^T S^-1, P -= K H P
    int m = (u < 10) ? 3 : 6;
    std::vector<double> Ht(nc * m); for (int i = 0; i < m; ++i) for (int j = 0; j < nc; ++j) Ht[j * m + i] = H[i * nc + j];
    std::vector<double> PHt(nc * m); matmul(P.data(), Ht.data(), PHt.data(), nc, nc, m);
    matmul(H.data(), PHt.data(), S.data(), m, nc, m);
    for (int i = 0; i < m; ++i) S[i * m + i] += 1e-3;
    for (int c = 0; c < m; ++c) { double d = S[c * m + c]; for (int r = 0; r < nc; ++r) K[r * m + c] = PHt[r * m + c] / d; }  // diagonal approx of S^-1
    std::vector<double> HP(m * nc); matmul(H.data(), P.data(), HP.data(), m, nc, nc);
    matmul(K.data(), HP.data(), tmp.data(), nc, m, nc);
    for (int i = 0; i < nc * nc; ++i) P[i] -= 1e-6 * tmp[i];
  }
  sink = P[0];
  return ms(t0, clk::now()) + 0 * sink;
}

int main(int argc, char** argv) {
  const char* model = argc > 1 ? argv[1] : "dispnet_faults.onnx";
  int runs = argc > 2 ? atoi(argv[2]) : 2000, threads = argc > 3 ? atoi(argv[3]) : 1;
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "pathfinder");
  Ort::SessionOptions so; so.SetIntraOpNumThreads(threads); so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  auto t_load = clk::now(); Ort::Session sess(env, model, so); double load_ms = ms(t_load, clk::now());
  std::vector<int64_t> shape{1, 6, 200}; std::vector<float> x(6 * 200);
  std::mt19937 g(0); std::normal_distribution<float> N(0, 1); for (auto& v : x) v = N(g);
  auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mem, x.data(), x.size(), shape.data(), shape.size());
  const char* in_names[] = {"imu"}; const char* out_names[] = {"disp", "logstd"};
  for (int i = 0; i < 50; ++i) sess.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 2);
  std::vector<double> net; net.reserve(runs);
  for (int i = 0; i < runs; ++i) { auto a = clk::now(); auto o = sess.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 2); net.push_back(ms(a, clk::now())); }
  std::vector<double> filt; for (int i = 0; i < 200; ++i) filt.push_back(filter_second(g));
  double p50 = pct(net, 50), filt50 = pct(filt, 50);
  printf("{\"model\":\"%s\",\"threads\":%d,\"runs\":%d,\"load_ms\":%.2f,"
         "\"net_ms\":{\"p50\":%.3f,\"p95\":%.3f,\"p99\":%.3f,\"max\":%.3f},"
         "\"filter_ms_per_second_of_data\":{\"p50\":%.3f,\"p95\":%.3f},"
         "\"cpu_share_one_core_pct\":%.2f}\n",
         model, threads, runs, load_ms, p50, pct(net, 95), pct(net, 99), pct(net, 100),
         filt50, pct(filt, 95), (10 * p50 + filt50) / 1000.0 * 100.0);
  return 0;
}
