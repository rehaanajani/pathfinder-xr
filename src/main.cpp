// Path Finder — Track 3 entry (C++): degraded IMU CSV -> 6-DoF trajectory (TUM format).
//
//     ./pathfinder <input> [-o estimated_trajectory.txt] [--extend-blind 5] [--model velnet_weights.bin] [...]
//
// A line-by-line port of src/main.py (same options, same arithmetic in double precision, same output); main.py is
// the readable reference and the documentation of every step. This build exists for the budgets of the brief:
// <= 8 MB peak RAM and <= 20 ms per second of IMU on one thread. It streams the input (folder, .zip or the CSV
// itself), keeps a constant-size state and needs only the C++ standard library and zlib (for zipped input).
//
// By default the learned velocity network is loaded from velnet_weights.bin next to the binary (--model none: off).
// Build: make -C src   (or: c++ -O2 -std=c++17 src/main.cpp -lz -o pathfinder)

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <zlib.h>

static const double G = 9.80665;
static const double MOUNT_G_TOL = 0.05;     // see main.py
// Gravity in the IMU frame at the start of the training recordings (EuRoC, TUM-VI); see main.py TRAIN_UP.
static const double TRAIN_UP[2][3] = {{0.9412, -0.0101, -0.3376}, {0.0160, -0.0040, 0.9999}};
// Rotor drag (see main.py DRONE_UP, DRAG_E1, DRAG_E2): thrust axis of the EuRoC airframe and the sideways plane.
struct DragAxes {
    double up[3], e1[3], e2[3];
    DragAxes() {
        double n = std::sqrt(0.9412 * 0.9412 + 0.0101 * 0.0101 + 0.3376 * 0.3376);
        up[0] = 0.9412 / n; up[1] = -0.0101 / n; up[2] = -0.3376 / n;
        double e[3] = {-up[0] * up[1], 1.0 - up[1] * up[1], -up[2] * up[1]};
        double m = 0;
        for (double c : e) m += c * c;
        m = std::sqrt(m);
        for (int i = 0; i < 3; i++) e1[i] = e[i] / m;
        e2[0] = up[1] * e1[2] - up[2] * e1[1];
        e2[1] = up[2] * e1[0] - up[0] * e1[2];
        e2[2] = up[0] * e1[1] - up[1] * e1[0];
    }
};
static const DragAxes DRAG;
static const double NaN = std::nan("");

// ------------------------------------------------------------------------------------------------ small math
struct Q { double w, x, y, z; };
struct V3 { double x[3]; double& operator[](int i) { return x[i]; } double operator[](int i) const { return x[i]; } };

static Q qmul(const Q& a, const Q& b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
static Q qnorm(const Q& q) {
    double n = std::sqrt(0.0 + q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (n == 0.0) n = 1.0;
    return {q.w / n, q.x / n, q.y / n, q.z / n};
}
static Q qexp(double w0, double w1, double w2) {
    double th = std::sqrt(w0 * w0 + w1 * w1 + w2 * w2);
    if (th < 1e-12) return {1.0, 0.5 * w0, 0.5 * w1, 0.5 * w2};
    double s = std::sin(0.5 * th) / th;
    return {std::cos(0.5 * th), w0 * s, w1 * s, w2 * s};
}
static V3 qrot(const Q& q, const V3& v) {
    double w = q.w, x = q.x, y = q.y, z = q.z;
    double tx = 2 * (y * v[2] - z * v[1]), ty = 2 * (z * v[0] - x * v[2]), tz = 2 * (x * v[1] - y * v[0]);
    return {{v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx}};
}
static Q q_from_two(const V3& u, const V3& v) {
    double d = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];
    V3 c = {{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]}};
    if (d < -0.999999) {
        V3 ax = std::fabs(u[0]) < 0.9 ? V3{{1.0, 0.0, 0.0}} : V3{{0.0, 1.0, 0.0}};
        c = {{u[1] * ax[2] - u[2] * ax[1], u[2] * ax[0] - u[0] * ax[2], u[0] * ax[1] - u[1] * ax[0]}};
        return qnorm({0.0, c[0], c[1], c[2]});
    }
    return qnorm({1.0 + d, c[0], c[1], c[2]});
}
static double norm3(const V3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }
static double sq(double x) { return x * x; }
static double pyround(double x) { return std::nearbyint(x); }   // Python round(): half to even

// ------------------------------------------------------------------------------------------------ options
struct Opt {
    std::string input, output = "estimated_trajectory.txt", model = "auto", preset, fallback_model = "auto";
    std::string columns = "auto", ts_unit = "auto", accel_unit = "auto";
    double mount_guard_deg = 25.0;
    double extend_blind = 0; int init_samples = 200; double still_gyro_var = 4e-4; int zupt_window = 40;
    double zupt_gyro = 0.04, zupt_acc = 0.15, zupt_tau = 0.05, zupt_gyro_std = NaN, zupt_acc_std = 0.15;
    double prescan = NaN, bias_rate = NaN, leak = 1.0, kp = NaN, ki = NaN, kp_boost = NaN, ki_calm = NaN;
    double tilt_lp = NaN, kp_boost_tau = 0.7, tilt_acc_tol = 1.5, acc_res_tau = 10.0, v_max = 3.0, pos_tau = NaN;
    double blind_vel_tau = 1.5, blind_turn_tau = 0.5, pos_tau_blind = NaN, pos_tau_nonet = NaN;
    double drag_k1 = -0.19, drag_k2 = -0.17, drag_tau = 0.5, drag_lp = 0.3, drag_robust = 2.0, drag_hp = 0.0;
    double drag_pos_tau = 12.0, drag_mount_deg = 15.0;
    int model_every = 20; double model_min_valid = 0.25, model_ref = NaN, model_vref = 1.0;
    double model_pos_gain = NaN, model_vel_gain = 1.0; int model_window = 200, model_dec = 1;
    std::string model_input = "world";
    double yaw_q = 0.0, yaw_q_blind = 2.0, model_blind_tau = 0.7, acc_max = 16 * G, gyro_max = 35.0, acc_min = 1.0;
};

static void usage_exit(const char* msg) {
    std::fprintf(stderr, "usage: pathfinder <input> [-o out.txt] [--extend-blind S] [--model weights.bin] "
                         "[--preset physics|learned] [options of src/main.py]\nerror: %s\n", msg);
    std::exit(2);
}

static void parse_args(int argc, char** argv, Opt& a) {
    struct D { const char* n; double* d; int* i; std::string* s; };
    D table[] = {
        {"output", nullptr, nullptr, &a.output}, {"model", nullptr, nullptr, &a.model},
        {"fallback-model", nullptr, nullptr, &a.fallback_model},
        {"columns", nullptr, nullptr, &a.columns}, {"ts-unit", nullptr, nullptr, &a.ts_unit},
        {"accel-unit", nullptr, nullptr, &a.accel_unit}, {"mount-guard-deg", &a.mount_guard_deg, nullptr, nullptr},
        {"preset", nullptr, nullptr, &a.preset}, {"extend-blind", &a.extend_blind, nullptr, nullptr},
        {"init-samples", nullptr, &a.init_samples, nullptr}, {"still-gyro-var", &a.still_gyro_var, nullptr, nullptr},
        {"zupt-window", nullptr, &a.zupt_window, nullptr}, {"zupt-gyro", &a.zupt_gyro, nullptr, nullptr},
        {"zupt-acc", &a.zupt_acc, nullptr, nullptr}, {"zupt-tau", &a.zupt_tau, nullptr, nullptr},
        {"zupt-gyro-std", &a.zupt_gyro_std, nullptr, nullptr}, {"zupt-acc-std", &a.zupt_acc_std, nullptr, nullptr},
        {"prescan", &a.prescan, nullptr, nullptr}, {"bias-rate", &a.bias_rate, nullptr, nullptr},
        {"leak", &a.leak, nullptr, nullptr}, {"kp", &a.kp, nullptr, nullptr}, {"ki", &a.ki, nullptr, nullptr},
        {"kp-boost", &a.kp_boost, nullptr, nullptr}, {"ki-calm", &a.ki_calm, nullptr, nullptr},
        {"tilt-lp", &a.tilt_lp, nullptr, nullptr}, {"kp-boost-tau", &a.kp_boost_tau, nullptr, nullptr},
        {"tilt-acc-tol", &a.tilt_acc_tol, nullptr, nullptr}, {"acc-res-tau", &a.acc_res_tau, nullptr, nullptr},
        {"v-max", &a.v_max, nullptr, nullptr}, {"pos-tau", &a.pos_tau, nullptr, nullptr}, {"pos-tau-blind", &a.pos_tau_blind, nullptr, nullptr},
        {"pos-tau-nonet", &a.pos_tau_nonet, nullptr, nullptr},
        {"drag-k1", &a.drag_k1, nullptr, nullptr}, {"drag-k2", &a.drag_k2, nullptr, nullptr},
        {"drag-tau", &a.drag_tau, nullptr, nullptr}, {"drag-lp", &a.drag_lp, nullptr, nullptr},
        {"drag-robust", &a.drag_robust, nullptr, nullptr}, {"drag-hp", &a.drag_hp, nullptr, nullptr},
        {"drag-pos-tau", &a.drag_pos_tau, nullptr, nullptr}, {"drag-mount-deg", &a.drag_mount_deg, nullptr, nullptr},
        {"blind-vel-tau", &a.blind_vel_tau, nullptr, nullptr}, {"blind-turn-tau", &a.blind_turn_tau, nullptr, nullptr},
        {"model-every", nullptr, &a.model_every, nullptr}, {"model-min-valid", &a.model_min_valid, nullptr, nullptr},
        {"model-ref", &a.model_ref, nullptr, nullptr}, {"model-vref", &a.model_vref, nullptr, nullptr},
        {"model-pos-gain", &a.model_pos_gain, nullptr, nullptr}, {"model-vel-gain", &a.model_vel_gain, nullptr, nullptr},
        {"model-window", nullptr, &a.model_window, nullptr}, {"model-dec", nullptr, &a.model_dec, nullptr},
        {"model-input", nullptr, nullptr, &a.model_input},
        {"yaw-q", &a.yaw_q, nullptr, nullptr}, {"yaw-q-blind", &a.yaw_q_blind, nullptr, nullptr},
        {"model-blind-tau", &a.model_blind_tau, nullptr, nullptr}, {"acc-max", &a.acc_max, nullptr, nullptr}, {"acc-min", &a.acc_min, nullptr, nullptr},
        {"gyro-max", &a.gyro_max, nullptr, nullptr},
    };
    bool have_input = false;
    for (int k = 1; k < argc; k++) {
        std::string s = argv[k], name, val;
        bool has_val = false;
        if (s == "-h" || s == "--help") usage_exit("help");
        if (s == "-o") s = "--output";
        if (s.rfind("--", 0) == 0) {
            name = s.substr(2);
            size_t eq = name.find('=');
            if (eq != std::string::npos) { val = name.substr(eq + 1); name = name.substr(0, eq); has_val = true; }
            D* d = nullptr;
            for (auto& t : table) if (name == t.n) d = &t;
            if (!d) usage_exit(("unknown option --" + name).c_str());
            if (!has_val) { if (k + 1 >= argc) usage_exit(("missing value for --" + name).c_str()); val = argv[++k]; }
            char* end = nullptr;
            if (d->s) *d->s = val;
            else if (d->d) { *d->d = std::strtod(val.c_str(), &end); if (*end) usage_exit(("bad number " + val).c_str()); }
            else { *d->i = (int)std::strtol(val.c_str(), &end, 10); if (*end) usage_exit(("bad integer " + val).c_str()); }
        } else if (!have_input) { a.input = s; have_input = true; }
        else usage_exit(("unexpected argument " + s).c_str());
    }
    if (!have_input) usage_exit("the input is required");
    if (a.model == "auto") {                    // default: velnet_weights.bin next to the program, if present
        std::string self = argv[0];
        size_t sl = self.find_last_of("/\\");
        std::string cand = (sl == std::string::npos ? std::string(".") : self.substr(0, sl)) + "/velnet_weights.bin";
        struct stat st;
        a.model = stat(cand.c_str(), &st) == 0 ? cand : "";
    } else if (a.model == "none") a.model = "";
    if (!std::isfinite(a.mount_guard_deg)) usage_exit("--mount-guard-deg must be a finite number");
    if (a.columns != "auto" && a.columns != "gyro-first" && a.columns != "accel-first")
        usage_exit("--columns must be auto, gyro-first or accel-first");
    if (a.ts_unit != "auto" && a.ts_unit != "ns" && a.ts_unit != "us" && a.ts_unit != "ms" && a.ts_unit != "s")
        usage_exit("--ts-unit must be auto, ns, us, ms or s");
    if (a.accel_unit != "auto" && a.accel_unit != "ms2" && a.accel_unit != "g")
        usage_exit("--accel-unit must be auto, ms2 or g");
    if (a.fallback_model == "auto") {           // the gravity-aligned network, used when the mount guard fires
        std::string self = argv[0];
        size_t sl = self.find_last_of("/\\");
        std::string cand = (sl == std::string::npos ? std::string(".") : self.substr(0, sl)) + "/velnet_weights_world.bin";
        struct stat st;
        a.fallback_model = stat(cand.c_str(), &st) == 0 ? cand : "";
    } else if (a.fallback_model == "none") a.fallback_model = "";
    std::string p = a.preset.empty() ? (a.model.empty() ? "physics" : "learned") : a.preset;
    if (p != "physics" && p != "learned") usage_exit("--preset must be physics or learned");
    bool L = p == "learned";
    auto def = [](double& x, double phys, double learn, bool l) { if (std::isnan(x)) x = l ? learn : phys; };
    def(a.kp, 1.0, 0.3, L); def(a.ki, 0.02, 0.1, L); def(a.kp_boost, 0.0, 10.0, L); def(a.tilt_lp, 0.0, 0.1, L);
    def(a.ki_calm, 0.0, 0.5, L); def(a.zupt_gyro_std, 0.0, 0.01, L); def(a.bias_rate, 0.2, 3.0, L);
    def(a.prescan, 0.0, 60.0, L); def(a.pos_tau, 0.0, 3.0, L);
    // Network coupling chosen on the organisers' degradation (16-17 Nov, validation, then held-out): trust the
    // network's displacement for position, pull back to the start more gently, and firmly once the IMU is dead.
    def(a.pos_tau_blind, 0.0, 0.0, L); def(a.pos_tau_nonet, 0.0, 3.0, L); def(a.model_pos_gain, 0.0, 0.0, L); def(a.model_ref, 0.1, 0.1, L);
}

// ------------------------------------------------------------------------------------------------ input
// Line source over a plain file or one member of a zip archive (stored or deflated), read in small chunks.
struct Source {
    FILE* f = nullptr;
    bool zip = false, zdone = false;
    z_stream zs{};
    long long zleft = 0;            // compressed bytes left to read
    int method = 0;
    unsigned char in[1 << 14], out[1 << 14];
    size_t opos = 0, olen = 0;
    std::string line;

    size_t fill() {                 // refill `out`; 0 at end of data
        opos = olen = 0;
        if (!zip) { olen = std::fread(out, 1, sizeof out, f); return olen; }
        if (method == 0) {
            size_t n = (size_t)std::min<long long>(zleft, (long long)sizeof out);
            olen = n ? std::fread(out, 1, n, f) : 0;
            zleft -= (long long)olen;
            return olen;
        }
        while (olen == 0 && !zdone) {
            if (zs.avail_in == 0 && zleft > 0) {
                size_t n = (size_t)std::min<long long>(zleft, (long long)sizeof in);
                n = std::fread(in, 1, n, f);
                if (n == 0) { zdone = true; break; }
                zleft -= (long long)n;
                zs.next_in = in; zs.avail_in = (uInt)n;
            }
            zs.next_out = out; zs.avail_out = sizeof out;
            int r = inflate(&zs, Z_NO_FLUSH);
            olen = sizeof out - zs.avail_out;
            if (r == Z_STREAM_END || (r != Z_OK && r != Z_BUF_ERROR) || (r == Z_BUF_ERROR && zleft == 0 && olen == 0))
                zdone = true;
        }
        return olen;
    }
    bool getline() {                // next line without the newline; false at end
        line.clear();
        for (;;) {
            if (opos >= olen && fill() == 0) return !line.empty();
            unsigned char* s = out + opos;
            unsigned char* nl = (unsigned char*)std::memchr(s, '\n', olen - opos);
            if (nl) { line.append((char*)s, nl - s); opos += (nl - s) + 1; return true; }
            line.append((char*)s, olen - opos); opos = olen;
        }
    }
};

static uint32_t le32(const unsigned char* p) { return p[0] | p[1] << 8 | p[2] << 16 | (uint32_t)p[3] << 24; }
static uint16_t le16(const unsigned char* p) { return (uint16_t)(p[0] | p[1] << 8); }

static void open_zip_member(Source& src, const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path); std::exit(1); }
    std::fseek(f, 0, SEEK_END);
    long size = std::ftell(f);
    long tail = std::min<long>(size, 65557);
    std::vector<unsigned char> buf(tail);
    std::fseek(f, size - tail, SEEK_SET);
    if (std::fread(buf.data(), 1, tail, f) != (size_t)tail) { std::fprintf(stderr, "bad zip\n"); std::exit(1); }
    long e = -1;
    for (long i = tail - 22; i >= 0; i--) if (le32(&buf[i]) == 0x06054b50) { e = i; break; }
    if (e < 0) { std::fprintf(stderr, "%s: not a zip file\n", path); std::exit(1); }
    uint32_t cd_size = le32(&buf[e + 12]), cd_off = le32(&buf[e + 16]);
    std::vector<unsigned char>().swap(buf);
    const std::string want = "imu0/data.csv";
    // Walk the central directory one entry at a time: memory stays constant however many files the zip holds.
    long pos = cd_off, end = (long)cd_off + (long)cd_size;
    std::string name;
    while (pos + 46 <= end) {
        unsigned char h[46];
        std::fseek(f, pos, SEEK_SET);
        if (std::fread(h, 1, 46, f) != 46 || le32(h) != 0x02014b50) break;
        uint16_t method = le16(h + 10), nlen = le16(h + 28), xlen = le16(h + 30), clen = le16(h + 32);
        uint32_t csize = le32(h + 20), loff = le32(h + 42);
        name.resize(nlen);
        if (nlen && std::fread(&name[0], 1, nlen, f) != nlen) break;
        pos += 46 + nlen + xlen + clen;
        if (name.size() >= want.size() && name.compare(name.size() - want.size(), want.size(), want) == 0) {
            unsigned char lh[30];
            std::fseek(f, loff, SEEK_SET);
            if (std::fread(lh, 1, 30, f) != 30 || le32(lh) != 0x04034b50) { std::fprintf(stderr, "bad zip entry\n"); std::exit(1); }
            std::fseek(f, loff + 30 + le16(lh + 26) + le16(lh + 28), SEEK_SET);
            src.f = f; src.zip = true; src.method = method; src.zleft = csize;
            if (method == 8) inflateInit2(&src.zs, -15);
            else if (method != 0) { std::fprintf(stderr, "unsupported zip compression %d\n", method); std::exit(1); }
            return;
        }
    }
    std::fprintf(stderr, "no imu0/data.csv in %s\n", path);
    std::exit(1);
}

static bool is_dir(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode); }
static bool exists(const std::string& p) { struct stat st; return stat(p.c_str(), &st) == 0; }

static void open_source(Source& src, const std::string& path) {
    if (is_dir(path)) {
        for (const char* c : {"/imu0/data.csv", "/mav0/imu0/data.csv"}) {
            std::string cand = path + c;
            if (exists(cand)) { src.f = std::fopen(cand.c_str(), "rb"); if (src.f) return; }
        }
        std::fprintf(stderr, "no imu0/data.csv under %s\n", path.c_str());
        std::exit(1);
    }
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".zip") == 0) { open_zip_member(src, path.c_str()); return; }
    src.f = std::fopen(path.c_str(), "rb");
    if (!src.f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
}

static void close_source(Source& src) {
    if (src.zip && src.method == 8) inflateEnd(&src.zs);
    if (src.f) std::fclose(src.f);
    src = Source{};
}

static bool py_space(unsigned char c) { return c == ' ' || (c >= 9 && c <= 13) || (c >= 0x1c && c <= 0x1f); }

static std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && py_space((unsigned char)s[a])) a++;
    while (b > a && py_space((unsigned char)s[b - 1])) b--;
    return s.substr(a, b - a);
}

static bool py_float(const std::string& s0, double& out) {      // Python float(): surrounding whitespace allowed
    std::string s = trim(s0);
    if (s.empty()) return false;
    if (s.find('x') != std::string::npos || s.find('X') != std::string::npos) return false;   // no hex floats
    char* end = nullptr;
    errno = 0;
    double v = std::strtod(s.c_str(), &end);
    if (*end != '\0') return false;
    out = v;
    return true;
}

static bool py_int(const std::string& s0, long long& out) {
    std::string s = trim(s0);
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    long long v = std::strtoll(s.c_str(), &end, 10);
    if (*end != '\0' || errno == ERANGE) return false;
    out = v;
    return true;
}

struct Row { long long t; V3 g, a; };

// Column order and units of the input (see main.py detect_layout): EuRoC is ns, gyro first, accel in m/s^2; the
// organisers' degradation script documents accel-first columns and accepts s/ms/us/ns timestamps and accel in g.
struct Layout { bool accel_first = false; long long t_scale = 1; double a_scale = 1.0; };
static Layout g_layout;

// One data row as main.py's parse_rows reads it; false for lines that are skipped.
static bool parse_line(const std::string& raw, Row& r) {
    std::string s = trim(raw);
    if (s.empty()) return false;
    unsigned char c0 = (unsigned char)s[0];
    if (c0 == '#' || std::isalpha(c0) || c0 >= 0x80) return false;
    for (char& c : s) if (c == ';') c = ',';
    std::vector<std::string> parts;
    size_t st = 0;
    for (;;) {
        size_t k = s.find(',', st);
        parts.push_back(s.substr(st, k == std::string::npos ? std::string::npos : k - st));
        if (k == std::string::npos) break;
        st = k + 1;
    }
    const std::string& p0 = parts[0];
    bool fl = false;
    for (char c : p0) if (c == 'e' || c == 'E' || c == '.') fl = true;
    if (g_layout.t_scale != 1) {                 // timestamps in s / ms / us: convert to ns (Python: round(v * scale))
        double v;
        if (!py_float(p0, v) || !std::isfinite(v)) return false;
        v = std::nearbyint(v * (double)g_layout.t_scale);
        if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0)) return false;
        r.t = (long long)v;
    } else if (fl) {
        double v;
        if (!py_float(p0, v) || !std::isfinite(v)) return false;
        v = std::trunc(v);
        if (!(v >= -9223372036854775808.0 && v < 9223372036854775808.0)) return false;   // must fit in int64
        r.t = (long long)v;
    } else if (!py_int(p0, r.t)) return false;
    double vals[6];
    for (int i = 0; i < 6; i++) {
        vals[i] = NaN;
        if (i + 1 < (int)parts.size()) { double v; if (py_float(parts[i + 1], v)) vals[i] = v; }
    }
    int gi = g_layout.accel_first ? 3 : 0, ai = g_layout.accel_first ? 0 : 3;
    r.g = {{vals[gi], vals[gi + 1], vals[gi + 2]}};
    r.a = {{vals[ai], vals[ai + 1], vals[ai + 2]}};
    if (g_layout.a_scale != 1.0)
        for (int i = 0; i < 3; i++) r.a[i] = r.a[i] * g_layout.a_scale;
    return true;
}

static double median_of(std::vector<double> z) {
    std::sort(z.begin(), z.end());
    return z[z.size() / 2];
}

// First data rows -> column order and units (same rules and thresholds as main.py detect_layout; each can be forced).
static Layout detect_layout(const std::string& path, const std::string& columns, const std::string& ts_unit,
                            const std::string& accel_unit, std::string& note, size_t n_max = 2000) {
    Layout L;
    Source src;
    open_source(src, path);
    std::vector<double> t, x, y;
    while (t.size() < n_max && src.getline()) {
        std::string s = trim(src.line);
        if (s.empty()) continue;
        unsigned char c0 = (unsigned char)s[0];
        if (c0 == '#' || std::isalpha(c0) || c0 >= 0x80) continue;
        for (char& c : s) if (c == ';') c = ',';
        std::vector<std::string> parts;
        size_t st = 0;
        for (;;) {
            size_t k = s.find(',', st);
            parts.push_back(s.substr(st, k == std::string::npos ? std::string::npos : k - st));
            if (k == std::string::npos) break;
            st = k + 1;
        }
        double t0;
        if (!py_float(parts[0], t0) || !std::isfinite(t0)) continue;
        t.push_back(t0);
        if (parts.size() < 7) continue;
        double v[6];
        bool ok = true, nonzero = false;
        for (int i = 0; i < 6 && ok; i++) ok = py_float(parts[i + 1], v[i]) && std::isfinite(v[i]);
        if (!ok) continue;
        for (int i = 0; i < 6; i++) if (v[i] != 0.0) nonzero = true;
        if (!nonzero) continue;
        x.push_back(std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
        y.push_back(std::sqrt(v[3] * v[3] + v[4] * v[4] + v[5] * v[5]));
    }
    close_source(src);
    std::vector<std::string> notes;
    auto si = [](double n) { return n >= 6.0 && n <= 14.0; };   // gravity in m/s^2
    auto gu = [](double n) { return n >= 0.6 && n <= 1.4; };    // gravity in g
    bool have = x.size() >= 10;
    double mx = have ? median_of(x) : 0, my = have ? median_of(y) : 0;
    if (columns != "auto") L.accel_first = columns == "accel-first";
    else if (!have) notes.push_back("too few rows to check columns, using gyro first");
    else {
        // a forced accel unit narrows the test to that unit; otherwise m/s^2 first, then g (as main.py)
        bool use_si = accel_unit != "g", use_gu = accel_unit != "ms2", decided = false;
        if (use_si) {
            if (si(mx) != si(my)) { L.accel_first = si(mx); decided = true; }
            else if (si(mx) || si(my)) use_gu = false;           // both look like m/s^2: g cannot settle it
        }
        if (!decided && use_gu && gu(mx) != gu(my)) {
            L.accel_first = gu(mx); decided = true;
            if (accel_unit == "auto") L.a_scale = G;
        }
        if (!decided) notes.push_back("column order unclear, using gyro first");
    }
    if (accel_unit == "g") L.a_scale = G;
    else if (accel_unit == "ms2") L.a_scale = 1.0;
    else if (columns != "auto" && have) {                        // forced order, automatic unit
        double na = L.accel_first ? mx : my;
        L.a_scale = (gu(na) && !si(na)) ? G : 1.0;
    }
    if (ts_unit != "auto") {
        L.t_scale = ts_unit == "s" ? 1000000000LL : ts_unit == "ms" ? 1000000LL : ts_unit == "us" ? 1000LL : 1LL;
    } else if (!t.empty()) {
        double tmax = 0;
        for (double c : t) tmax = std::max(tmax, std::fabs(c));
        if (tmax < 1e17) {
            std::vector<double> dts;
            for (size_t i = 1; i < t.size(); i++) if (t[i] > t[i - 1]) dts.push_back(t[i] - t[i - 1]);
            if (!dts.empty()) {
                double dt = median_of(dts);
                // the unit giving a believable IMU rate, 10 Hz - 10 kHz (see main.py)
                L.t_scale = 1;
                for (long long sc : {1LL, 1000LL, 1000000LL, 1000000000LL})
                    if (1e9 / (dt * (double)sc) >= 10.0 && 1e9 / (dt * (double)sc) < 10000.0) { L.t_scale = sc; break; }
                if (tmax * (double)L.t_scale >= 9.2e18) {
                    L.t_scale = 1;
                    notes.push_back("timestamp unit would overflow, using ns");
                }
            }
        }
    }
    note.clear();
    for (size_t i = 0; i < notes.size(); i++) note += (i ? "; " : "") + notes[i];
    return L;
}

// ------------------------------------------------------------------------------------------------ the network
// Same file format and arithmetic as src/velnet.py (float32 weights and activations, BatchNorm folded in).
struct Conv { int cout, cin, k, stride, pad; std::vector<float> w, b; };
struct Dense { int out, in; std::vector<float> w, b; };

struct VelNet {
    int version = 0, window = 0, dec = 1, n_in = 0;
    float rate = 0;
    std::vector<float> scale;
    std::vector<Conv> convs;
    std::vector<Dense> fcs;
    std::vector<float> x, y, xp, xe, xo;    // activation buffers, reused (xe/xo: even/odd samples for stride 2)
    std::vector<double> acc;

    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) { std::fprintf(stderr, "cannot open model %s\n", path.c_str()); return false; }
        bool ok = true;
        auto i32 = [&]() { int32_t v = 0; ok &= std::fread(&v, 4, 1, f) == 1; return (int)v; };
        auto f32 = [&](std::vector<float>& v, size_t n) { v.resize(n); ok &= std::fread(v.data(), 4, n, f) == n; };
        char magic[4];
        ok &= std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "PFVN", 4) == 0;
        version = i32();            // 2: gravity-aligned input; 3: body-frame input and outputs (AirIO style)
        window = i32(); dec = i32();
        std::vector<float> r; f32(r, 1); rate = ok ? r[0] : 0;
        n_in = i32();
        if (!ok || n_in < 1 || n_in > 64) { std::fclose(f); std::fprintf(stderr, "%s: not a Path Finder velocity-network file\n", path.c_str()); return false; }
        f32(scale, n_in - 1);
        int nc = i32();
        for (int i = 0; i < nc && ok; i++) {
            Conv c; c.cout = i32(); c.cin = i32(); c.k = i32(); c.stride = i32(); c.pad = i32();
            f32(c.w, (size_t)c.cout * c.cin * c.k); f32(c.b, c.cout);
            convs.push_back(std::move(c));
        }
        int nf = i32();
        for (int i = 0; i < nf && ok; i++) {
            Dense d; d.out = i32(); d.in = i32();
            f32(d.w, (size_t)d.out * d.in); f32(d.b, d.out);
            fcs.push_back(std::move(d));
        }
        std::fclose(f);
        if (!ok) std::fprintf(stderr, "%s: truncated model file\n", path.c_str());
        return ok;
    }

    // in: (n_in x window) row-major, already scaled. out: the 12 raw outputs.
    void forward(std::vector<float>& in, float o[12]) {
        int C = n_in, L = window;
        x = in;
        for (const Conv& c : convs) {
            int lout = (L + 2 * c.pad - c.k) / c.stride + 1;
            // Zero-padded copy of the input, then accumulate with the output position innermost (vectorises).
            // For every output the products are summed in the same order (ci, then k) as a direct loop would.
            int Lp = L + 2 * c.pad;
            xp.assign((size_t)c.cin * Lp, 0.0f);
            for (int ci = 0; ci < c.cin; ci++)
                std::memcpy(&xp[(size_t)ci * Lp + c.pad], &x[(size_t)ci * L], sizeof(float) * L);
            acc.resize(lout);
            y.assign((size_t)c.cout * lout, 0.0f);
            // Stride 2: split each padded channel into its even and odd samples once, so the inner loop reads
            // contiguous memory (vectorises); src[2l] with src = xr + k is then half_k[l + k / 2]. Same products, same
            // summation order: bit-identical to the direct loop.
            int Lh = (Lp + 1) / 2;
            if (c.stride == 2) {
                xe.assign((size_t)c.cin * Lh, 0.0f);
                xo.assign((size_t)c.cin * Lh, 0.0f);
                for (int ci = 0; ci < c.cin; ci++)
                    for (int j = 0; j < Lp; j++)
                        ((j & 1) ? xo : xe)[(size_t)ci * Lh + j / 2] = xp[(size_t)ci * Lp + j];
            }
            for (int co = 0; co < c.cout; co++) {
                const float* wc = &c.w[(size_t)co * c.cin * c.k];
                std::fill(acc.begin(), acc.end(), 0.0);
                double* ac = acc.data();
                for (int ci = 0; ci < c.cin; ci++) {
                    const float* xr = &xp[(size_t)ci * Lp];
                    const float* wr = wc + (size_t)ci * c.k;
                    for (int k = 0; k < c.k; k++) {
                        double w = wr[k];
                        if (c.stride == 2) {
                            const float* src = ((k & 1) ? &xo[(size_t)ci * Lh] : &xe[(size_t)ci * Lh]) + k / 2;
                            for (int l = 0; l < lout; l++) ac[l] += w * src[l];
                        } else {
                            const float* src = xr + k;
                            for (int l = 0; l < lout; l++) ac[l] += w * src[l * c.stride];
                        }
                    }
                }
                for (int l = 0; l < lout; l++) {
                    float v = (float)ac[l] + c.b[co];
                    y[(size_t)co * lout + l] = v > 0 ? v : 0.0f;
                }
            }
            x.swap(y);
            C = c.cout; L = lout;
        }
        std::vector<float> z(C);
        for (int ci = 0; ci < C; ci++) {
            double s = 0;
            for (int l = 0; l < L; l++) s += x[(size_t)ci * L + l];
            z[ci] = (float)(s / L);
        }
        for (size_t j = 0; j < fcs.size(); j++) {
            const Dense& d = fcs[j];
            std::vector<float> nz(d.out);
            for (int r = 0; r < d.out; r++) {
                double s = 0;
                for (int i = 0; i < d.in; i++) s += (double)d.w[(size_t)r * d.in + i] * z[i];
                float v = (float)s + d.b[r];
                nz[r] = (j + 1 < fcs.size() && v < 0) ? 0.0f : v;
            }
            z.swap(nz);
        }
        for (int i = 0; i < 12; i++) o[i] = i < (int)z.size() ? z[i] : 0.0f;
    }
};

// Rotate m (per-axis std s) from the body frame into the world frame; s becomes sqrt(diag(R diag(s^2) R^T)).
static void rot_cov(const Q& q, double m[3], double s[3]) {
    V3 mw = qrot(q, {{m[0], m[1], m[2]}});
    V3 c[3] = {qrot(q, {{1.0, 0.0, 0.0}}), qrot(q, {{0.0, 1.0, 0.0}}), qrot(q, {{0.0, 0.0, 1.0}})};
    double sw[3];
    for (int i = 0; i < 3; i++) {
        double acc = 0;
        for (int j = 0; j < 3; j++) acc += sq(c[j][i] * s[j]);
        sw[i] = std::sqrt(acc);
    }
    for (int i = 0; i < 3; i++) { m[i] = mw[i]; s[i] = sw[i]; }
}

// ------------------------------------------------------------------------------------------------ the estimator
struct Feat { bool has = false; double v[10]; };   // world: 7 channels, body: 10 (see main.py push_slot)

struct PathFinder {
    Opt A;                                   // a copy: the mount guard changes the coupling when it switches networks
    Q q{1, 0, 0, 0};
    V3 p{{0, 0, 0}}, v{{0, 0, 0}}, bg{{0, 0, 0}}, ba{{0, 0, 0}}, w_last{{0, 0, 0}}, acc_res{{0, 0, 0}};
    bool have_t_prev = false; long long t_prev = 0;
    std::vector<std::pair<V3, V3>> init_buf;
    bool initialised = false;
    bool have_prev_raw = false; double prev_raw[6]; int frozen_run = 0;
    std::deque<double> still_g, still_a;
    std::deque<std::vector<double>> still_w;
    double still_s[6] = {0, 0, 0, 0, 0, 0}, still_ss[6] = {0, 0, 0, 0, 0, 0};
    V3 still_gyro{{0, 0, 0}};
    double bad_time = 0.0, good_time = 1e9;
    bool bias_known = false;
    bool have_f_lp = false; V3 f_lp{{0, 0, 0}};
    bool drag_on = false, have_f_drag = false, have_f_drag_slow = false;
    V3 f_drag{{0, 0, 0}}, f_drag_slow{{0, 0, 0}};
    double yaw_var = 0.0;
    bool have_dt_nom = false; double dt_nom = 0;
    VelNet* net = nullptr;
    int W, dec, rate_mult = 1;              // rate_mult: input samples averaged per network sample (1 kHz in: 5)
    double v_blind_tau;
    bool ring_made = false;
    std::vector<Feat> ring; std::vector<V3> ring_p;
    bool body = false; int nb = 6;
    double bin[9]; int bin_n = 0, bin_k = 0; long long slots = 0;
    int ring_i = 0, ring_valid = 0; long long since = 0;
    bool net_ok = false;
    V3 p_then{{0, 0, 0}};
    std::vector<float> xin;

    PathFinder(const Opt& a, VelNet* n) : A(a), net(n), W(a.model_window), dec(a.model_dec), v_blind_tau(a.blind_vel_tau) {
        body = A.model_input == "body";
        if (net) {
            W = net->window; dec = net->dec;
            body = net->version >= 3;
            if (A.model_blind_tau > 0) v_blind_tau = A.model_blind_tau;
        }
    }

    bool healthy(const V3& g, const V3& a) {
        double vals[6] = {g[0], g[1], g[2], a[0], a[1], a[2]};
        for (double x : vals) if (!std::isfinite(x)) return false;
        bool allz = true;
        for (double x : vals) if (x != 0.0) allz = false;
        if (allz) return false;
        bool same = have_prev_raw;
        if (same) for (int i = 0; i < 6; i++) if (!(vals[i] == prev_raw[i])) same = false;
        frozen_run = same ? frozen_run + 1 : 0;
        std::memcpy(prev_raw, vals, sizeof vals); have_prev_raw = true;
        if (frozen_run >= 3) return false;
        double na = norm3(a);
        if (na > A.acc_max || norm3(g) > A.gyro_max) return false;
        if (na < A.acc_min) return false;       // no gravity at all: a dead accelerometer, not motion
        return true;
    }

    void init_from() {
        size_t n = init_buf.size();
        V3 ma{{0, 0, 0}}, mg{{0, 0, 0}};
        for (int i = 0; i < 3; i++) {
            double sa = 0, sg = 0;
            for (auto& b : init_buf) { sa += b.second[i]; sg += b.first[i]; }
            ma[i] = sa / n; mg[i] = sg / n;
        }
        double nm = norm3(ma);
        if (nm > 1e-6) {
            q = q_from_two({{ma[0] / nm, ma[1] / nm, ma[2] / nm}}, {{0.0, 0.0, 1.0}});
            check_mount(ma);
            if (A.drag_k1 != 0 && A.drag_k2 != 0 && A.drag_mount_deg > 0) {   // see main.py init_from
                double c = 0;
                for (int i = 0; i < 3; i++) c += ma[i] / nm * DRAG.up[i];
                double ang = std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
                drag_on = ang <= A.drag_mount_deg;
                if (drag_on && A.drag_pos_tau > 0) { A.pos_tau = A.drag_pos_tau; A.pos_tau_nonet = A.drag_pos_tau; }
                if (drag_on)
                    std::fprintf(stderr, "rotor-drag aiding on (start-up gravity %.0f deg from the drone's thrust axis)\n", ang);
            }
        }
        double var = 0;
        for (auto& b : init_buf) {
            double s = 0;
            for (int i = 0; i < 3; i++) s += sq(b.first[i] - mg[i]);
            var += s;
        }
        var /= n;
        if (var < A.still_gyro_var && !bias_known) bg = mg;
        initialised = true;
    }

    // Rotor-drag aiding (see main.py drag_update).
    void drag_update(const V3& f, double dt, bool still) {
        if (!have_f_drag || good_time <= dt) { f_drag = f; have_f_drag = true; return; }
        double kl = std::min(1.0, dt / A.drag_lp);
        for (int i = 0; i < 3; i++) f_drag[i] = f_drag[i] + kl * (f[i] - f_drag[i]);
        V3 fd = f_drag;
        if (A.drag_hp > 0) {
            if (!have_f_drag_slow) { f_drag_slow = fd; have_f_drag_slow = true; }
            double ks = std::min(1.0, dt / A.drag_hp);
            for (int i = 0; i < 3; i++) f_drag_slow[i] = f_drag_slow[i] + ks * (fd[i] - f_drag_slow[i]);
            for (int i = 0; i < 3; i++) fd[i] = fd[i] - f_drag_slow[i];
        }
        if (still) return;
        V3 vb = qrot({q.w, -q.x, -q.y, -q.z}, v);
        double kg = std::min(1.0, dt / A.drag_tau);
        const double* es[2] = {DRAG.e1, DRAG.e2};
        double ks2[2] = {A.drag_k1, A.drag_k2};
        for (int j = 0; j < 2; j++) {
            const double* e = es[j];
            double m = (fd[0] * e[0] + fd[1] * e[1] + fd[2] * e[2]) / ks2[j];
            double d = m - (vb[0] * e[0] + vb[1] * e[1] + vb[2] * e[2]);
            double wr = A.drag_robust > 0 ? 1.0 / (1.0 + sq(d / A.drag_robust)) : 1.0;
            for (int i = 0; i < 3; i++) vb[i] += kg * wr * d * e[i];
        }
        v = qrot(q, vb);
    }

    // Mount guard (see main.py check_mount): far from every training mount -> the gravity-aligned network.
    VelNet fallback;
    void check_mount(const V3& ma) {
        if (!net || !body || A.mount_guard_deg <= 0 || A.fallback_model.empty()) return;
        double n = norm3(ma), best = -1.0;
        for (const auto& m : TRAIN_UP) {
            double nm = std::sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
            best = std::max(best, (ma[0] * m[0] + ma[1] * m[1] + ma[2] * m[2]) / (n * nm));
        }
        double ang = std::acos(std::max(-1.0, std::min(1.0, best))) * 180.0 / M_PI;
        if (ang <= A.mount_guard_deg) return;
        if (std::fabs(n / G - 1.0) > MOUNT_G_TOL) {      // sustained acceleration tilts the mean; not a mount
            std::fprintf(stderr, "mount guard: start-up specific force %.2f m/s^2 is not gravity alone; "
                                 "keeping the body-frame network\n", n);
            return;
        }
        if (!fallback.load(A.fallback_model)) {           // unusable file: keep the body-frame network (as main.py)
            std::fprintf(stderr, "mount guard: cannot load %s; keeping the body-frame network\n", A.fallback_model.c_str());
            return;
        }
        net = &fallback;
        // the coupling the fallback network was validated with (see main.py FALLBACK_COUPLING)
        A.model_pos_gain = 0.0; A.pos_tau = 3.0; A.model_ref = 0.1; A.pos_tau_blind = 0.0;
        W = net->window; dec = net->dec;
        body = net->version >= 3;
        const std::string& f = A.fallback_model;
        size_t sl = f.find_last_of("/\\");
        std::fprintf(stderr, "IMU mount %.0f deg from the training mounts: using %s\n", ang,
                     (sl == std::string::npos ? f : f.substr(sl + 1)).c_str());
    }

    bool zero_velocity(const V3& g, const V3& a) {
        int w = A.zupt_window;
        if (A.zupt_gyro_std > 0) {
            std::vector<double> x = {g[0], g[1], g[2], a[0], a[1], a[2]};
            for (int i = 0; i < 6; i++) { still_s[i] += x[i]; still_ss[i] += x[i] * x[i]; }
            still_w.push_back(std::move(x));
            if ((int)still_w.size() > w) {
                const std::vector<double>& y = still_w.front();
                for (int i = 0; i < 6; i++) { still_s[i] -= y[i]; still_ss[i] -= y[i] * y[i]; }
                still_w.pop_front();
            }
            if ((int)still_w.size() < w) return false;
            double var[6];
            for (int i = 0; i < 6; i++) var[i] = std::max(0.0, still_ss[i] / w - sq(still_s[i] / w));
            double vg = std::max(var[0], std::max(var[1], var[2])), va = std::max(var[3], std::max(var[4], var[5]));
            if (vg < sq(A.zupt_gyro_std) && va < sq(A.zupt_acc_std)) {
                still_gyro = {{still_s[0] / w, still_s[1] / w, still_s[2] / w}};
                return true;
            }
            return false;
        }
        still_g.push_back(norm3(g));
        still_a.push_back(norm3(a) - G);
        if ((int)still_g.size() > w) { still_g.pop_front(); still_a.pop_front(); }
        if ((int)still_g.size() < w) return false;
        double sg = 0, sa = 0;
        for (double x : still_g) sg += x;
        for (double x : still_a) sa += std::fabs(x);
        still_gyro = g;
        return sg / w < A.zupt_gyro && sa / w < A.zupt_acc;
    }

    // Advance to time t; if anything turns non-finite, keep the last finite state (never emit NaN).
    void step(long long t, const V3& g, const V3& a, bool ok) {
        Q q0 = q; V3 p0 = p, v0 = v, bg0 = bg, w0 = w_last, r0 = acc_res;
        step_(t, g, a, ok);
        bool fin = std::isfinite(q.w) && std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z);
        for (int i = 0; i < 3; i++)
            fin = fin && std::isfinite(p[i]) && std::isfinite(v[i]) && std::isfinite(bg[i]) &&
                  std::isfinite(w_last[i]) && std::isfinite(acc_res[i]);
        if (!fin) { q = q0; p = p0; v = {{0, 0, 0}}; bg = bg0; w_last = w0; acc_res = r0; (void)v0; }
    }

    void step_(long long t, const V3& g, const V3& a, bool ok) {
        if (!have_t_prev) { t_prev = t; have_t_prev = true; return; }
        double dt = (double)(t - t_prev) * 1e-9;
        t_prev = t;
        if (dt <= 0) return;
        if (!have_dt_nom || (ok && dt < 2 * dt_nom)) {
            dt_nom = !have_dt_nom ? dt : 0.99 * dt_nom + 0.01 * dt;
            have_dt_nom = true;
        }
        bool gap = dt > 3 * dt_nom;
        double leak;
        if (ok && !gap) {
            bad_time = 0.0;
            yaw_var += A.yaw_q * dt;
            V3 w = {{g[0] - bg[0], g[1] - bg[1], g[2] - bg[2]}};
            V3 w_raw = w;           // before the gravity correction (body-frame network input)
            V3 f = {{a[0] - ba[0], a[1] - ba[1], a[2] - ba[2]}};
            double fn = norm3(f);
            double boost = A.kp_boost > 0 ? A.kp_boost * std::exp(-good_time / A.kp_boost_tau) : 0.0;
            V3 fl = f;
            if (A.tilt_lp > 0) {
                if (!have_f_lp || good_time < dt) { f_lp = f; have_f_lp = true; }
                else {
                    double kl = std::min(1.0, dt / A.tilt_lp);
                    for (int i = 0; i < 3; i++) f_lp[i] = f_lp[i] + kl * (f[i] - f_lp[i]);
                }
                fl = f_lp;
                fn = norm3(fl);
            }
            good_time += dt;
            if (fn > 1e-6 && A.kp + boost > 0) {
                double trust = std::max(0.0, 1.0 - std::fabs(fn - G) / A.tilt_acc_tol);
                if (boost > 0) trust = std::max(std::max(trust, 1.0 - std::fabs(fn - G) / (3.0 * A.tilt_acc_tol)), 0.0);
                if (trust > 0) {
                    V3 up_b = qrot({q.w, -q.x, -q.y, -q.z}, {{0.0, 0.0, 1.0}});
                    V3 fu = {{fl[0] / fn, fl[1] / fn, fl[2] / fn}};
                    V3 e = {{fu[1] * up_b[2] - fu[2] * up_b[1], fu[2] * up_b[0] - fu[0] * up_b[2],
                             fu[0] * up_b[1] - fu[1] * up_b[0]}};
                    double tb = trust;
                    if (A.ki_calm > 0) {
                        double wn = norm3(w);
                        tb = std::max(0.0, 1.0 - std::fabs(fn - G) / (A.tilt_acc_tol / 3.0)) * std::max(0.0, 1.0 - wn / A.ki_calm);
                    }
                    for (int i = 0; i < 3; i++) {
                        w[i] += (A.kp + boost) * trust * e[i];
                        bg[i] -= A.ki * tb * e[i] * dt;
                    }
                }
            }
            Q q_mid = qnorm(qmul(q, qexp(0.5 * w[0] * dt, 0.5 * w[1] * dt, 0.5 * w[2] * dt)));
            V3 acc_w = qrot(q_mid, f);
            acc_w[2] = acc_w[2] - G;
            double ka = A.acc_res_tau > 0 ? std::min(1.0, dt / A.acc_res_tau) : 0.0;
            for (int i = 0; i < 3; i++) {
                acc_res[i] += ka * (acc_w[i] - acc_res[i]);
                acc_w[i] -= acc_res[i];
            }
            q = qnorm(qmul(q, qexp(w[0] * dt, w[1] * dt, w[2] * dt)));
            for (int i = 0; i < 3; i++) {
                p[i] += v[i] * dt + 0.5 * acc_w[i] * dt * dt;
                v[i] += acc_w[i] * dt;
            }
            w_last = w;
            if (net && push_slot(body ? &w_raw : &w, &f)) learned_update();
            bool still = zero_velocity(g, a);
            if (drag_on) drag_update(f, dt, still);
            if (still) {
                double k = std::min(1.0, dt / A.zupt_tau);
                for (int i = 0; i < 3; i++) {
                    v[i] *= 1.0 - k;
                    bg[i] += std::min(1.0, A.bias_rate * dt) * (still_gyro[i] - bg[i]);
                }
            }
            leak = A.leak;
        } else {
            bad_time += dt;
            yaw_var += A.yaw_q_blind * dt;
            if (bad_time > 0.25) good_time = 0.0;
            double dk = std::exp(-dt / A.blind_turn_tau);
            for (int i = 0; i < 3; i++) w_last[i] = w_last[i] * dk;
            const V3& w = w_last;
            q = qnorm(qmul(q, qexp(w[0] * dt, w[1] * dt, w[2] * dt)));
            double vk = std::exp(-dt / (net_ok ? v_blind_tau : A.blind_vel_tau));
            for (int i = 0; i < 3; i++) {
                p[i] += v[i] * dt;
                v[i] *= vk;
            }
            leak = 0.0;
            if (net) {
                long long n = have_dt_nom ? std::max(1LL, (long long)pyround(dt / dt_nom)) : 1;
                long long m = std::min(n, (long long)W * dec);
                for (long long j = 0; j < m; j++) push_slot(nullptr, nullptr);
            }
        }
        if (leak > 0) {
            double k = std::min(1.0, leak * dt);
            for (int i = 0; i < 3; i++) v[i] *= 1.0 - k;
        }
        // Bounded-space prior (see main.py): exponential pull to the start, firmer for the part of the step past 1 s of
        // unusable IMU.
        double tb = A.pos_tau_blind > 0 ? std::min(dt, std::max(0.0, bad_time - 1.0)) : 0.0;
        double tau = (A.pos_tau_nonet > 0 && !net_usable()) ? A.pos_tau_nonet : A.pos_tau;   // see main.py
        double f = 1.0;
        if (tau > 0) f *= std::exp(-(dt - tb) / tau);
        if (tb > 0) f *= std::exp(-tb / A.pos_tau_blind);
        if (f != 1.0)
            for (int i = 0; i < 3; i++) p[i] *= f;
        double sp = norm3(v);
        if (sp > A.v_max) {
            double s = A.v_max / sp;
            for (int i = 0; i < 3; i++) v[i] = v[i] * s;
        }
    }

    bool push_slot(const V3* w, const V3* f) {
        if (!ring_made) {
            if (net && have_dt_nom) {           // the network's sample rate, whatever the input's (see main.py)
                rate_mult = std::max(1, (int)pyround(1.0 / (dt_nom * net->rate)));
                dec = net->dec * rate_mult;
            }
            ring.assign(W, Feat{}); ring_p.assign(W, V3{{0, 0, 0}});
            nb = body ? 9 : 6;
            for (double& b : bin) b = 0.0;
            bin_n = bin_k = 0; slots = 0; ring_made = true;
        }
        if (w) {
            if (body) {             // body frame, gravity kept, plus our "up" seen from the body
                V3 u = qrot({q.w, -q.x, -q.y, -q.z}, {{0.0, 0.0, 1.0}});
                bin[0] += (*w)[0]; bin[1] += (*w)[1]; bin[2] += (*w)[2];
                bin[3] += (*f)[0]; bin[4] += (*f)[1]; bin[5] += (*f)[2];
                bin[6] += u[0]; bin[7] += u[1]; bin[8] += u[2];
            } else {
                V3 gw = qrot(q, *w), aw = qrot(q, *f);
                bin[0] += gw[0]; bin[1] += gw[1]; bin[2] += gw[2];
                bin[3] += aw[0]; bin[4] += aw[1]; bin[5] += aw[2] - G;
            }
            bin_n++; since++;
        }
        bin_k++;
        if (bin_k < dec) return false;
        Feat feat;
        if (bin_n) {
            double n = bin_n;
            feat.has = true;
            for (int i = 0; i < nb; i++) feat.v[i] = bin[i] / n;
            feat.v[nb] = n / dec;
        }
        for (double& b : bin) b = 0.0;
        bin_n = bin_k = 0;
        if (ring[ring_i].has) ring_valid--;
        if (feat.has) ring_valid++;
        p_then = ring_p[ring_i];
        ring[ring_i] = feat;
        ring_p[ring_i] = p;
        ring_i = (ring_i + 1) % W;
        slots++;
        return feat.has;
    }

    bool net_usable() const {               // the network runs at its own rate (input averaged by rate_mult)
        return net && have_dt_nom && std::fabs(dt_nom * net->rate * rate_mult - 1.0) <= 0.15;
    }

    void learned_update() {
        if (!net || since < (long long)A.model_every * rate_mult) return;
        if (slots < W || ring_valid < A.model_min_valid * W) return;
        if (!net_usable()) return;
        since = 0;
        int ni = net->n_in;
        xin.assign((size_t)ni * W, 0.0f);
        for (int j = 0; j < W; j++) {
            const Feat& s = ring[(ring_i + j) % W];
            if (!s.has) continue;
            for (int c = 0; c < ni && c < nb + 1; c++) xin[(size_t)c * W + j] = (float)s.v[c];
        }
        for (int c = 0; c < nb && c < ni - 1; c++)
            for (int j = 0; j < W; j++) xin[(size_t)c * W + j] *= net->scale[c];
        float o[12];
        net->forward(xin, o);
        double d[3], sd[3], vv[3], sv[3];
        for (int i = 0; i < 3; i++) {
            d[i] = o[i]; vv[i] = o[6 + i];
            sd[i] = (double)std::exp(std::min(3.0f, std::max(-7.0f, o[3 + i])));
            sv[i] = (double)std::exp(std::min(3.0f, std::max(-7.0f, o[9 + i])));
        }
        if (body) {                 // body-frame outputs -> our world frame (diagonal covariance rotated)
            rot_cov(q, d, sd);
            rot_cov(q, vv, sv);
        }
        double k = std::exp(-0.5 * yaw_var);
        d[0] *= k; d[1] *= k; vv[0] *= k; vv[1] *= k;
        for (int i = 0; i < 3; i++) {
            double gp = A.model_pos_gain * sq(A.model_ref) / (sq(A.model_ref) + sq(sd[i]));
            double gv = A.model_vel_gain * sq(A.model_vref) / (sq(A.model_vref) + sq(sv[i]));
            p[i] += gp * (p_then[i] + d[i] - p[i]);
            v[i] += gv * (vv[i] - v[i]);
        }
        net_ok = true;
    }
};

// First pass over the start of the file: gyro bias = mean gyro of the first still stretch (main.py prescan_bias).
static bool prescan_bias(const Opt& a, V3& out) {
    PathFinder pf(a, nullptr);
    V3 sum{{0, 0, 0}};
    long long n_still = 0;
    bool have_t0 = false; long long t0 = 0;
    Source src;
    open_source(src, a.input);
    Row r;
    while (src.getline()) {
        if (!parse_line(src.line, r)) continue;
        if (!have_t0) { t0 = r.t; have_t0 = true; }
        if ((double)(r.t - t0) * 1e-9 > a.prescan) break;
        if (pf.healthy(r.g, r.a) && pf.zero_velocity(r.g, r.a)) {
            for (int i = 0; i < 3; i++) sum[i] += pf.still_gyro[i];
            n_still++;
        } else if (n_still >= 100) break;
        else { sum = {{0, 0, 0}}; n_still = 0; }
    }
    close_source(src);
    if (n_still < 100) return false;
    for (int i = 0; i < 3; i++) out[i] = sum[i] / n_still;
    return true;
}

int main(int argc, char** argv) {
    Opt a;
    parse_args(argc, argv, a);
    VelNet net;
    VelNet* np = nullptr;
    if (!a.model.empty()) { if (!net.load(a.model)) return 1; np = &net; }
    std::string note;
    g_layout = detect_layout(a.input, a.columns, a.ts_unit, a.accel_unit, note);
    if (g_layout.accel_first || g_layout.t_scale != 1 || g_layout.a_scale != 1.0 || !note.empty()) {
        const char* unit = g_layout.t_scale == 1000000000LL ? "s" : g_layout.t_scale == 1000000LL ? "ms"
                         : g_layout.t_scale == 1000LL ? "us" : "ns";
        std::fprintf(stderr, "input layout: %s columns first, timestamps in %s, accel in %s%s%s%s\n",
                     g_layout.accel_first ? "accel" : "gyro", unit, g_layout.a_scale != 1.0 ? "g" : "m/s^2",
                     note.empty() ? "" : " (", note.c_str(), note.empty() ? "" : ")");
    }
    PathFinder pf(a, np);
    if (a.prescan > 0 && a.zupt_gyro_std > 0) {
        V3 b;
        if (prescan_bias(a, b)) { pf.bg = b; pf.bias_known = true; }
    }
    Source src;
    open_source(src, a.input);
    FILE* o = std::fopen(a.output.c_str(), "w");
    if (!o) { std::fprintf(stderr, "cannot write %s\n", a.output.c_str()); return 1; }
    static char obuf[1 << 15];
    std::setvbuf(o, obuf, _IOFBF, sizeof obuf);
    std::fputs("# timestamp tx ty tz qx qy qz qw\n", o);
    auto emit = [&](long long t) {
        std::fprintf(o, "%.9f %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n", (double)t * 1e-9, pf.p[0], pf.p[1], pf.p[2],
                     pf.q.x, pf.q.y, pf.q.z, pf.q.w);
    };
    struct P { Row r; bool ok; };
    std::vector<P> pending;
    long long n_out = 0;
    bool have_last = false; long long last_t = 0;
    Row r;
    auto replay = [&](bool set_last) {
        for (auto& x : pending) {
            pf.step(x.r.t, x.r.g, x.r.a, x.ok);
            emit(x.r.t);
            n_out++;
            if (set_last) { last_t = x.r.t; have_last = true; }
        }
        pending.clear();
    };
    while (src.getline()) {
        if (!parse_line(src.line, r)) continue;
        bool ok = pf.healthy(r.g, r.a);
        if (!pf.initialised) {
            pending.push_back({r, ok});
            if (ok) pf.init_buf.push_back({r.g, r.a});
            if ((int)pf.init_buf.size() >= a.init_samples || (int)pending.size() >= 4 * a.init_samples) {
                if (!pf.init_buf.empty()) pf.init_from();
                else pf.initialised = true;
                replay(true);
                pf.init_buf.clear();
                pf.init_buf.shrink_to_fit();
                pending.shrink_to_fit();
            }
            continue;
        }
        pf.step(r.t, r.g, r.a, ok);
        emit(r.t);
        n_out++;
        last_t = r.t; have_last = true;
    }
    if (!pending.empty()) {
        if (!pf.init_buf.empty()) pf.init_from();
        replay(true);
    }
    if (a.extend_blind > 0 && have_last && pf.have_dt_nom && pf.dt_nom != 0) {
        long long step = (long long)pyround(pf.dt_nom * 1e9);
        long long t = last_t;
        long long n = (long long)pyround(a.extend_blind / pf.dt_nom);
        V3 z{{0, 0, 0}};
        for (long long i = 0; i < n; i++) {
            t += step;
            pf.step(t, z, z, false);
            emit(t);
            n_out++;                                // the message counts every pose written
        }
    }
    close_source(src);
    std::fclose(o);
    std::fprintf(stderr, "wrote %lld poses to %s\n", n_out, a.output.c_str());
    return 0;
}
