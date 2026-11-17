/* Hot strapdown step for the Python entry. Same associations as PathFinder._step in main.py
 * (double precision, UCRT libm, which is what CPython's math module calls on Windows).
 * The learned-velocity correction stays in Python; this file does not own the network.
 *
 * Build (x64 Native Tools, or the INCLUDE/LIB setup used from this repo):
 *   cl /O2 /LD /Fe:hotstep.dll hotstep.c
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

API int hot_abi(void) { return 1; }

#define G0 9.80665

typedef struct Hot {
    double q[4], p[3], v[3], bg[3], ba[3], w_last[3], acc_res[3], f_lp[3];
    int has_flp;
    long long t_prev;
    int has_t;
    double bad_time, good_time, yaw_var, dt_nom, dt;
    int has_dt;
    int since, slots, ring_i, ring_valid, bin_n, bin_k, nb, body, dec, W, has_net;
    int net_ok, skip_rest, healthy_branch, want_learned;
    double leak_now;
    double bin[9];
    double p_then[3];
    double still_gyro[3];
    unsigned char *slot_set;
    double *ring_p;                 /* W * 3 */
    float *ring_f;                  /* (nb+1) * W, row-major, may be NULL */
    /* zero-velocity, std branch */
    int zw, zn, zi;
    double *zring;                  /* zw * 6 */
    double still_s[6], still_ss[6];
    /* magnitude branch */
    double *mg, *ma;
    int mn, mi;
    /* parameters */
    double kp, ki, kp_boost, kp_boost_tau, tilt_lp, ki_calm, tilt_acc_tol;
    double acc_res_tau, zupt_tau, bias_rate, leak, pos_tau, v_max;
    double blind_turn_tau, blind_vel_tau, v_blind_tau;
    double yaw_q, yaw_q_blind, zupt_gyro_std, zupt_acc_std, zupt_gyro, zupt_acc;
    int model_every;
    double model_min_valid, net_rate;
} Hot;

static double nrm(double x, double y, double z) {
    return sqrt(x * x + y * y + z * z);
}

static void qexp(double w0, double w1, double w2, double *o) {
    double th = sqrt(w0 * w0 + w1 * w1 + w2 * w2);
    if (th < 1e-12) {
        o[0] = 1.0;
        o[1] = 0.5 * w0;
        o[2] = 0.5 * w1;
        o[3] = 0.5 * w2;
        return;
    }
    double s = sin(0.5 * th) / th;
    o[0] = cos(0.5 * th);
    o[1] = w0 * s;
    o[2] = w1 * s;
    o[3] = w2 * s;
}

static void qmul(const double *a, const double *b, double *o) {
    double aw = a[0], ax = a[1], ay = a[2], az = a[3];
    double bw = b[0], bx = b[1], by = b[2], bz = b[3];
    o[0] = aw * bw - ax * bx - ay * by - az * bz;
    o[1] = aw * bx + ax * bw + ay * bz - az * by;
    o[2] = aw * by - ax * bz + ay * bw + az * bx;
    o[3] = aw * bz + ax * by - ay * bx + az * bw;
}

static void qnorm(const double *q, double *o) {
    double n = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n == 0.0)
        n = 1.0;
    o[0] = q[0] / n;
    o[1] = q[1] / n;
    o[2] = q[2] / n;
    o[3] = q[3] / n;
}

static void qrot(const double *q, double vx, double vy, double vz, double *o) {
    double w = q[0], x = q[1], y = q[2], z = q[3];
    double tx = 2 * (y * vz - z * vy);
    double ty = 2 * (z * vx - x * vz);
    double tz = 2 * (x * vy - y * vx);
    o[0] = vx + w * tx + y * tz - z * ty;
    o[1] = vy + w * ty + z * tx - x * tz;
    o[2] = vz + w * tz + x * ty - y * tx;
}

/* params: 23 doubles, then ints passed separately. See hot_create in main.py. */
API Hot *hot_create(const double *prm, int body, int dec, int W, int nb,
                                       int has_net, int zupt_window, float *ring_f, int model_every) {
    if (zupt_window < 0 || W < 0 || (has_net && (W == 0 || (nb != 6 && nb != 9) || !ring_f)))
        return NULL;
    Hot *h = (Hot *)calloc(1, sizeof(Hot));
    if (!h)
        return NULL;
    h->kp = prm[0];
    h->ki = prm[1];
    h->kp_boost = prm[2];
    h->kp_boost_tau = prm[3];
    h->tilt_lp = prm[4];
    h->ki_calm = prm[5];
    h->tilt_acc_tol = prm[6];
    h->acc_res_tau = prm[7];
    h->zupt_tau = prm[8];
    h->bias_rate = prm[9];
    h->leak = prm[10];
    h->pos_tau = prm[11];
    h->v_max = prm[12];
    h->blind_turn_tau = prm[13];
    h->blind_vel_tau = prm[14];
    h->v_blind_tau = prm[15];
    h->yaw_q = prm[16];
    h->yaw_q_blind = prm[17];
    h->zupt_gyro_std = prm[18];
    h->zupt_acc_std = prm[19];
    h->zupt_gyro = prm[20];
    h->zupt_acc = prm[21];
    h->model_min_valid = prm[22];
    h->net_rate = prm[23];
    h->model_every = model_every > 0 ? model_every : 1;
    h->body = body;
    h->dec = dec > 0 ? dec : 1;
    h->W = W;
    h->nb = nb;
    h->has_net = has_net;
    h->good_time = 1e9;
    h->q[0] = 1.0;
    h->zw = zupt_window;
    if (zupt_window > 0) {
        h->zring = (double *)calloc((size_t)zupt_window * 6, sizeof(double));
        h->mg = (double *)calloc((size_t)zupt_window, sizeof(double));
        h->ma = (double *)calloc((size_t)zupt_window, sizeof(double));
    }
    if (has_net && W > 0) {
        h->slot_set = (unsigned char *)calloc((size_t)W, 1);
        h->ring_p = (double *)calloc((size_t)W * 3, sizeof(double));
        h->ring_f = ring_f;
    }
    if ((zupt_window > 0 && (!h->zring || !h->mg || !h->ma)) ||
        (has_net && (!h->slot_set || !h->ring_p))) {
        free(h->zring); free(h->mg); free(h->ma); free(h->slot_set); free(h->ring_p); free(h);
        return NULL;
    }
    return h;
}

API void hot_free(Hot *h) {
    if (!h)
        return;
    free(h->zring);
    free(h->mg);
    free(h->ma);
    free(h->slot_set);
    free(h->ring_p);
    free(h);
}

/* Fields the Python side may have changed since the previous sample. */
API void hot_pull(Hot *h, const double *q, const double *p, const double *v,
                                    const double *bg, const double *ba, const double *w,
                                    const double *ar, int since, int net_ok) {
    memcpy(h->q, q, 4 * sizeof(double));
    memcpy(h->p, p, 3 * sizeof(double));
    memcpy(h->v, v, 3 * sizeof(double));
    memcpy(h->bg, bg, 3 * sizeof(double));
    memcpy(h->ba, ba, 3 * sizeof(double));
    memcpy(h->w_last, w, 3 * sizeof(double));
    memcpy(h->acc_res, ar, 3 * sizeof(double));
    h->since = since;
    h->net_ok = net_ok;
}

API void hot_seed(Hot *h, double bad, double good, double yaw, int has_t, long long t_prev) {
    h->bad_time = bad;
    h->good_time = good;
    h->yaw_var = yaw;
    h->has_t = has_t;
    h->t_prev = t_prev;
}

static int zupt_std(Hot *h, double x0, double x1, double x2, double x3, double x4, double x5) {
    int w = h->zw;
    if (w <= 0)
        return 0;
    double *s = h->still_s, *ss = h->still_ss;
    int zi = h->zi;
    int base = zi * 6;
    if (h->zn == w) {
        double y0 = h->zring[base], y1 = h->zring[base + 1], y2 = h->zring[base + 2];
        double y3 = h->zring[base + 3], y4 = h->zring[base + 4], y5 = h->zring[base + 5];
        s[0] -= y0; ss[0] -= y0 * y0;
        s[1] -= y1; ss[1] -= y1 * y1;
        s[2] -= y2; ss[2] -= y2 * y2;
        s[3] -= y3; ss[3] -= y3 * y3;
        s[4] -= y4; ss[4] -= y4 * y4;
        s[5] -= y5; ss[5] -= y5 * y5;
    } else {
        h->zn += 1;
    }
    h->zring[base] = x0; h->zring[base + 1] = x1; h->zring[base + 2] = x2;
    h->zring[base + 3] = x3; h->zring[base + 4] = x4; h->zring[base + 5] = x5;
    s[0] += x0; ss[0] += x0 * x0;
    s[1] += x1; ss[1] += x1 * x1;
    s[2] += x2; ss[2] += x2 * x2;
    s[3] += x3; ss[3] += x3 * x3;
    s[4] += x4; ss[4] += x4 * x4;
    s[5] += x5; ss[5] += x5 * x5;
    h->zi = (zi + 1 < w) ? zi + 1 : 0;
    if (h->zn < w)
        return 0;
    double vg = 0.0, va = 0.0;
    for (int i = 0; i < 3; i++) {
        double var = ss[i] / w - (s[i] / w) * (s[i] / w);
        if (var > vg)
            vg = var;
    }
    for (int i = 3; i < 6; i++) {
        double var = ss[i] / w - (s[i] / w) * (s[i] / w);
        if (var > va)
            va = var;
    }
    if (vg < 0.0)
        vg = 0.0;
    if (va < 0.0)
        va = 0.0;
    if (vg < h->zupt_gyro_std * h->zupt_gyro_std && va < h->zupt_acc_std * h->zupt_acc_std) {
        h->still_gyro[0] = s[0] / w;
        h->still_gyro[1] = s[1] / w;
        h->still_gyro[2] = s[2] / w;
        return 1;
    }
    return 0;
}

static int zupt_mag(Hot *h, double gx, double gy, double gz, double ax, double ay, double az) {
    int w = h->zw;
    if (w <= 0)
        return 0;
    double ng = nrm(gx, gy, gz);
    double na = nrm(ax, ay, az) - G0;
    if (h->mn == w) {
        /* drop oldest: we overwrite at mi and the sum is recomputed if the window just slides.
         * Match the Python lists: append, then pop(0) once over the window. Running sum. */
    }
    if (h->mn < w) {
        h->mg[h->mn] = ng;
        h->ma[h->mn] = na;
        h->mn += 1;
        h->mi = 0;
        if (h->mn < w)
            return 0;
    } else {
        h->mg[h->mi] = ng;
        h->ma[h->mi] = na;
        h->mi = (h->mi + 1) % w;
    }
    double sg = 0.0, sa = 0.0;
    for (int i = 0; i < w; i++) {
        sg += h->mg[i];
        sa += fabs(h->ma[i]);
    }
    h->still_gyro[0] = gx;
    h->still_gyro[1] = gy;
    h->still_gyro[2] = gz;
    return (sg / w < h->zupt_gyro) && (sa / w < h->zupt_acc);
}

static void push_one(Hot *h, int have, double w0, double w1, double w2, double f0, double f1, double f2) {
    if (!h->has_net || h->W <= 0)
        return;
    if (have) {
        double *b = h->bin;
        if (h->body) {
            double qi[4] = {h->q[0], -h->q[1], -h->q[2], -h->q[3]};
            double u[3];
            qrot(qi, 0.0, 0.0, 1.0, u);
            b[0] += w0; b[1] += w1; b[2] += w2;
            b[3] += f0; b[4] += f1; b[5] += f2;
            b[6] += u[0]; b[7] += u[1]; b[8] += u[2];
        } else {
            double gw[3], aw[3];
            qrot(h->q, w0, w1, w2, gw);
            qrot(h->q, f0, f1, f2, aw);
            b[0] += gw[0]; b[1] += gw[1]; b[2] += gw[2];
            b[3] += aw[0]; b[4] += aw[1]; b[5] += aw[2] - G0;
        }
        h->bin_n += 1;
        h->since += 1;
    }
    h->bin_k += 1;
    if (h->bin_k < h->dec) {
        h->want_learned = 0;
        return;
    }
    int slot = h->ring_i;
    int feat = 0;
    if (h->bin_n) {
        double n = (double)h->bin_n;
        double health = n / (double)h->dec;
        if (h->ring_f) {
            int W = h->W;
            for (int i = 0; i < h->nb; i++)
                h->ring_f[i * W + slot] = (float)(h->bin[i] / n);
            h->ring_f[h->nb * W + slot] = (float)health;
        }
        for (int i = 0; i < h->nb; i++)
            h->bin[i] = 0.0;
        feat = 1;
    } else if (h->ring_f) {
        int W = h->W;
        for (int i = 0; i < h->nb + 1; i++)
            h->ring_f[i * W + slot] = 0.0f;
    }
    h->bin_n = 0;
    h->bin_k = 0;
    if (h->slot_set[slot])
        h->ring_valid -= 1;
    if (feat)
        h->ring_valid += 1;
    h->slot_set[slot] = (unsigned char)feat;
    h->p_then[0] = h->ring_p[slot * 3];
    h->p_then[1] = h->ring_p[slot * 3 + 1];
    h->p_then[2] = h->ring_p[slot * 3 + 2];
    h->ring_p[slot * 3] = h->p[0];
    h->ring_p[slot * 3 + 1] = h->p[1];
    h->ring_p[slot * 3 + 2] = h->p[2];
    h->ring_i = (slot + 1) % h->W;
    h->slots += 1;
    /* Python's learned_update returns immediately unless every guard passes. Stopping only then
     * keeps the same updates without a round trip on the other 19 samples out of 20. */
    h->want_learned = 0;
    if (feat && h->since >= h->model_every && h->slots >= h->W
        && h->ring_valid >= h->model_min_valid * (double)h->W
        && h->has_dt && h->net_rate > 0.0
        && fabs(h->dt_nom * h->net_rate - 1.0) <= 0.15)
        h->want_learned = 1;
}

/* Returns 1 when the Python learned update should run (a completed healthy slot). */
API int hot_integrate(Hot *h, long long t, double g0, double g1, double g2,
                                        double a0, double a1, double a2, int ok) {
    h->want_learned = 0;
    h->skip_rest = 0;
    h->healthy_branch = 0;
    h->leak_now = 0.0;
    if (!h->has_t) {
        h->t_prev = t;
        h->has_t = 1;
        h->skip_rest = 1;
        return 0;
    }
    long long idt = t - h->t_prev;
    h->t_prev = t;
    double dt = (double)idt * 1e-9;
    h->dt = dt;
    if (dt <= 0.0) {
        h->skip_rest = 1;
        return 0;
    }
    if (!h->has_dt || (ok && dt < 2.0 * h->dt_nom)) {
        h->dt_nom = !h->has_dt ? dt : 0.99 * h->dt_nom + 0.01 * dt;
        h->has_dt = 1;
    }
    int gap = dt > 3.0 * h->dt_nom;
    if (ok && !gap) {
        h->healthy_branch = 1;
        h->bad_time = 0.0;
        h->yaw_var += h->yaw_q * dt;
        double w0 = g0 - h->bg[0], w1 = g1 - h->bg[1], w2 = g2 - h->bg[2];
        double wr0 = w0, wr1 = w1, wr2 = w2;
        double f0 = a0 - h->ba[0], f1 = a1 - h->ba[1], f2 = a2 - h->ba[2];
        double fn = nrm(f0, f1, f2);
        double boost = h->kp_boost > 0.0 ? h->kp_boost * exp(-h->good_time / h->kp_boost_tau) : 0.0;
        double fl0 = f0, fl1 = f1, fl2 = f2;
        if (h->tilt_lp > 0.0) {
            if (!h->has_flp || h->good_time < dt) {
                h->f_lp[0] = f0; h->f_lp[1] = f1; h->f_lp[2] = f2;
                h->has_flp = 1;
            } else {
                double kl = dt / h->tilt_lp;
                if (kl > 1.0)
                    kl = 1.0;
                h->f_lp[0] = h->f_lp[0] + kl * (f0 - h->f_lp[0]);
                h->f_lp[1] = h->f_lp[1] + kl * (f1 - h->f_lp[1]);
                h->f_lp[2] = h->f_lp[2] + kl * (f2 - h->f_lp[2]);
            }
            fl0 = h->f_lp[0]; fl1 = h->f_lp[1]; fl2 = h->f_lp[2];
            fn = nrm(fl0, fl1, fl2);
        }
        h->good_time += dt;
        if (fn > 1e-6 && h->kp + boost > 0.0) {
            double trust = 1.0 - fabs(fn - G0) / h->tilt_acc_tol;
            if (trust < 0.0)
                trust = 0.0;
            if (boost > 0.0) {
                double t2 = 1.0 - fabs(fn - G0) / (3.0 * h->tilt_acc_tol);
                if (t2 < 0.0)
                    t2 = 0.0;
                if (t2 > trust)
                    trust = t2;
            }
            if (trust > 0.0) {
                double qi[4] = {h->q[0], -h->q[1], -h->q[2], -h->q[3]};
                double up[3];
                qrot(qi, 0.0, 0.0, 1.0, up);
                double fu0 = fl0 / fn, fu1 = fl1 / fn, fu2 = fl2 / fn;
                double e0 = fu1 * up[2] - fu2 * up[1];
                double e1 = fu2 * up[0] - fu0 * up[2];
                double e2 = fu0 * up[1] - fu1 * up[0];
                double tb = trust;
                if (h->ki_calm > 0.0) {
                    double wn = nrm(w0, w1, w2);
                    double tb1 = 1.0 - fabs(fn - G0) / (h->tilt_acc_tol / 3.0);
                    if (tb1 < 0.0)
                        tb1 = 0.0;
                    double tb2 = 1.0 - wn / h->ki_calm;
                    if (tb2 < 0.0)
                        tb2 = 0.0;
                    tb = tb1 * tb2;
                }
                double kpb = (h->kp + boost) * trust;
                w0 += kpb * e0; w1 += kpb * e1; w2 += kpb * e2;
                h->bg[0] -= h->ki * tb * e0 * dt;
                h->bg[1] -= h->ki * tb * e1 * dt;
                h->bg[2] -= h->ki * tb * e2 * dt;
            }
        }
        double qe[4], qm[4], qn[4];
        qexp(0.5 * w0 * dt, 0.5 * w1 * dt, 0.5 * w2 * dt, qe);
        qmul(h->q, qe, qm);
        qnorm(qm, qn);
        double acc[3];
        qrot(qn, f0, f1, f2, acc);
        acc[2] -= G0;
        double ka = 0.0;
        if (h->acc_res_tau > 0.0) {
            ka = dt / h->acc_res_tau;
            if (ka > 1.0)
                ka = 1.0;
        }
        for (int i = 0; i < 3; i++) {
            h->acc_res[i] += ka * (acc[i] - h->acc_res[i]);
            acc[i] -= h->acc_res[i];
        }
        qexp(w0 * dt, w1 * dt, w2 * dt, qe);
        qmul(h->q, qe, qm);
        qnorm(qm, qn);
        h->q[0] = qn[0]; h->q[1] = qn[1]; h->q[2] = qn[2]; h->q[3] = qn[3];
        for (int i = 0; i < 3; i++) {
            h->p[i] += h->v[i] * dt + 0.5 * acc[i] * dt * dt;
            h->v[i] += acc[i] * dt;
        }
        h->w_last[0] = w0; h->w_last[1] = w1; h->w_last[2] = w2;
        if (h->has_net)
            push_one(h, 1, h->body ? wr0 : w0, h->body ? wr1 : w1, h->body ? wr2 : w2, f0, f1, f2);
        h->leak_now = h->leak;
        return h->want_learned;
    }
    /* Dead reckoning. */
    h->bad_time += dt;
    h->yaw_var += h->yaw_q_blind * dt;
    if (h->bad_time > 0.25)
        h->good_time = 0.0;
    double dk = exp(-dt / h->blind_turn_tau);
    h->w_last[0] *= dk; h->w_last[1] *= dk; h->w_last[2] *= dk;
    double qe[4], qm[4], qn[4];
    qexp(h->w_last[0] * dt, h->w_last[1] * dt, h->w_last[2] * dt, qe);
    qmul(h->q, qe, qm);
    qnorm(qm, qn);
    h->q[0] = qn[0]; h->q[1] = qn[1]; h->q[2] = qn[2]; h->q[3] = qn[3];
    double tau = h->net_ok ? h->v_blind_tau : h->blind_vel_tau;
    double vk = exp(-dt / tau);
    for (int i = 0; i < 3; i++) {
        h->p[i] += h->v[i] * dt;
        h->v[i] *= vk;
    }
    h->leak_now = 0.0;
    if (h->has_net) {
        int nrep;
        if (h->has_dt) {
            double r = dt / h->dt_nom;
            double fl = floor(r);
            double frac = r - fl;
            long long ni = (long long)fl;
            if (frac > 0.5)
                ni += 1;
            else if (frac == 0.5 && (ni & 1LL))
                ni += 1;
            if (ni < 1)
                ni = 1;
            nrep = (int)ni;
        } else {
            nrep = 1;
        }
        int cap = h->W * h->dec;
        if (nrep > cap)
            nrep = cap;
        for (int j = 0; j < nrep; j++)
            push_one(h, 0, 0, 0, 0, 0, 0, 0);
    }
    return 0;
}

/* Zero-velocity update for the healthy branch, then leak, position pull and speed limit.
 * g/a are the raw sample (the std detector uses them, not the bias-corrected values). */
API void hot_rest(Hot *h, double g0, double g1, double g2, double a0, double a1, double a2) {
    if (h->skip_rest)
        return;
    double dt = h->dt;
    if (h->healthy_branch) {
        int still = 0;
        if (h->zupt_gyro_std > 0.0)
            still = zupt_std(h, g0, g1, g2, a0, a1, a2);
        else
            still = zupt_mag(h, g0, g1, g2, a0, a1, a2);
        if (still) {
            double k = dt / h->zupt_tau;
            if (k > 1.0)
                k = 1.0;
            double br = h->bias_rate * dt;
            if (br > 1.0)
                br = 1.0;
            double one_k = 1.0 - k;
            for (int i = 0; i < 3; i++) {
                h->v[i] *= one_k;
                h->bg[i] += br * (h->still_gyro[i] - h->bg[i]);
            }
        }
    }
    if (h->leak_now > 0.0) {
        double k = h->leak_now * dt;
        if (k > 1.0)
            k = 1.0;
        double s = 1.0 - k;
        h->v[0] *= s; h->v[1] *= s; h->v[2] *= s;
    }
    if (h->pos_tau > 0.0) {
        double k = dt / h->pos_tau;
        if (k > 1.0)
            k = 1.0;
        double s = 1.0 - k;
        h->p[0] *= s; h->p[1] *= s; h->p[2] *= s;
    }
    double sp = nrm(h->v[0], h->v[1], h->v[2]);
    if (sp > h->v_max) {
        double s = h->v_max / sp;
        h->v[0] *= s; h->v[1] *= s; h->v[2] *= s;
    }
}

/* Pack the fields Python reads. Layout in main.py (_HOT_OUT). */
API void hot_read(const Hot *h, double *o, long long *t_prev, int *has_t, int *has_dt,
                                    int *since, int *slots, int *ring_i, int *ring_valid, int *net_ok) {
    o[0] = h->q[0]; o[1] = h->q[1]; o[2] = h->q[2]; o[3] = h->q[3];
    o[4] = h->p[0]; o[5] = h->p[1]; o[6] = h->p[2];
    o[7] = h->v[0]; o[8] = h->v[1]; o[9] = h->v[2];
    o[10] = h->bg[0]; o[11] = h->bg[1]; o[12] = h->bg[2];
    o[13] = h->w_last[0]; o[14] = h->w_last[1]; o[15] = h->w_last[2];
    o[16] = h->acc_res[0]; o[17] = h->acc_res[1]; o[18] = h->acc_res[2];
    o[19] = h->p_then[0]; o[20] = h->p_then[1]; o[21] = h->p_then[2];
    o[22] = h->yaw_var;
    o[23] = h->dt_nom;
    o[24] = h->good_time;
    o[25] = h->bad_time;
    *t_prev = h->t_prev;
    *has_t = h->has_t;
    *has_dt = h->has_dt;
    *since = h->since;
    *slots = h->slots;
    *ring_i = h->ring_i;
    *ring_valid = h->ring_valid;
    *net_ok = h->net_ok;
}

API void hot_push_learned(Hot *h, double p0, double p1, double p2,
                                            double v0, double v1, double v2, int since, int net_ok,
                                            double yaw) {
    h->p[0] = p0; h->p[1] = p1; h->p[2] = p2;
    h->v[0] = v0; h->v[1] = v1; h->v[2] = v2;
    h->since = since;
    h->net_ok = net_ok;
    h->yaw_var = yaw;
}

static void write_pose(const Hot *h, double *o) {
    /* TUM order: tx ty tz qx qy qz qw. q is stored (w, x, y, z). */
    o[0] = h->p[0];
    o[1] = h->p[1];
    o[2] = h->p[2];
    o[3] = h->q[1];
    o[4] = h->q[2];
    o[5] = h->q[3];
    o[6] = h->q[0];
}

/* Walk samples [begin, end). Writes one pose per sample that finished hot_rest, packed at poses[0].
 * Returns how many samples were consumed. *hold is 1 when the last consumed sample is waiting for the
 * Python learned update and hot_rest (no pose written for it yet). */
API int hot_batch(Hot *h, const long long *ts, const double *imu, const int *ok,
                                    int begin, int end, double *poses, int *hold) {
    int finished = 0;
    *hold = 0;
    for (int i = begin; i < end; i++) {
        const double *s = imu + (size_t)i * 6;
        int want = hot_integrate(h, ts[i], s[0], s[1], s[2], s[3], s[4], s[5], ok[i]);
        if (want) {
            *hold = 1;
            return i - begin + 1;
        }
        hot_rest(h, s[0], s[1], s[2], s[3], s[4], s[5]);
        write_pose(h, poses + (size_t)finished * 7);
        finished++;
    }
    return end - begin;
}
