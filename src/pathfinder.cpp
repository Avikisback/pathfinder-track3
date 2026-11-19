// Path Finder: Samsung XR Hackathon 2026, Track 3. IMU-only 6-DoF trajectory plus a 5 s blind forecast.
//
// The whole file is known before any pose is written, so the sensor errors are solved for, not filtered:
//   1. Attitude: gyro integration with a bias curve (linear spline, 8 s knots) fitted against gravity.
//   2. Self-calibration: one least-squares solve finds the bias curves (gyro and accel), a tilt per stretch between gaps
//      and a velocity jump per gap, so that the 5 s average velocity and the position stay small. On drones the velocity
//      is pulled toward what rotor drag says instead of zero. The drift stiffness of the bias curves is chosen per file
//      by Bayesian evidence (random-walk prior, MacKay / Fellner-Schall updates).
//   3. Post-filter: a self-check on the 5 s average speed, then a per-octave Wiener gain (drones) or fixed band gains
//      (handheld); if the check fails, a plain high-pass keeps only fast motion.
//   4. Blind 5 s: hold plus a damped share of the last velocity. Handheld files fit that share on their own path
//      (near zero in a room, larger on a steady walk); drones use 0.5 and descend 0.25 m over 3 s if the rotors
//      are still on.
// No network, no learned weights: a handful of platform constants in constants.h, fitted on training data only.
//
// This is a port of experiments/scripts/ref_entry.py; results match it to about a millimetre (tools/compare_cpp.py).
// Usage: pathfinder <imu.csv | dataset_dir> <out.txt> [--debug]

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "constants.h"

// Build-time settings. The defaults are the entry; override with -D only to test a variant
// (experiments/scripts/score_bin.py compares builds). Tested and not adopted, 15 Nov: PF_GYRO_AMP=111111 (anchor
// 0.003 rad/s) and PF_GAP_BRIDGE 1 or 2.
#ifndef PF_GYRO_AMP
#define PF_GYRO_AMP 100.0  // 1 / (0.1 rad/s)^2: how tightly the gyro-bias curve is held to the starting calibration
#endif
#ifndef PF_GAP_BRIDGE
#define PF_GAP_BRIDGE 0  // turn rate inside a dropout: 0 = straight line between both sides, 1 = zero, 2 = each side fades
#endif
#ifndef PF_GAP_TAU
#define PF_GAP_TAU 0.35  // s, fade time for PF_GAP_BRIDGE 2
#endif
#ifndef PF_BRIDGE_HAND
#define PF_BRIDGE_HAND 0.08  // s: handheld, dead runs up to this long are filled by interpolation; longer ones are dropouts
#endif
#ifndef PF_BRIDGE_DRONE
#define PF_BRIDGE_DRONE 0.25  // s: the same for drones, whose turn rate is smooth enough to interpolate further
#endif
#ifndef PF_CLEAN_ROWS
#define PF_CLEAN_ROWS 1  // 1 = remove spikes and fill short holes before solving (clean_rows)
#endif
#ifndef PF_RATE_LIMIT
#define PF_RATE_LIMIT 0.3  // m/s^2: evidence-chosen accel drift rate above which the solve is distrusted (1e9 = off)
#endif
#ifndef PF_FORCE_CHECK
#define PF_FORCE_CHECK 0  // self-check: 0 = as designed, 1 = always fall back to the high-pass, 2 = never fall back
#endif
#ifndef PF_TAIL_DISTRUST_HOLD
#define PF_TAIL_DISTRUST_HOLD 1  // 1 = when the self-check falls back, the blind tail holds (plus the drone descent)
#endif
#ifndef PF_TAIL_DRONE
#define PF_TAIL_DRONE 1  // drone blind tail (before the descent): 0 = hold, 1 = fixed gain 0.5, 2 = fitted like handheld
#endif
#ifndef PF_TAIL_HAND
#define PF_TAIL_HAND 2  // handheld blind tail: 0 = hold, 1 = old fixed gain 0.5, 2 = gain fitted on the file itself
#endif

using std::vector;
typedef std::pair<int, int> Run;

static bool g_debug = false;
#define DBG(...) \
    do {                                         \
        if (g_debug) std::fprintf(stderr, __VA_ARGS__); \
    } while (0)

// ---------------------------------------------------------------- small maths
struct V3 {
    double x, y, z;
};
static inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static inline V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
static inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline double norm(V3 a) { return std::sqrt(dot(a, a)); }
static inline double comp(V3 a, int i) { return i == 0 ? a.x : (i == 1 ? a.y : a.z); }
static inline V3 unit(int i) { return {i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0}; }

struct Qt {
    double w, x, y, z;
};
static inline Qt qmul(Qt a, Qt b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
static inline Qt qnormalize(Qt q) {
    double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    return {q.w / n, q.x / n, q.y / n, q.z / n};
}
static inline Qt qconj(Qt q) { return {q.w, -q.x, -q.y, -q.z}; }
// Rotation vector to quaternion, as the Python _qexp (used for gyro integration).
static inline Qt qexp(V3 r) {
    double a = norm(r);
    if (a < 1e-12) return qnormalize({1.0, 0.5 * r.x, 0.5 * r.y, 0.5 * r.z});
    double s = std::sin(a * 0.5) / a;
    return {std::cos(a * 0.5), r.x * s, r.y * s, r.z * s};
}
// Rotation vector to quaternion, as scipy's from_rotvec.
static inline Qt q_from_rotvec(V3 r) {
    double a = norm(r);
    double s = a <= 1e-3 ? 0.5 - a * a / 48.0 + a * a * a * a / 3840.0 : std::sin(a / 2) / a;
    return {std::cos(a / 2), r.x * s, r.y * s, r.z * s};
}
// Quaternion to rotation vector, as scipy's as_rotvec.
static inline V3 rotvec_from_q(Qt q) {
    q = qnormalize(q);
    if (q.w < 0) q = {-q.w, -q.x, -q.y, -q.z};
    double vn = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
    double a = 2 * std::atan2(vn, q.w);
    double s = a <= 1e-3 ? 2 + a * a / 12 + 7 * a * a * a * a / 2880 : a / std::sin(a / 2);
    return {q.x * s, q.y * s, q.z * s};
}
// Composition as scipy's Rotation __mul__ (normalised Hamilton product).
static inline Qt qcompose(Qt a, Qt b) { return qnormalize(qmul(a, b)); }

struct M3 {
    double m[3][3];
};
static inline M3 qmat(Qt q) {  // scipy as_matrix (normalises first)
    q = qnormalize(q);
    double x = q.x, y = q.y, z = q.z, w = q.w;
    double x2 = x * x, y2 = y * y, z2 = z * z, w2 = w * w;
    double xy = x * y, zw = z * w, xz = x * z, yw = y * w, yz = y * z, xw = x * w;
    M3 r;
    r.m[0][0] = x2 - y2 - z2 + w2;
    r.m[1][0] = 2 * (xy + zw);
    r.m[2][0] = 2 * (xz - yw);
    r.m[0][1] = 2 * (xy - zw);
    r.m[1][1] = -x2 + y2 - z2 + w2;
    r.m[2][1] = 2 * (yz + xw);
    r.m[0][2] = 2 * (xz + yw);
    r.m[1][2] = 2 * (yz - xw);
    r.m[2][2] = -x2 - y2 + z2 + w2;
    return r;
}
static inline V3 mv(const M3& a, V3 v) {
    return {a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z, a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
            a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z};
}
static inline M3 mm(const M3& a, const M3& b) {
    M3 r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
    return r;
}
static inline V3 col(const M3& a, int j) { return {a.m[0][j], a.m[1][j], a.m[2][j]}; }
static const M3 kI3 = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};

// numpy-compatible helpers
static double median_of(vector<double>& a) {  // np.median; reorders a
    size_t n = a.size();
    if (!n) return 0.0;
    size_t h = n / 2;
    std::nth_element(a.begin(), a.begin() + h, a.end());
    double hi = a[h];
    if (n & 1) return hi;
    double lo = *std::max_element(a.begin(), a.begin() + h);
    return (lo + hi) / 2.0;
}
static inline int py_round(double x) { return (int)std::nearbyint(x); }  // Python round(): half to even
static void linspace(double a, double b, int n, vector<double>& out) {
    out.resize(n);
    double step = n > 1 ? (b - a) / (n - 1) : 0.0;
    for (int i = 0; i < n; i++) out[i] = a + i * step;
    if (n > 1) out[n - 1] = b;
}
// Linear spline basis as attitude.spline_basis: the two active knots and the weight of the upper one.
static inline void basis(const vector<double>& kn, double tt, int& j, double& w) {
    int K = (int)kn.size();
    int i = (int)(std::lower_bound(kn.begin(), kn.end(), tt) - kn.begin());
    j = std::min(std::max(i - 1, 0), K - 2);
    w = (tt - kn[j]) / (kn[j + 1] - kn[j]);
    w = std::min(std::max(w, 0.0), 1.0);
}
static inline V3 spline_eval(const vector<double>& kn, const vector<V3>& c, double tt) {
    int j;
    double w;
    basis(kn, tt, j, w);
    return c[j] * (1 - w) + c[j + 1] * w;
}
// scipy.ndimage.uniform_filter1d(x, size, mode="nearest") on a strided array of n values
static void box_filter(const double* x, int n, int stride, int size, double* out, int ostride) {
    int L = size / 2, R = size - L - 1;
    double s = 0;
    for (int j = -L; j <= R; j++) s += x[(size_t)std::min(std::max(j, 0), n - 1) * stride];
    for (int i = 0; i < n; i++) {
        out[(size_t)i * ostride] = s / size;
        int add = std::min(i + R + 1, n - 1), rem = std::max(i - L, 0);
        s += x[(size_t)add * stride] - x[(size_t)rem * stride];
    }
}

// ---------------------------------------------------------------- dense symmetric solves
// Cholesky in place (lower triangle). Returns false if not positive definite.
static bool cholesky(double* A, int n) {
    for (int j = 0; j < n; j++) {
        double* Aj = A + (size_t)j * n;
        double s = Aj[j];
        for (int k = 0; k < j; k++) s -= Aj[k] * Aj[k];
        if (!(s > 0)) return false;
        double d = std::sqrt(s);
        Aj[j] = d;
        for (int i = j + 1; i < n; i++) {
            double* Ai = A + (size_t)i * n;
            double t = Ai[j];
            for (int k = 0; k < j; k++) t -= Ai[k] * Aj[k];
            Ai[j] = t / d;
        }
    }
    return true;
}
static void chol_solve(const double* L, int n, double* b) {
    for (int i = 0; i < n; i++) {
        const double* Li = L + (size_t)i * n;
        double s = b[i];
        for (int k = 0; k < i; k++) s -= Li[k] * b[k];
        b[i] = s / Li[i];
    }
    for (int i = n - 1; i >= 0; i--) {
        double s = b[i];
        for (int k = i + 1; k < n; k++) s -= L[(size_t)k * n + i] * b[k];
        b[i] = s / L[(size_t)i * n + i];
    }
}
// Solves (S) x = b where S is symmetric, given in the upper triangle of `src` plus `diag_add` on the diagonal.
// Uses `work` (n*n). Adds a tiny jitter and retries if the matrix is not numerically positive definite.
static bool sym_solve(const double* src, const double* diag_add, int n, double* work, double* b) {
    for (int attempt = 0; attempt < 6; attempt++) {
        double jit = attempt ? std::pow(10.0, -12 + 2 * attempt) : 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) work[(size_t)i * n + j] = src[(size_t)j * n + i];
        for (int i = 0; i < n; i++) work[(size_t)i * n + i] += diag_add[i] + jit * (1.0 + std::fabs(work[(size_t)i * n + i]));
        if (cholesky(work, n)) {
            chol_solve(work, n, b);
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------- FFT (any length, Bluestein over radix 2)
typedef std::complex<double> cplx;
static void fft_pow2(cplx* a, int n, bool inv) {
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (inv ? 1 : -1);
        for (int i = 0; i < n; i += len)
            for (int k = 0; k < len / 2; k++) {
                cplx w(std::cos(ang * k), std::sin(ang * k));
                cplx u = a[i + k], v = a[i + k + len / 2] * w;
                a[i + k] = u + v;
                a[i + k + len / 2] = u - v;
            }
    }
    if (inv)
        for (int i = 0; i < n; i++) a[i] /= n;
}
// In-place DFT of length n (forward: exp(-2 pi i k t / n)); inverse includes 1/n.
static void dft(vector<cplx>& x, bool inv) {
    int n = (int)x.size();
    if (n <= 1) return;
    if ((n & (n - 1)) == 0) {
        fft_pow2(x.data(), n, inv);
        return;
    }
    int m = 1;
    while (m < 2 * n - 1) m <<= 1;
    double sg = inv ? 1.0 : -1.0;
    vector<cplx> w(n), a(m), b(m);
    for (int k = 0; k < n; k++) {
        double ang = M_PI * (double)((int64_t)k * k % (2LL * n)) / n;
        w[k] = cplx(std::cos(ang), sg * std::sin(ang));
    }
    for (int k = 0; k < n; k++) a[k] = x[k] * w[k];
    b[0] = std::conj(w[0]);
    for (int k = 1; k < n; k++) b[k] = b[m - k] = std::conj(w[k]);
    fft_pow2(a.data(), m, false);
    fft_pow2(b.data(), m, false);
    for (int i = 0; i < m; i++) a[i] *= b[i];
    fft_pow2(a.data(), m, true);
    for (int k = 0; k < n; k++) x[k] = a[k] * w[k] * (inv ? 1.0 / n : 1.0);
}

// ---------------------------------------------------------------- input rows on a regular grid
struct Rows {
    vector<int64_t> tns;   // grid timestamps: input rows, rows re-inserted in deleted gaps, blind tail rows
    vector<float> imu;     // gyro xyz, accel xyz per row; zero on invalid rows
    vector<uint8_t> flag;  // 1: valid, 2: written to the output (input rows and blind tail rows)
    int n = 0, last = -1;
    double med_dt = 0.005;  // np.median(np.diff(t)) of the grid
    inline double t(int k) const { return (double)(tns[k] - tns[0]) * 1e-9; }
    inline bool ok(int k) const { return flag[k] & 1; }
    inline V3 gyro(int k) const { return {imu[6 * (size_t)k], imu[6 * (size_t)k + 1], imu[6 * (size_t)k + 2]}; }
    inline V3 acc(int k) const { return {imu[6 * (size_t)k + 3], imu[6 * (size_t)k + 4], imu[6 * (size_t)k + 5]}; }
};

// One CSV row: timestamp (ns) and six values. Returns 0 at end of file, 1 for a row. `good` is false for a row whose
// values are missing, not finite, all zero, an exact repeat of the row before (frozen sensor), or beyond any sensor's range.
struct Reader {
    FILE* f;
    double prev[6];
    bool have_prev = false;
    int next(int64_t& t, double v[6], bool& good) {
        char line[1024];
        while (std::fgets(line, sizeof line, f)) {
            char* p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '#' || *p == '\n' || *p == '\r' || *p == 0) continue;
            char* end = nullptr;
            long long tt = std::strtoll(p, &end, 10);
            if (end == p) continue;  // header or junk
            t = tt;
            p = end;
            good = true;
            for (int i = 0; i < 6; i++) {
                while (*p == ',' || *p == ' ' || *p == '\t') p++;
                v[i] = std::strtod(p, &end);
                if (end == p) {
                    good = false;
                    v[i] = 0;
                } else
                    p = end;
            }
            bool zero = true, same = have_prev;
            for (int i = 0; i < 6; i++) {
                if (!std::isfinite(v[i])) good = false;
                if (v[i] != 0.0) zero = false;
                if (!(v[i] == prev[i])) same = false;
            }
            if (zero || same) good = false;
            std::memcpy(prev, v, sizeof prev);
            have_prev = true;
            if (good && (norm({v[0], v[1], v[2]}) > 60.0 || norm({v[3], v[4], v[5]}) > 400.0)) good = false;
            return 1;
        }
        return 0;
    }
};

static FILE* open_input(const char* path) {
    static char buf[2048];
    const char* tries[] = {"%s/imu0/data.csv", "%s/mav0/imu0/data.csv"};
    for (const char* t : tries) {
        std::snprintf(buf, sizeof buf, t, path);
        if (FILE* f = std::fopen(buf, "rb")) return f;
    }
    return std::fopen(path, "rb");
}

// First pass: sample period, size of the regular grid (common.to_grid) and the blind tail (pipeline.prepare).
struct Scan {
    int64_t med = 0, t_first = 0, thr = 0;
    long n_in = 0, n_grid = 0, last_grid = -1, n_add = 0;
    int fs = 200, dec = 1;
};
static const double kBlindS = 5.0;

static bool scan_file(const char* path, Scan& S) {
    FILE* f = open_input(path);
    if (!f) return false;
    static char fbuf[1 << 16];
    std::setvbuf(f, fbuf, _IOFBF, sizeof fbuf);
    Reader rd{f, {0}};
    int64_t t, tprev = 0;
    double v[6];
    bool good;
    vector<int32_t> d;                          // row-to-row steps, clamped (only the median needs them)
    vector<std::pair<long, int64_t>> big;       // steps that do not fit
    long last_in = -1;
    while (rd.next(t, v, good)) {
        if (S.n_in == 0)
            S.t_first = t;
        else {
            int64_t dt = t - tprev;
            if (dt > INT32_MAX || dt < INT32_MIN) big.push_back({S.n_in, dt});
            d.push_back((int32_t)std::min<int64_t>(std::max<int64_t>(dt, INT32_MIN), INT32_MAX));
        }
        if (good) last_in = S.n_in;
        tprev = t;
        S.n_in++;
    }
    std::fclose(f);
    if (S.n_in < 10 || last_in < 0) return false;
    {
        vector<int32_t> s(d);
        size_t h = s.size() / 2;
        std::nth_element(s.begin(), s.begin() + h, s.end());
        if (s.size() & 1)
            S.med = s[h];
        else {
            int32_t lo = *std::max_element(s.begin(), s.begin() + h);
            S.med = (int64_t)(((double)lo + (double)s[h]) / 2.0);
        }
    }
    if (S.med <= 0) S.med = 1;
    S.thr = S.med * 3 / 2;
    long g = 0;
    size_t bi = 0;
    for (long k = 0; k < S.n_in; k++) {
        if (k > 0) {
            int64_t dt = d[k - 1];
            while (bi < big.size() && big[bi].first < k) bi++;
            if (bi < big.size() && big[bi].first == k) dt = big[bi].second;
            if (dt > S.thr) g += py_round((double)dt / (double)S.med) - 1;
        }
        if (k == last_in) S.last_grid = g;
        g++;
    }
    S.n_grid = g;
    long n_tail = py_round(kBlindS * 1e9 / (double)S.med);
    S.n_add = std::max(0L, n_tail - (S.n_grid - 1 - S.last_grid));
    S.fs = std::max(1, py_round(1e9 / (double)S.med));
    // IMUs faster than 300 Hz are solved on the average of every `dec` rows (about 200 Hz) and written at full rate
    S.dec = S.fs > 300 ? std::max(2, py_round(S.fs / 200.0)) : 1;
    return true;
}

// Grid rows [r0, r1) of the file (rows re-inserted in deleted gaps), plus the blind tail if requested.
static bool load_range(const char* path, const Scan& S, long r0, long r1, bool tail, Rows& R) {
    FILE* f = open_input(path);
    if (!f) return false;
    static char fbuf[1 << 16];
    std::setvbuf(f, fbuf, _IOFBF, sizeof fbuf);
    Reader rd{f, {0}};
    size_t cap = (size_t)(r1 - r0) + (tail ? (size_t)S.n_add : 0);
    R.tns.clear();
    R.imu.clear();
    R.flag.clear();
    R.tns.reserve(cap);
    R.imu.reserve(cap * 6);
    R.flag.reserve(cap);
    int64_t t, tprev = 0;
    double v[6];
    bool good;
    long g = 0;
    for (long k = 0; g < r1 && rd.next(t, v, good); k++) {
        if (k > 0 && t - tprev > S.thr) {
            int64_t dt = t - tprev;
            int m = py_round((double)dt / (double)S.med) - 1;
            for (int i = 1; i <= m; i++, g++)
                if (g >= r0 && g < r1) {
                    R.tns.push_back(tprev + (dt * i) / (m + 1));
                    for (int c = 0; c < 6; c++) R.imu.push_back(0.0f);
                    R.flag.push_back(0);
                }
        }
        if (g >= r0 && g < r1) {
            R.tns.push_back(t);
            for (int c = 0; c < 6; c++) R.imu.push_back(good ? (float)v[c] : 0.0f);
            R.flag.push_back((uint8_t)((good ? 1 : 0) | 2));
        }
        g++;
        tprev = t;
    }
    std::fclose(f);
    if (tail && !R.tns.empty()) {
        int64_t tl = R.tns.back();
        for (long i = 1; i <= S.n_add; i++) {
            R.tns.push_back(tl + S.med * i);
            for (int c = 0; c < 6; c++) R.imu.push_back(0.0f);
            R.flag.push_back(2);
        }
    }
    R.n = (int)R.tns.size();
    R.last = -1;
    for (int k = R.n - 1; k >= 0; k--)
        if (R.flag[k] & 1) {
            R.last = k;
            break;
        }
    if (R.n < 2) return false;
    vector<double> dd(R.n - 1);
    for (int k = 1; k < R.n; k++) dd[k - 1] = R.t(k) - R.t(k - 1);
    R.med_dt = median_of(dd);
    return true;
}

// Full-rate grid rows one at a time, straight from the file: input rows, rows re-inserted in deleted gaps (not written),
// then the blind tail if requested. `g` is the grid index. Holds nothing but the current row.
struct GridRow {
    int64_t tns;
    double v[6];
    bool ok, out;
    long g;
};
struct GridStream {
    FILE* f = nullptr;
    Reader rd{nullptr, {0}};
    const Scan* S = nullptr;
    bool tail = false, eof = false, held = false, held_good = false;
    long g = 0, k = 0, tail_i = 0;
    int pend_m = 0, pend_i = 0;
    int64_t tprev = 0, pend_t0 = 0, pend_dt = 0, held_t = 0, t_last = 0;
    double held_v[6];
    vector<char> buf;
    bool open(const char* path, const Scan& s, bool with_tail) {
        f = open_input(path);
        if (!f) return false;
        buf.resize(1 << 16);
        std::setvbuf(f, buf.data(), _IOFBF, buf.size());
        rd.f = f;
        S = &s;
        tail = with_tail;
        return true;
    }
    ~GridStream() {
        if (f) std::fclose(f);
    }
    bool next(GridRow& r) {
        for (;;) {
            if (pend_i < pend_m) {
                pend_i++;
                r.tns = pend_t0 + (pend_dt * pend_i) / (pend_m + 1);
                std::fill(r.v, r.v + 6, 0.0);
                r.ok = r.out = false;
                r.g = g++;
                return true;
            }
            if (held) {
                held = false;
                r.tns = held_t;
                for (int c = 0; c < 6; c++) r.v[c] = held_good ? held_v[c] : 0.0;
                r.ok = held_good;
                r.out = true;
                r.g = g++;
                tprev = t_last = held_t;
                return true;
            }
            if (!eof) {
                int64_t t;
                bool good;
                if (rd.next(t, held_v, good)) {
                    if (k > 0 && t - tprev > S->thr) {
                        pend_dt = t - tprev;
                        pend_t0 = tprev;
                        pend_m = py_round((double)pend_dt / (double)S->med) - 1;
                        pend_i = 0;
                    }
                    k++;
                    held = true;
                    held_t = t;
                    held_good = good;
                    continue;
                }
                eof = true;
            }
            if (tail && tail_i < S->n_add) {
                tail_i++;
                r.tns = t_last + S->med * tail_i;
                std::fill(r.v, r.v + 6, 0.0);
                r.ok = false;
                r.out = true;
                r.g = g++;
                return true;
            }
            return false;
        }
    }
};

// Grid rows [r0, r1) (+ blind tail) averaged over groups of D rows: the rows the solver sees for a fast IMU. A group's
// time is that of its last row, so integrating the group's mean rate from the previous group covers exactly its rows.
static bool load_range_dec(const char* path, const Scan& S, long r0, long r1, bool tail, int D, Rows& R) {
    GridStream gs;
    if (!gs.open(path, S, tail)) return false;
    R.tns.clear();
    R.imu.clear();
    R.flag.clear();
    size_t cap = (size_t)((r1 - r0) + (tail ? S.n_add : 0)) / D + 2;
    R.tns.reserve(cap);
    R.imu.reserve(cap * 6);
    R.flag.reserve(cap);
    GridRow r;
    double sum[6] = {0};
    int cnt = 0, nv = 0;
    int64_t tl = 0;
    auto flush = [&]() {
        R.tns.push_back(tl);
        for (int c = 0; c < 6; c++) R.imu.push_back(nv ? (float)(sum[c] / nv) : 0.0f);
        R.flag.push_back((uint8_t)((nv ? 1 : 0) | 2));
        std::fill(sum, sum + 6, 0.0);
        cnt = nv = 0;
    };
    while (gs.next(r)) {
        if (r.g < r0) continue;
        if (r.g >= r1 && !(tail && r.g >= S.n_grid)) break;
        if (r.ok) {
            for (int c = 0; c < 6; c++) sum[c] += r.v[c];
            nv++;
        }
        tl = r.tns;
        if (++cnt == D) flush();
    }
    if (cnt) flush();
    R.n = (int)R.tns.size();
    R.last = -1;
    for (int k = R.n - 1; k >= 0; k--)
        if (R.flag[k] & 1) {
            R.last = k;
            break;
        }
    if (R.n < 2) return false;
    vector<double> dd(R.n - 1);
    for (int k = 1; k < R.n; k++) dd[k - 1] = R.t(k) - R.t(k - 1);
    R.med_dt = median_of(dd);
    return true;
}

// ---------------------------------------------------------------- row repair for the organisers' degradation
// tools/degrade_imu.py removes single rows and bursts of up to 50 rows, repeats a value for 2-10 rows, and puts a spike
// on every channel of isolated rows. Left alone, each of the hundreds of holes would start a new segment with its own
// tilt and velocity step, and each spike would kick the attitude. So:
//   1. a live row that is far from the median of the five live rows around it, on any channel, is marked dead. "Far" is
//      10 times the usual size of that deviation in the same 2 s of data, plus a floor, so it follows the noise level.
//      A steady change (a fast turn) has zero deviation from the median of five and is never flagged.
//   2. short dead runs with live rows on both sides are filled by straight-line interpolation and count as live from
//      then on: up to 0.08 s on handheld files and 0.25 s on drones. A hand turns too erratically to interpolate
//      further, so its longer bursts stay dropouts (own tilt and velocity step); a drone's turn rate is smooth.
//      The dead rows at the end stay the blind window.
static double g_bridge_s = PF_BRIDGE_HAND;  // longest dead run that is filled by interpolation (s)
static void bridge_rows(Rows& R, double seconds);
static inline float med5(float a, float b, float c, float d, float e) {
    float v[5] = {a, b, c, d, e};
    std::sort(v, v + 5);
    return v[2];
}
static void clean_rows(Rows& R) {
#if PF_CLEAN_ROWS
    int last = R.last;
    if (last < 8) return;
    int fs = py_round(1.0 / R.med_dt);
    {
        vector<float> dev(6 * ((size_t)last + 1), 0.0f);
        vector<uint8_t> has((size_t)last + 1, 0);
        for (int k = 2; k + 2 <= last; k++) {
            if (!(R.ok(k - 2) && R.ok(k - 1) && R.ok(k) && R.ok(k + 1) && R.ok(k + 2))) continue;
            has[k] = 1;
            for (int c = 0; c < 6; c++) {
                const float* p = &R.imu[6 * (size_t)k + c];
                dev[6 * (size_t)k + c] = std::fabs(p[0] - med5(p[-12], p[-6], p[0], p[6], p[12]));
            }
        }
        int blk = std::max(10, 2 * fs);
        vector<double> tmp;
        for (int b0 = 0; b0 <= last; b0 += blk) {
            int b1 = std::min(last + 1, b0 + blk);
            double sc[6];
            for (int c = 0; c < 6; c++) {
                tmp.clear();
                for (int k = b0; k < b1; k++)
                    if (has[k]) tmp.push_back(dev[6 * (size_t)k + c]);
                sc[c] = tmp.empty() ? 0.0 : 1.4826 * median_of(tmp);
            }
            for (int k = b0; k < b1; k++) {
                if (!has[k]) continue;
                bool spike = false;
                for (int c = 0; c < 6 && !spike; c++) spike = dev[6 * (size_t)k + c] > 10.0 * sc[c] + (c < 3 ? 0.02 : 0.3);
                if (spike) R.flag[k] &= (uint8_t)~1;
            }
        }
    }
    bridge_rows(R, g_bridge_s);
#else
    (void)R;
#endif
}
// Step 2 of the repair. Called again with the longer limit once a file is known to be a drone.
static void bridge_rows(Rows& R, double seconds) {
#if PF_CLEAN_ROWS
    int last = R.last;
    if (last < 8) return;
    int max_run = std::max(1, (int)(seconds * py_round(1.0 / R.med_dt)));
    for (int k = 1; k < last;) {
        if (R.ok(k)) {
            k++;
            continue;
        }
        int s = k;
        while (k <= last && !R.ok(k)) k++;
        if (k > last || k - s > max_run || !R.ok(s - 1)) continue;
        const float* a = &R.imu[6 * (size_t)(s - 1)];
        const float* b = &R.imu[6 * (size_t)k];
        double ta = R.t(s - 1), span = R.t(k) - ta;
        for (int i = s; i < k; i++) {
            double w = span > 0 ? (R.t(i) - ta) / span : 0.5;
            for (int c = 0; c < 6; c++) R.imu[6 * (size_t)i + c] = (float)(a[c] * (1 - w) + b[c] * w);
            R.flag[i] |= 1;
        }
    }
#else
    (void)R;
#endif
}

static bool load_any(const char* path, const Scan& S, long r0, long r1, bool tail, Rows& R) {
    bool ok = S.dec > 1 ? load_range_dec(path, S, r0, r1, tail, S.dec, R) : load_range(path, S, r0, r1, tail, R);
    if (ok) clean_rows(R);
    return ok;
}

// Runs of invalid rows (attitude.find_gaps)
static vector<Run> find_gaps(const Rows& R) {
    vector<Run> g;
    int k = 0;
    while (k < R.n) {
        if (!R.ok(k)) {
            int s = k;
            while (k < R.n && !R.ok(k)) k++;
            g.push_back({s, k});
        } else
            k++;
    }
    return g;
}

// Gap-filled values of one column (0..2 gyro, 3..5 accel) for rows [k0, k1): np.interp over valid rows, edges held.
static void fill_range(const Rows& R, int col, int k0, int k1, double* out) {
    int prev = k0 - 1;
    while (prev >= 0 && !R.ok(prev)) prev--;
    int next = -1;
    for (int k = k0; k < k1; k++) {
        if (R.ok(k)) {
            out[k - k0] = R.imu[6 * (size_t)k + col];
            prev = k;
            continue;
        }
        if (next < k) {
            next = k + 1;
            while (next < R.n && !R.ok(next)) next++;
        }
        double a = prev >= 0 ? R.imu[6 * (size_t)prev + col] : 0.0, b = next < R.n ? R.imu[6 * (size_t)next + col] : 0.0;
        if (prev < 0 && next >= R.n)
            out[k - k0] = 0.0;
        else if (prev < 0)
            out[k - k0] = b;
        else if (next >= R.n)
            out[k - k0] = a;
        else
            out[k - k0] = a + (b - a) * (double)(k - prev) / (double)(next - prev);
    }
}
static inline double diff3(const double* x) { return ((x[3] - x[2]) - (x[2] - x[1])) - ((x[2] - x[1]) - (x[1] - x[0])); }

// phys.rotors_on: rotor signature per 1 s window, from the third difference of the gap-filled accelerometer.
static vector<uint8_t> rotor_windows(const Rows& R, int fs, int& win) {
    win = std::max(1, fs);
    int n = R.n;
    vector<uint8_t> on((n + win - 1) / win);
    vector<double> buf, seg(win);
    for (int i = 0, w = 0; i < n; i += win, w++) {
        int len = std::min(win, n - i);
        double s = 0;
        if (len > 5) {
            buf.clear();
            for (int c = 0; c < 3; c++) {
                fill_range(R, 3 + c, i, i + len, seg.data());
                for (int k = 0; k + 3 < len; k++) buf.push_back(std::fabs(diff3(&seg[k])));
            }
            s = median_of(buf) / (0.6745 * std::sqrt(20.0));
        }
        on[w] = s > 0.35;
    }
    return on;
}

// phys.hardened_rest: gyro means of still blocks with the rotors off (rows [k0, k1) only), appended to `out`.
static void rest_blocks(const Rows& R, double g, int k0, int k1, vector<V3>& out) {
    int fs = py_round(1.0 / R.med_dt);
    int win = (int)(0.5 * fs);
    if (win < 8) return;
    vector<double> buf, x(win);
    auto noise3 = [&](int s, int c) {
        buf.clear();
        for (int k = 0; k < win; k++) x[k] = R.imu[6 * (size_t)(s + k) + c];
        for (int k = 0; k + 3 < win; k++) buf.push_back(std::fabs(diff3(&x[k])));
        return median_of(buf) / (0.6745 * std::sqrt(20.0));
    };
    for (int s = k0; s < std::min(k1, R.n - win); s += win) {
        bool all = true;
        for (int k = s; k < s + win && all; k++) all = R.ok(k);
        if (!all) continue;
        double mean[6] = {0}, sd[6] = {0};
        for (int c = 0; c < 6; c++) {
            double m = 0;
            for (int k = s; k < s + win; k++) m += R.imu[6 * (size_t)k + c];
            m /= win;
            double q = 0;
            for (int k = s; k < s + win; k++) {
                double d = R.imu[6 * (size_t)k + c] - m;
                q += d * d;
            }
            mean[c] = m;
            sd[c] = std::sqrt(q / win);
        }
        double sdmax = std::max(sd[0], std::max(sd[1], sd[2]));
        V3 am{mean[3], mean[4], mean[5]};
        if (!(std::fabs(norm(am) - g) < 0.2 && sdmax < 0.03)) continue;
        double ratio = 0, aN3 = 0;
        for (int c = 0; c < 3; c++) ratio = std::max(ratio, sd[c] / std::max(noise3(s, c), 1e-4));
        for (int c = 3; c < 6; c++) aN3 = std::max(aN3, noise3(s, c));
        double aw = std::max(sd[3], std::max(sd[4], sd[5]));
        if ((ratio < 1.8 || sdmax < 0.006) && aw / std::max(aN3, 1e-4) < 3.0 && aN3 < 0.3) out.push_back({mean[0], mean[1], mean[2]});
    }
}
static bool rest_median(const vector<V3>& bl, V3& rest) {
    if (bl.size() < 2) return false;
    vector<double> bx, by, bz;
    for (auto& b : bl) {
        bx.push_back(b.x);
        by.push_back(b.y);
        bz.push_back(b.z);
    }
    rest = {median_of(bx), median_of(by), median_of(bz)};
    return true;
}
static double accel_g(const Rows& R) {
    vector<double> an;
    an.reserve(R.n);
    for (int k = 0; k < R.n; k++)
        if (R.ok(k)) an.push_back(norm(R.acc(k)));
    return median_of(an);
}

// ---------------------------------------------------------------- gyro bias from short windows (wbias.py)
// Used only for a drone whose file has no rest period, to check the declared calibration against the file itself.
// In each 4 s window without dead rows, the up direction that the accelerometer shows at the start is carried to the end
// by the gyro and compared with the up direction shown there. The mismatch is linear in the gyro bias, so all windows
// give one small least-squares problem (three passes, Huber weights). Unlike the whole-file gravity fit this works from
// any starting error. Only the bias across the up direction can be seen, so the component along the mean accelerometer
// direction `up` is removed from the result. On a drone the accelerometer shows the thrust axis, and real tilt changes
// act as noise with zero mean. Returns the number of windows used.
static void rot_exp3(V3 r, double* E) {
    double th = norm(r);
    double K[9] = {0, -r.z, r.y, r.z, 0, -r.x, -r.y, r.x, 0};
    double a = 1.0, c = 0.0;
    if (th >= 1e-9) {
        a = std::sin(th) / th;
        c = (1 - std::cos(th)) / (th * th);
    }
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            double kk = 0;
            for (int m = 0; m < 3; m++) kk += K[3 * i + m] * K[3 * m + j];
            E[3 * i + j] = (i == j ? 1.0 : 0.0) + a * K[3 * i + j] + c * kk;
        }
}
static bool solve3(double* A, double* b) {  // Gaussian elimination with partial pivoting; A and b are overwritten
    for (int c = 0; c < 3; c++) {
        int p = c;
        for (int r = c + 1; r < 3; r++)
            if (std::fabs(A[3 * r + c]) > std::fabs(A[3 * p + c])) p = r;
        if (std::fabs(A[3 * p + c]) < 1e-300) return false;
        if (p != c) {
            for (int j = 0; j < 3; j++) std::swap(A[3 * p + j], A[3 * c + j]);
            std::swap(b[p], b[c]);
        }
        for (int r = c + 1; r < 3; r++) {
            double f = A[3 * r + c] / A[3 * c + c];
            for (int j = c; j < 3; j++) A[3 * r + j] -= f * A[3 * c + j];
            b[r] -= f * b[c];
        }
    }
    for (int c = 2; c >= 0; c--) {
        for (int j = c + 1; j < 3; j++) b[c] -= A[3 * c + j] * b[j];
        b[c] /= A[3 * c + c];
    }
    return true;
}
static int window_bias(const Rows& R, V3& b_out, V3& up) {
    b_out = {0, 0, 0};
    up = {0, 0, 0};
    int fs = py_round(1.0 / R.med_dt);
    int wn = py_round(4.0 * fs), en = std::max(1, py_round(0.5 * fs));
    if (R.last + 1 <= wn) return 0;
    for (int k = 0; k <= R.last; k++)
        if (R.ok(k)) up = up + R.acc(k);
    double nu = norm(up);
    if (nu < 1e-9) return 0;
    up = up * (1.0 / nu);
    vector<int> wins;
    for (int k = 0; k < R.last + 1 - wn; k += wn) {
        bool live = true;
        for (int i = k; i < k + wn && live; i++) live = R.ok(i);
        if (live) wins.push_back(k);
    }
    size_t nw = wins.size();
    if (!nw) return 0;
    vector<double> rr(3 * nw), SS(9 * nw), res, tmp;
    vector<uint8_t> used(nw);
    V3 b{0, 0, 0};
    int n_used = 0;
    for (int it = 0; it < 3; it++) {
        res.clear();
        for (size_t w = 0; w < nw; w++) {
            int k0 = wins[w], k1 = k0 + wn;
            V3 m0{0, 0, 0}, m1{0, 0, 0};
            for (int i = k0; i < k0 + en; i++) m0 = m0 + R.acc(i);
            for (int i = k1 - en; i < k1; i++) m1 = m1 + R.acc(i);
            m0 = m0 * (1.0 / en);
            m1 = m1 * (1.0 / en);
            double n0 = norm(m0), n1 = norm(m1);
            used[w] = !(n0 < 1e-6 || n1 < 1e-6);
            if (!used[w]) continue;
            m0 = m0 * (1.0 / n0);
            m1 = m1 * (1.0 / n1);
            double C[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, IC[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, E[9], Cn[9];
            for (int i = k0 + en / 2; i < k1 - en / 2; i++) {
                double dt = i + 1 < R.n ? R.t(i + 1) - R.t(i) : 1.0 / fs;
                for (int j = 0; j < 9; j++) IC[j] += C[j] * dt;
                rot_exp3((R.gyro(i) - b) * dt, E);
                for (int a = 0; a < 3; a++)
                    for (int c = 0; c < 3; c++) Cn[3 * a + c] = C[3 * a] * E[c] + C[3 * a + 1] * E[3 + c] + C[3 * a + 2] * E[6 + c];
                for (int j = 0; j < 9; j++) C[j] = Cn[j];
            }
            // mismatch r = m1 - transpose(C) m0 and its sensitivity S = -transpose(C) [m0]x IC
            double mm[3] = {m0.x, m0.y, m0.z}, m1v[3] = {m1.x, m1.y, m1.z};
            double K[9] = {0, -mm[2], mm[1], mm[2], 0, -mm[0], -mm[1], mm[0], 0}, KI[9];
            for (int a = 0; a < 3; a++)
                for (int c = 0; c < 3; c++) KI[3 * a + c] = K[3 * a] * IC[c] + K[3 * a + 1] * IC[3 + c] + K[3 * a + 2] * IC[6 + c];
            double r2 = 0;
            for (int a = 0; a < 3; a++) {
                double v = m1v[a] - (C[a] * mm[0] + C[3 + a] * mm[1] + C[6 + a] * mm[2]);
                rr[3 * w + a] = v;
                r2 += v * v;
                for (int c = 0; c < 3; c++) SS[9 * w + 3 * a + c] = -(C[a] * KI[c] + C[3 + a] * KI[3 + c] + C[6 + a] * KI[6 + c]);
            }
            res.push_back(std::sqrt(r2));
        }
        n_used = (int)res.size();
        if (!n_used) return 0;
        tmp = res;
        double sc = 1.4826 * median_of(tmp) + 1e-6;
        double N[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0}, g[3] = {0, 0, 0};
        size_t ri = 0;
        for (size_t w = 0; w < nw; w++) {
            if (!used[w]) continue;
            double wt = std::min(1.0, 1.5 * sc / std::max(res[ri++], 1e-12));
            const double* S = &SS[9 * w];
            const double* r = &rr[3 * w];
            for (int a = 0; a < 3; a++) {
                for (int c = 0; c < 3; c++) N[3 * a + c] += wt * (S[a] * S[c] + S[3 + a] * S[3 + c] + S[6 + a] * S[6 + c]);
                g[a] += wt * (S[a] * r[0] + S[3 + a] * r[1] + S[6 + a] * r[2]);
            }
        }
        double lam = 1e-2 * (N[0] + N[4] + N[8]) / 3.0, bv[3] = {b.x, b.y, b.z}, rhs[3], A[9];
        for (int a = 0; a < 3; a++) {
            rhs[a] = g[a] + N[3 * a] * bv[0] + N[3 * a + 1] * bv[1] + N[3 * a + 2] * bv[2];
            for (int c = 0; c < 3; c++) A[3 * a + c] = N[3 * a + c] + (a == c ? lam : 0.0);
        }
        if (!solve3(A, rhs)) return 0;
        b = {rhs[0], rhs[1], rhs[2]};
    }
    b_out = b - up * dot(b, up);
    return n_used;
}
// A drone with no rest period: keep the declared calibration only if the file agrees with it across the up direction
// within 0.03 rad/s (normal files: at most 0.022; bias-compensated or re-oriented files: 0.035 and more).
static V3 check_cal_windows(const Rows& R, V3 cal) {
    V3 bw, up;
    if (window_bias(R, bw, up) < 5) return cal;
    V3 d = bw - (cal - up * dot(cal, up));
    return norm(d) < 0.03 ? cal : bw;
}

// ---------------------------------------------------------------- gyro integration (attitude.integrate_rows)
// Visits every row with its attitude. Valid rows use gyro - bias(t); invalid rows coast on a rate interpolated across
// the gap (or decaying, after the last valid row). Optional world tilt steps spread a correction across bracketed gaps.
struct Integrator {
    const Rows& R;
    const vector<Run>& gaps_all;
    const vector<double>& kn;   // bias spline knots
    const vector<V3>& coef;     // bias spline values at the knots
    V3 b0;
    double t_end;
    int side;
    const vector<Run>* tilt_gaps = nullptr;            // bracketed gaps that carry a tilt step
    const vector<std::pair<double, double>>* theta = nullptr;

    inline V3 bias(int k) const { return spline_eval(kn, coef, std::min(R.t(k), t_end)) + b0; }
    template <class F>
    void run(Qt q0, F visit) const {
        Qt q = q0;
        visit(0, q);
        size_t gi = 0, ti = 0;
        V3 wb{0, 0, 0}, wa{0, 0, 0};
        bool has_after = false;
        int gs = -1, ge = -1;
        // bias at row k is evaluated incrementally (rows are visited in time order)
        int K = (int)kn.size();
        int cnt = 0;  // number of knots strictly below tt
        for (int k = 1; k < R.n; k++) {
            double dt = R.t(k) - R.t(k - 1);
            V3 w;
            if (R.ok(k)) {
                double tt = std::min(R.t(k), t_end);
                while (cnt < K && kn[cnt] < tt) cnt++;
                int j = std::min(std::max(cnt - 1, 0), K - 2);
                double x = std::min(std::max((tt - kn[j]) / (kn[j + 1] - kn[j]), 0.0), 1.0);
                V3 b = coef[j] * (1 - x) + coef[j + 1] * x + b0;
                w = R.gyro(k) - b;
            } else {
                if (k >= ge) {  // entering a new gap
                    while (gi < gaps_all.size() && gaps_all[gi].second <= k) gi++;
                    gs = gaps_all[gi].first;
                    ge = gaps_all[gi].second;
                    wb = {0, 0, 0};
                    int a0 = std::max(0, gs - side);
                    for (int r = a0; r < gs; r++) wb = wb + (R.gyro(r) - bias(r));
                    if (gs > a0) wb = wb * (1.0 / (gs - a0));
                    has_after = ge < R.n;
                    wa = {0, 0, 0};
                    if (has_after) {
                        int a1 = std::min(R.n, ge + side);
                        for (int r = ge; r < a1; r++) wa = wa + (R.gyro(r) - bias(r));
                        wa = wa * (1.0 / (a1 - ge));
                    }
                }
                int nn = ge - gs;
                if (has_after) {
#if PF_GAP_BRIDGE == 0
                    double x = (double)(k - gs + 1) / (nn + 1.0);
                    w = wb * (1 - x) + wa * x;
#elif PF_GAP_BRIDGE == 1
                    (void)nn;
                    w = {0, 0, 0};
#else
                    (void)nn;
                    double tb_ = gs > 0 ? R.t(gs - 1) : R.t(gs), ta_ = R.t(ge);
                    w = wb * std::exp(-(R.t(k) - tb_) / PF_GAP_TAU) + wa * std::exp(-(ta_ - R.t(k)) / PF_GAP_TAU);
#endif
                } else {
                    double t0 = gs > 0 ? R.t(gs - 1) : R.t(gs);
                    w = wb * std::exp(-(R.t(k) - t0) / 0.5);
                }
            }
            q = qmul(q, qexp(w * dt));
            if (tilt_gaps) {
                while (ti < tilt_gaps->size() && (*tilt_gaps)[ti].second <= k) ti++;
                if (ti < tilt_gaps->size() && (*tilt_gaps)[ti].first <= k) {
                    const Run& g = (*tilt_gaps)[ti];
                    double d = (double)std::max(1, g.second - g.first);
                    V3 ts{(*theta)[ti].first / d, (*theta)[ti].second / d, 0.0};
                    if (ts.x != 0.0 || ts.y != 0.0) q = qmul(qexp(ts), q);
                }
            }
            double nq = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
            q = {q.w / nq, q.x / nq, q.y / nq, q.z / nq};
            visit(k, q);
        }
    }
};

// common.quiet_init: start attitude from the quietest second, gyro already bias-corrected by b0.
static Qt quiet_init(const Rows& R, V3 b0) {
    int fs = py_round(1.0 / R.med_dt);
    double best = 1e300;
    int bk = -1;
    V3 ba{0, 0, 9.81};
    for (int s = 0; s < R.n - fs; s += fs) {
        int e = s + fs, m = 0;
        V3 sw{0, 0, 0}, sa{0, 0, 0};
        double sww = 0, saa = 0;
        for (int k = s; k < e; k++)
            if (R.ok(k)) {
                V3 w = R.gyro(k) - b0, a = R.acc(k);
                sw = sw + w;
                sa = sa + a;
                sww += dot(w, w);
                saa += dot(a, a);
                m++;
            }
        if (m < 0.8 * fs) continue;
        V3 mw = sw * (1.0 / m), ma = sa * (1.0 / m);
        double var_w = sww / m - dot(mw, mw), var_a = saa / m - dot(ma, ma);
        double off = norm(ma) - 9.81;
        double score = var_w + 0.01 * var_a + 0.01 * off * off;
        if (score < best) {
            best = score;
            bk = s + fs / 2;
            ba = ma;
        }
    }
    if (bk < 0) bk = 0;
    V3 a = ba * (1.0 / norm(ba)), b{0, 0, 1};
    double d = dot(a, b);
    Qt qup;
    if (d < -0.999999)
        qup = {0, 1, 0, 0};
    else {
        V3 c = cross(a, b);
        qup = qnormalize({1 + d, c.x, c.y, c.z});
    }
    Qt q{1, 0, 0, 0};
    for (int k = 1; k < bk; k++)
        if (R.ok(k)) q = qmul(q, qexp((R.gyro(k) - b0) * (R.t(k) - R.t(k - 1))));
    q = qnormalize(q);
    return qnormalize(qmul(qup, qconj(q)));
}

// ---------------------------------------------------------------- gravity fit of the gyro bias (attitude.offline_attitude)
// Returns the bias spline values at its knots (8 s), used to seed the main solve.
static void offline_bias(const Rows& R, V3 b0, double g, double knot_s, vector<double>& kn_out, vector<V3>& c) {
    const double s_r = 0.3, sigma_b = 0.003, sigma_gap = 0.5;
    const int iters = 3;
    int n = R.n, last = R.last;
    double t0 = R.t(0), t_end = R.t(last);
    Qt q0 = quiet_init(R, b0);
    int J = std::max(2, py_round((t_end - t0) / knot_s) + 1);
    linspace(t0, t_end, J, kn_out);
    kn_out[J - 1] += 1e-3;
    vector<Run> gaps_all = find_gaps(R), gaps;
    for (auto& r : gaps_all)
        if (r.second < n && r.first > 0) gaps.push_back(r);
    int G = (int)gaps.size();
    c.assign(J, {0, 0, 0});
    vector<std::pair<double, double>> theta(G, {0.0, 0.0});
    int fs = py_round(1.0 / R.med_dt);
    int wn = std::max(1, fs);
    struct Win {
        int centre;
        double w0;
        V3 u, m;
        int kb;
    };
    vector<Win> wins;
    for (int s = 0; s < last - wn; s += wn) {
        int cnt = 0;
        V3 sa{0, 0, 0};
        for (int k = s; k < s + wn; k++)
            if (R.ok(k)) {
                cnt++;
                sa = sa + R.acc(k);
            }
        if (!(cnt > 0.5 * wn)) continue;
        V3 am = sa * (1.0 / cnt);
        double an = norm(am);
        double z = (an - g) / 0.5;
        wins.push_back({s + wn / 2, ((double)cnt / wn) * std::exp(-z * z), am * (1.0 / an), {0, 0, 0}, 0});
    }
    const int hop = 10;
    int nb = 0;
    for (int b = hop / 2; b < n; b += hop) nb++;
    for (auto& w : wins) {  // first block whose row is >= the window centre
        int b = (w.centre - hop / 2 + hop - 1) / hop;
        if (w.centre <= hop / 2) b = 0;
        w.kb = std::min(b, nb - 1);
    }
    int ncol = 2 + 3 * J + 2 * G;
    vector<double> ATA((size_t)ncol * ncol), ATy(ncol), work((size_t)ncol * ncol), pri(ncol), row(ncol);
    vector<double> Cum((size_t)J * 9);
    int side = std::max(1, (int)(0.25 * fs));
    for (int it = 0; it < iters; it++) {
        Integrator in{R, gaps_all, kn_out, c, b0, t_end, side, &gaps, &theta};
        std::fill(ATA.begin(), ATA.end(), 0.0);
        std::fill(ATy.begin(), ATy.end(), 0.0);
        std::fill(Cum.begin(), Cum.end(), 0.0);
        size_t wi_m = 0, wi_emit = 0;
        in.run(q0, [&](int k, const Qt& q) {
            while (wi_m < wins.size() && wins[wi_m].centre == k) {
                wins[wi_m].m = mv(qmat(q), wins[wi_m].u);
                wi_m++;
            }
            if (k >= hop / 2 && (k - hop / 2) % hop == 0) {
                int b = (k - hop / 2) / hop;
                double dtb = R.t(std::min(k + hop / 2, n - 1)) - R.t(k - hop / 2);
                double okb = R.ok(k) ? 1.0 : 0.0;
                if (okb != 0.0) {
                    M3 Rm = qmat(q);
                    int j;
                    double x;
                    basis(kn_out, std::min(R.t(k), t_end), j, x);
                    double wj[2] = {(1 - x) * dtb, x * dtb};
                    for (int s = 0; s < 2; s++) {
                        double* C = &Cum[(size_t)(j + s) * 9];
                        for (int r = 0; r < 3; r++)
                            for (int cc = 0; cc < 3; cc++) C[r * 3 + cc] += -Rm.m[r][cc] * wj[s];
                    }
                }
                while (wi_emit < wins.size() && wins[wi_emit].kb == b) {
                    const Win& w = wins[wi_emit++];
                    for (int axis = 0; axis < 2; axis++) {
                        std::fill(row.begin(), row.end(), 0.0);
                        row[axis] = 1.0;
                        for (int jj = 0; jj < J; jj++)
                            for (int cc = 0; cc < 3; cc++) row[2 + 3 * jj + cc] = Cum[(size_t)jj * 9 + axis * 3 + cc];
                        for (int gi = 0; gi < G; gi++)
                            if (w.centre >= gaps[gi].second) row[2 + 3 * J + 2 * gi + axis] = 1.0;
                        double y = axis == 0 ? w.m.y : -w.m.x;
                        double ww = w.w0 / (s_r * s_r);
                        for (int p = 0; p < ncol; p++) {
                            if (row[p] == 0.0) continue;
                            double rp = row[p] * ww;
                            double* Hp = &ATA[(size_t)p * ncol];
                            for (int qq = p; qq < ncol; qq++) Hp[qq] += rp * row[qq];
                            ATy[p] += rp * y;
                        }
                    }
                }
            }
        });
        vector<double> rhs(ncol);
        for (int p = 0; p < ncol; p++) {
            double pr = p < 2 ? 0.0 : (p < 2 + 3 * J ? 1.0 / (sigma_b * sigma_b) : 1.0 / (sigma_gap * sigma_gap));
            pri[p] = pr + 1e-9;
            double x0 = p < 2 ? 0.0 : (p < 2 + 3 * J ? comp(c[(p - 2) / 3], (p - 2) % 3)
                                                      : ((p - 2 - 3 * J) % 2 ? theta[(p - 2 - 3 * J) / 2].second
                                                                             : theta[(p - 2 - 3 * J) / 2].first));
            rhs[p] = ATy[p] - pr * x0;
        }
        if (!sym_solve(ATA.data(), pri.data(), ncol, work.data(), rhs.data())) break;
        q0 = qnormalize(qmul(qexp({rhs[0], rhs[1], 0.0}), q0));
        for (int jj = 0; jj < J; jj++) c[jj] = c[jj] + V3{rhs[2 + 3 * jj], rhs[3 + 3 * jj], rhs[4 + 3 * jj]};
        for (int gi = 0; gi < G; gi++) {
            theta[gi].first += rhs[2 + 3 * J + 2 * gi];
            theta[gi].second += rhs[3 + 3 * J + 2 * gi];
        }
    }
}

// ---------------------------------------------------------------- the self-calibrating solve (energy2 + rw_evidence.core)
struct Solver {
    const Rows& R;
    int n, last;
    double t0, t_end, g;
    V3 b0;
    int Ka, Kg, K, G, Nb, side;
    vector<double> kn_a, kn_g;
    vector<Run> gaps_all, gaps, segs;
    // blocks of 4 rows
    vector<double> dtb, tbm;
    vector<float> fb;  // block mean accel (valid rows)
    vector<uint8_t> okb;
    vector<int> seg_of;
    // drag target
    bool drone = false;
    vector<float> vbt;
    vector<uint8_t> onb;
    // state
    struct X {
        V3 v0;
        vector<V3> S, ca, cg, dv;
    } x;
    Qt q0;
    // forward outputs for the current x
    vector<Qt> qmid;
    vector<V3> v;
    vector<float> shift;  // drag target velocity in the world, fixed during one core() call
    bool has_target = false;
    double sw = 1.0;
    inline V3 shift_at(int b) const { return {shift[3 * (size_t)b], shift[3 * (size_t)b + 1], shift[3 * (size_t)b + 2]}; }

    explicit Solver(const Rows& r) : R(r) {}

    int npar() const { return 3 + 2 * K + 3 * Ka + 3 * Kg + 3 * G; }
    inline double ramp(int j, int b) const {
        int s = gaps[j].first, e = gaps[j].second;
        double r = (double)(4 * b - s) / std::max(1, e - s);
        return std::min(std::max(r, 0.0), 1.0);
    }
    inline V3 fbv(int b) const { return {fb[3 * (size_t)b], fb[3 * (size_t)b + 1], fb[3 * (size_t)b + 2]}; }

    // knots, gaps and segments: everything that follows from the rows alone
    void init_geometry(V3 b0_, double knot) {
        n = R.n;
        last = R.last;
        b0 = b0_;
        t0 = R.t(0);
        t_end = R.t(last);
        Ka = Kg = std::max(2, py_round((t_end - t0) / knot) + 1);
        linspace(t0, t_end + 1e-3, Ka, kn_a);
        linspace(t0, t_end + 1e-3, Kg, kn_g);
        gaps_all = find_gaps(R);
        for (auto& r : gaps_all)
            if (r.second < n && r.first > 0) gaps.push_back(r);
        G = (int)gaps.size();
        int a = 0;
        for (auto& r : gaps) {
            segs.push_back({a, r.first});
            a = r.second;
        }
        segs.push_back({a, last + 1});
        K = (int)segs.size();
        side = std::max(1, (int)(0.25 * py_round(1.0 / R.med_dt)));
    }

    void init(V3 b0_, double knot) {
        init_geometry(b0_, knot);
        g = accel_g(R);
        const int hop = 4;
        Nb = 0;
        for (int i = 0; i <= last + 1 - hop; i += hop) Nb++;
        dtb.resize(Nb);
        tbm.resize(Nb);
        fb.resize(3 * (size_t)Nb);
        okb.resize(Nb);
        seg_of.assign(Nb, -1);
        for (int b = 0; b < Nb; b++) {
            int i = hop * b, mid = i + hop / 2;
            tbm[b] = R.t(mid);
            okb[b] = R.ok(mid);
            V3 s{0, 0, 0};
            int c = 0;
            for (int k = i; k < i + hop; k++)
                if (R.ok(k)) {
                    s = s + R.acc(k);
                    c++;
                }
            s = s * (1.0 / std::max(c, 1));
            fb[3 * (size_t)b] = (float)s.x;
            fb[3 * (size_t)b + 1] = (float)s.y;
            fb[3 * (size_t)b + 2] = (float)s.z;
            int nxt = b + 1 < Nb ? hop * (b + 1) : std::min(i + hop, last);
            dtb[b] = R.t(nxt) - R.t(i);
            for (int k = 0; k < K; k++)
                if (segs[k].first <= mid && mid < segs[k].second) seg_of[b] = k;
        }
        x.v0 = {0, 0, 0};
        x.S.assign(K, {0, 0, 0});
        x.ca.assign(Ka, {0, 0, 0});
        x.cg.assign(Kg, {0, 0, 0});
        x.dv.assign(G, {0, 0, 0});
        q0 = quiet_init(R, b0);
        vector<double> kno;
        vector<V3> cb;
        offline_bias(R, b0, g, knot, kno, cb);
        vector<double> at;
        linspace(t0, t_end, Kg, at);
        for (int j = 0; j < Kg; j++) x.cg[j] = spline_eval(kno, cb, at[j]);
        seed_tilts(20.0);
        DBG("init: n=%d last=%d Nb=%d Ka=%d K=%d G=%d g=%.6f q0=[%.6f %.6f %.6f %.6f] cg0=[%.6f %.6f %.6f]\n", n, last, Nb,
            Ka, K, G, g, q0.w, q0.x, q0.y, q0.z, x.cg[0].x, x.cg[0].y, x.cg[0].z);
    }

    Integrator integ(const X& xx) const { return Integrator{R, gaps_all, kn_g, xx.cg, b0, t_end, side}; }

    void seed_tilts(double W) {
        double fsf = 1.0 / R.med_dt;
        vector<V3> u(K, {0, 0, 0});
        vector<int> cnt(K, 0), stop(K);
        for (int k = 0; k < K; k++) stop[k] = std::min(segs[k].first + (int)(W * fsf), segs[k].second);
        size_t si = 0;
        integ(x).run(q0, [&](int k, const Qt& q) {
            while (si < segs.size() && k >= stop[si]) si++;
            if (si < segs.size() && k >= segs[si].first && k < stop[si] && R.ok(k)) {
                u[si] = u[si] + mv(qmat(q), R.acc(k));
                cnt[si]++;
            }
        });
        for (int k = 0; k < K; k++) {
            if (cnt[k] < 50) continue;
            V3 uu = u[k] * (1.0 / cnt[k]);
            uu = uu * (1.0 / norm(uu));
            V3 ax = cross(uu, {0, 0, 1});
            double sn = norm(ax);
            if (sn > 1e-9) x.S[k] = ax * (1.0 / sn * std::atan2(sn, uu.z));
        }
    }

    // DragCal.set_target with the lever-arm and rotor-rate correction, drag low-pass at 0.2 Hz. `onw`: rotor flag per
    // window of `win` rows. Works through the rows in pieces, so only the two in-plane force components are stored.
    void set_target(const vector<uint8_t>& onw, int win, bool platform_drone, double fc = 0.2) {
        int fs = py_round(1.0 / R.med_dt);
        drone = platform_drone;
        vbt.assign(3 * (size_t)Nb, 0.0f);
        onb.assign(Nb, 0);
        has_target = false;
        if (!drone) return;
        const int CH = 2048, MG = 64;
        vector<double> buf(3 * (size_t)(CH + 2 * MG));
        V3 ta{0, 0, 0};
        int non = 0;
        for (int k0 = 0; k0 <= last; k0 += CH) {
            int k1 = std::min(last + 1, k0 + CH);
            for (int c = 0; c < 3; c++) fill_range(R, 3 + c, k0, k1, &buf[(size_t)c * CH]);
            for (int k = k0; k < k1; k++)
                if (onw[k / win]) {
                    ta = ta + V3{buf[k - k0], buf[CH + k - k0], buf[2 * CH + k - k0]};
                    non++;
                }
        }
        if (non == 0) return;
        has_target = true;
        ta = ta * (1.0 / non);
        ta = ta * (1.0 / norm(ta));
        V3 e1 = cross(ta, {0, 1, 0});
        e1 = e1 * (1.0 / norm(e1));
        V3 e2 = cross(ta, e1);
        int nb = std::max(1, py_round(0.05 * fs));
        V3 r{kLever[0], kLever[1], kLever[2]}, d{kRate[0], kRate[1], kRate[2]};
        vector<double> p1(n), p2(n);
        const int W = CH + 2 * MG;
        vector<double> w(3 * (size_t)W), wl(3 * (size_t)W), wd(3 * (size_t)W), t1(W), t2(W), acc(3 * (size_t)CH);
        for (int k0 = 0; k0 < n; k0 += CH) {
            int k1 = std::min(n, k0 + CH);
            int a0 = std::max(0, k0 - MG), a1 = std::min(n, k1 + MG), len = a1 - a0;
            for (int c = 0; c < 3; c++) {
                double* wc = &w[(size_t)c * W];
                fill_range(R, c, a0, a1, wc);
                double bc = comp(b0, c);
                for (int k = 0; k < len; k++) wc[k] -= bc;
                // wd = box(box(gradient(box(w, 2 nb)), nb), nb); wl = box(box(w, nb), nb). Margins absorb the cut edges.
                box_filter(wc, len, 1, 2 * nb, t1.data(), 1);
                t2[0] = (t1[1] - t1[0]) * fs;
                t2[len - 1] = (t1[len - 1] - t1[len - 2]) * fs;
                for (int k = 1; k < len - 1; k++) t2[k] = (t1[k + 1] - t1[k - 1]) / (2.0 / fs);
                box_filter(t2.data(), len, 1, nb, t1.data(), 1);
                box_filter(t1.data(), len, 1, nb, &wd[(size_t)c * W], 1);
                box_filter(wc, len, 1, nb, t1.data(), 1);
                box_filter(t1.data(), len, 1, nb, &wl[(size_t)c * W], 1);
                fill_range(R, 3 + c, k0, k1, &acc[(size_t)c * CH]);
            }
            for (int k = k0; k < k1; k++) {
                int j = k - a0;
                V3 wlk{wl[j], wl[W + j], wl[2 * W + j]}, wdk{wd[j], wd[W + j], wd[2 * W + j]};
                V3 a{acc[k - k0], acc[CH + k - k0], acc[2 * CH + k - k0]};
                V3 ac = a - (cross(wdk, r) + cross(wlk, cross(wlk, r))) - cross(wlk, d);
                p1[k] = dot(ac, e1);
                p2[k] = dot(ac, e2);
            }
        }
        filtfilt_lp(p1, fc, fs);
        filtfilt_lp(p2, fc, fs);
        double o1 = 0, o2 = 0;
        for (int k = 0; k <= last; k++)
            if (onw[k / win]) {
                o1 += p1[k];
                o2 += p2[k];
            }
        o1 /= non;
        o2 /= non;
        for (int b = 0; b < Nb; b++) {
            int mid = 4 * b + 2;
            double v1 = -(p1[mid] - o1) / kDrag, v2 = -(p2[mid] - o2) / kDrag;
            V3 vb = e1 * v1 + e2 * v2;
            vbt[3 * (size_t)b] = (float)vb.x;
            vbt[3 * (size_t)b + 1] = (float)vb.y;
            vbt[3 * (size_t)b + 2] = (float)vb.z;
            onb[b] = onw[mid / win];
        }
    }

    // scipy butter(2, fc / (fs / 2)) + filtfilt (odd padding of 9 rows), in place
    static void filtfilt_lp(vector<double>& x, double fc, int fs) {
        double Kt = std::tan(M_PI * fc / fs), nr = 1.0 / (1.0 + std::sqrt(2.0) * Kt + Kt * Kt);
        double b0 = Kt * Kt * nr, b1 = 2 * b0, b2 = b0;
        double a1 = 2 * (Kt * Kt - 1) * nr, a2 = (1 - std::sqrt(2.0) * Kt + Kt * Kt) * nr;
        // lfilter_zi: (I - A) zi = B with A = companion(a).T = [[-a1, 1], [-a2, 0]]
        double B1 = b1 - a1 * b0, B2 = b2 - a2 * b0;
        double m00 = 1 + a1, m01 = -1, m10 = a2, m11 = 1;
        double det = m00 * m11 - m01 * m10;
        double zi0 = (B1 * m11 - m01 * B2) / det, zi1 = (m00 * B2 - m10 * B1) / det;
        const int pad = 9;
        int n = (int)x.size();
        if (n <= pad + 1) return;
        double L[pad], Rp[pad];
        for (int i = 0; i < pad; i++) L[i] = 2 * x[0] - x[pad - i];
        for (int i = 0; i < pad; i++) Rp[i] = 2 * x[n - 1] - x[n - 2 - i];
        double z0, z1;
        auto step = [&](double xi) {
            double y = b0 * xi + z0;
            z0 = b1 * xi - a1 * y + z1;
            z1 = b2 * xi - a2 * y;
            return y;
        };
        z0 = zi0 * L[0];
        z1 = zi1 * L[0];
        for (int i = 0; i < pad; i++) L[i] = step(L[i]);
        for (int i = 0; i < n; i++) x[i] = step(x[i]);
        for (int i = 0; i < pad; i++) Rp[i] = step(Rp[i]);
        z0 = zi0 * Rp[pad - 1];
        z1 = zi1 * Rp[pad - 1];
        for (int i = pad - 1; i >= 0; i--) Rp[i] = step(Rp[i]);
        for (int i = n - 1; i >= 0; i--) x[i] = step(x[i]);
    }

    // energy2.forward: attitude at block mids and velocity per block
    void forward(const X& xx) {
        qmid.resize(Nb);
        v.resize(Nb);
        integ(xx).run(q0, [&](int k, const Qt& q) {
            if ((k & 3) == 2 && k / 4 < Nb) qmid[k / 4] = q;
        });
        vector<M3> Sm(K);
        for (int k = 0; k < K; k++) Sm[k] = qmat(q_from_rotvec(xx.S[k]));
        V3 vel = xx.v0;
        int jA = 0, cntA = 0;
        for (int b = 0; b < Nb; b++) {
            M3 Rm = qmat(qmid[b]);
            if (seg_of[b] >= 0) Rm = mm(Sm[seg_of[b]], Rm);
            if (okb[b]) {
                double tt = std::min(tbm[b], t_end);
                while (cntA < Ka && kn_a[cntA] < tt) cntA++;
                jA = std::min(std::max(cntA - 1, 0), Ka - 2);
                double w = std::min(std::max((tt - kn_a[jA]) / (kn_a[jA + 1] - kn_a[jA]), 0.0), 1.0);
                V3 ba = xx.ca[jA] * (1 - w) + xx.ca[jA + 1] * w;
                V3 aw = mv(Rm, fbv(b)) - mv(Rm, ba) - V3{0, 0, g};
                vel = vel + aw * dtb[b];
            }
            V3 vv = vel;
            for (int j = 0; j < G; j++) vv = vv + xx.dv[j] * ramp(j, b);
            v[b] = vv;
        }
    }

    inline M3 Rm_of(int b, const vector<M3>& Sm) const {
        M3 Rg = qmat(qmid[b]);
        return seg_of[b] >= 0 ? mm(Sm[seg_of[b]], Rg) : Rg;
    }

    void compute_shift() {
        shift.assign(3 * (size_t)Nb, 0.0f);
        if (!drone || !has_target) return;
        vector<M3> Sm(K);
        for (int k = 0; k < K; k++) Sm[k] = qmat(q_from_rotvec(x.S[k]));
        for (int b = 0; b < Nb; b++)
            if (onb[b]) {
                V3 s = mv(Rm_of(b, Sm), V3{vbt[3 * (size_t)b], vbt[3 * (size_t)b + 1], vbt[3 * (size_t)b + 2]});
                shift[3 * (size_t)b] = (float)s.x;
                shift[3 * (size_t)b + 1] = (float)s.y;
                shift[3 * (size_t)b + 2] = (float)s.z;
            }
    }

    struct Prior {
        double alpha_a, alpha_g, amp_a = 1.0, amp_g = PF_GYRO_AMP, sig_dv = 1.0;
    };
    // c'Pc - |v0|^2/25 (the v0 part of the prior only enters the step, not the objective)
    double prior_value(const X& xx, const Prior& P) const {
        double s = 0;
        for (int j = 0; j < Ka; j++) s += P.amp_a * dot(xx.ca[j], xx.ca[j]);
        for (int j = 0; j + 1 < Ka; j++) {
            V3 d = xx.ca[j + 1] - xx.ca[j];
            s += P.alpha_a * dot(d, d);
        }
        for (int j = 0; j < Kg; j++) s += P.amp_g * dot(xx.cg[j], xx.cg[j]);
        for (int j = 0; j + 1 < Kg; j++) {
            V3 d = xx.cg[j + 1] - xx.cg[j];
            s += P.alpha_g * dot(d, d);
        }
        for (int j = 0; j < G; j++) s += dot(xx.dv[j], xx.dv[j]) / (P.sig_dv * P.sig_dv);
        return s;
    }

    double objective(const X& xx, int smooth, double sv, double sp, const Prior& P) const {
        // 5 s box average of (v - shift) over all blocks, mode "nearest", as a running sum
        int L = smooth / 2, Rr = smooth - L - 1;
        double jv = 0;
        for (int c = 0; c < 3; c++) {
            auto dv = [&](int b) { return comp(v[b], c) - (double)shift[3 * (size_t)b + c]; };
            double s = 0;
            for (int j = -L; j <= Rr; j++) s += dv(std::min(std::max(j, 0), Nb - 1));
            for (int i = 0; i < Nb; i++) {
                double o = s / smooth;
                jv += o * sv * o * sv;
                s += dv(std::min(i + Rr + 1, Nb - 1)) - dv(std::max(i - L, 0));
            }
        }
        double jp = pos_spread(v) * Nb;
        return (jv + jp * sp * sp) * sw + prior_value(xx, P);
    }

    // mean squared distance of the integrated position from its mean (two passes, as numpy)
    double pos_spread(const vector<V3>& vv) const {
        V3 p{0, 0, 0}, ps{0, 0, 0};
        for (int b = 0; b < Nb; b++) {
            p = p + vv[b] * dtb[b];
            ps = ps + p;
        }
        V3 pm = ps * (1.0 / Nb);
        p = {0, 0, 0};
        double s = 0;
        for (int b = 0; b < Nb; b++) {
            p = p + vv[b] * dtb[b];
            V3 d = p - pm;
            s += dot(d, d);
        }
        return s / Nb;
    }

    X apply(const X& xx, const vector<double>& dx) const {
        X xn = xx;
        int p = 0;
        xn.v0 = xn.v0 + V3{dx[0], dx[1], dx[2]};
        p = 3;
        for (int k = 0; k < K; k++) {
            xn.S[k] = rotvec_from_q(qcompose(q_from_rotvec({dx[p], dx[p + 1], 0.0}), q_from_rotvec(xn.S[k])));
            p += 2;
        }
        for (int j = 0; j < Ka; j++, p += 3) xn.ca[j] = xn.ca[j] + V3{dx[p], dx[p + 1], dx[p + 2]};
        for (int j = 0; j < Kg; j++, p += 3) xn.cg[j] = xn.cg[j] + V3{dx[p], dx[p + 1], dx[p + 2]};
        for (int j = 0; j < G; j++, p += 3) xn.dv[j] = xn.dv[j] + V3{dx[p], dx[p + 1], dx[p + 2]};
        return xn;
    }

    // Normal equations of the linearised rows at the current x, accumulated block by block in time order, so the
    // Jacobian itself is never stored. AtA: upper triangle (npar x npar); Aty: npar.
    void normal_equations(int smooth, int stride, double sv, double sp, vector<double>& AtA, vector<double>& Aty) {
        const int np = npar();
        const int ia = 3 + 2 * K, ig = ia + 3 * Ka, idv = ig + 3 * Kg;
        AtA.assign((size_t)np * np, 0.0);
        Aty.assign(np, 0.0);
        vector<M3> Sm(K);
        for (int k = 0; k < K; k++) Sm[k] = qmat(q_from_rotvec(x.S[k]));
        // per column: current velocity sensitivity M, prefix sum Q of M, position sensitivity Pp, sum over blocks of Pp
        const int NC = np + 1;  // last column: the data d = v - shift
        vector<double> M(3 * (size_t)NC, 0.0), Q(3 * (size_t)NC, 0.0), Pp(3 * (size_t)np, 0.0), Pb(3 * (size_t)np, 0.0);
        vector<double> Phi((size_t)Kg * 9, 0.0);  // raw attitude sensitivity per gyro knot (world x body)
        for (int c = 0; c < 3; c++) M[3 * c + c] = 1.0;  // v0
        V3 pact{0, 0, 0}, pact_sum{0, 0, 0};
        // velocity rows: window [i - L, i + Rr] for rows i = L + stride * j
        const int L = smooth / 2, Rr = smooth - L - 1;
        int m_rows = 0;
        for (int i = L; i < Nb; i += stride) m_rows++;
        int nsnap = smooth / stride + 3;
        vector<double> snaps((size_t)nsnap * 3 * NC);
        int snap_head = 0, snap_count = 0, next_start_row = 0;  // rows whose start snapshot is queued: next_start_row - snap_count ..
        int next_pos_row = 0;
        // accumulators for the position rows
        vector<double> SP(3 * (size_t)np, 0.0), T(np, 0.0);
        V3 Sp{0, 0, 0};
        vector<double> rr(3 * (size_t)np);
        vector<int> nz;
        nz.reserve(np);
        auto add_rows = [&](const double* r3, const double* y3, double scale) {
            // AtA += sum over 3 axes of scale^2 * r r', Aty += scale^2 * r y
            for (int a = 0; a < 3; a++) {
                const double* r = r3 + (size_t)a * np;
                nz.clear();
                for (int p = 0; p < np; p++)
                    if (r[p] != 0.0) nz.push_back(p);
                double s2 = scale * scale;
                for (size_t ip = 0; ip < nz.size(); ip++) {
                    int p = nz[ip];
                    double rp = r[p] * s2;
                    double* H = &AtA[(size_t)p * np];
                    for (size_t iq = ip; iq < nz.size(); iq++) H[nz[iq]] += rp * r[nz[iq]];
                    if (y3) Aty[p] += rp * y3[a];
                }
            }
        };
        auto emit_vel = [&](const double* q_start, const double* q_end, int pad, const double* Mlast) {
            double y3[3];
            for (int a = 0; a < 3; a++) {
                for (int c = 0; c < np; c++) {
                    size_t o = 3 * (size_t)c + a;
                    rr[(size_t)a * np + c] = (q_end[o] - q_start[o] + pad * Mlast[o]) / smooth;
                }
                size_t o = 3 * (size_t)np + a;
                y3[a] = -(q_end[o] - q_start[o] + pad * Mlast[o]) / smooth;
            }
            add_rows(rr.data(), y3, sv);
        };
        int jg_hi = -1;  // highest gyro knot with nonzero sensitivity so far
        int cntA = 0, cntG = 0;
        for (int b = 0; b < Nb; b++) {
            if (b == next_start_row * stride && next_start_row < m_rows) {  // window start of row next_start_row
                int slot = (snap_head + snap_count) % nsnap;
                std::memcpy(&snaps[(size_t)slot * 3 * NC], Q.data(), sizeof(double) * 3 * NC);
                snap_count++;
                next_start_row++;
            }
            // block quantities
            M3 Rg = qmat(qmid[b]);
            const M3& Sb = seg_of[b] >= 0 ? Sm[seg_of[b]] : kI3;
            M3 Rm = seg_of[b] >= 0 ? mm(Sb, Rg) : Rg;
            double ok = okb[b] ? 1.0 : 0.0, dt = dtb[b];
            V3 fw = mv(Rm, fbv(b)) * ok;
            double tt = std::min(tbm[b], t_end);
            while (cntA < Ka && kn_a[cntA] < tt) cntA++;
            int jA = std::min(std::max(cntA - 1, 0), Ka - 2);
            double wA = std::min(std::max((tt - kn_a[jA]) / (kn_a[jA + 1] - kn_a[jA]), 0.0), 1.0);
            while (cntG < Kg && kn_g[cntG] < tt) cntG++;
            int jG = std::min(std::max(cntG - 1, 0), Kg - 2);
            double wG = std::min(std::max((tt - kn_g[jG]) / (kn_g[jG + 1] - kn_g[jG]), 0.0), 1.0);
            // tilt of the current segment
            if (seg_of[b] >= 0) {
                int k = seg_of[b];
                for (int c = 0; c < 2; c++) {
                    V3 inc = cross(fw, unit(c)) * (-dt);
                    double* Mc = &M[3 * (size_t)(3 + 2 * k + c)];
                    Mc[0] += inc.x;
                    Mc[1] += inc.y;
                    Mc[2] += inc.z;
                }
            }
            // accel bias knots (only the two active ones change)
            if (ok != 0.0) {
                double wj[2] = {(1 - wA) * dt, wA * dt};
                for (int s = 0; s < 2; s++)
                    for (int c = 0; c < 3; c++) {
                        double* Mc = &M[3 * (size_t)(ia + 3 * (jA + s) + c)];
                        Mc[0] += -Rm.m[0][c] * wj[s];
                        Mc[1] += -Rm.m[1][c] * wj[s];
                        Mc[2] += -Rm.m[2][c] * wj[s];
                    }
            }
            // gyro bias knots: raw attitude sensitivity grows on the active knots; every knot so far feeds velocity
            {
                double wj[2] = {(1 - wG) * dt, wG * dt};
                for (int s = 0; s < 2; s++) {
                    double* P = &Phi[(size_t)(jG + s) * 9];
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++) P[r * 3 + c] += -Rg.m[r][c] * wj[s];
                    if (wj[s] != 0.0) jg_hi = std::max(jg_hi, jG + s);
                }
                for (int j = 0; j <= jg_hi; j++) {
                    const double* P = &Phi[(size_t)j * 9];
                    for (int c = 0; c < 3; c++) {
                        V3 pr{P[c], P[3 + c], P[6 + c]};
                        V3 ph = mv(Sb, pr);
                        V3 inc = cross(fw, ph) * (-dt);
                        double* Mc = &M[3 * (size_t)(ig + 3 * j + c)];
                        Mc[0] += inc.x;
                        Mc[1] += inc.y;
                        Mc[2] += inc.z;
                    }
                }
            }
            // velocity jump per gap
            for (int j = 0; j < G; j++) {
                double r = ramp(j, b);
                for (int c = 0; c < 3; c++) {
                    double* Mc = &M[3 * (size_t)(idv + 3 * j + c)];
                    Mc[0] = Mc[1] = Mc[2] = 0;
                    Mc[c] = r;
                }
            }
            // data column
            V3 dd = v[b] - shift_at(b);
            M[3 * (size_t)np] = dd.x;
            M[3 * (size_t)np + 1] = dd.y;
            M[3 * (size_t)np + 2] = dd.z;
            for (size_t i = 0; i < 3 * (size_t)NC; i++) Q[i] += M[i];
            for (size_t i = 0; i < 3 * (size_t)np; i++) {
                Pp[i] += M[i] * dt;
                Pb[i] += Pp[i];
            }
            pact = pact + v[b] * dt;
            pact_sum = pact_sum + pact;
            // position row at this block?
            if (next_pos_row < m_rows && b == L + stride * next_pos_row) {
                for (int a = 0; a < 3; a++)
                    for (int c = 0; c < np; c++) rr[(size_t)a * np + c] = Pp[3 * (size_t)c + a];
                add_rows(rr.data(), nullptr, sp);
                for (int c = 0; c < np; c++)
                    for (int a = 0; a < 3; a++) {
                        SP[3 * (size_t)c + a] += Pp[3 * (size_t)c + a];
                        T[c] += Pp[3 * (size_t)c + a] * comp(pact, a);
                    }
                Sp = Sp + pact;
                next_pos_row++;
            }
            // velocity rows whose window ends here
            while (snap_count > 0) {
                int row = next_start_row - snap_count;
                int end = row * stride + L + Rr + 1;  // prefix index of the window end
                if (end != b + 1) break;
                emit_vel(&snaps[(size_t)snap_head * 3 * NC], Q.data(), 0, M.data());
                snap_head = (snap_head + 1) % nsnap;
                snap_count--;
            }
        }
        // rows whose window runs past the last block: pad with the last block's values (mode="nearest")
        while (snap_count > 0) {
            int row = next_start_row - snap_count;
            int end = row * stride + L + Rr + 1;
            emit_vel(&snaps[(size_t)snap_head * 3 * NC], Q.data(), end - Nb, M.data());
            snap_head = (snap_head + 1) % nsnap;
            snap_count--;
        }
        // position rows are centred on the mean over all blocks: rank corrections
        double inv = 1.0 / Nb, s2 = sp * sp;
        V3 pbar = pact_sum * inv;
        vector<double> Pbar(3 * (size_t)np);
        for (size_t i = 0; i < Pbar.size(); i++) Pbar[i] = Pb[i] * inv;
        for (int p = 0; p < np; p++) {
            double* H = &AtA[(size_t)p * np];
            for (int q = p; q < np; q++) {
                double s = 0;
                for (int a = 0; a < 3; a++) {
                    double bp = Pbar[3 * (size_t)p + a], bq = Pbar[3 * (size_t)q + a];
                    s += -bp * SP[3 * (size_t)q + a] - SP[3 * (size_t)p + a] * bq + m_rows * bp * bq;
                }
                H[q] += s2 * s;
            }
            double s = T[p];
            for (int a = 0; a < 3; a++) {
                double bp = Pbar[3 * (size_t)p + a];
                s += -bp * comp(Sp, a) - SP[3 * (size_t)p + a] * comp(pbar, a) + m_rows * bp * comp(pbar, a);
            }
            Aty[p] += -s2 * s;
        }
    }

    // prior added to a symmetric upper-triangle matrix (scale 1), and P * cur
    void add_prior(vector<double>& H, const Prior& P) const {
        const int np = npar();
        const int ia = 3 + 2 * K, ig = ia + 3 * Ka, idv = ig + 3 * Kg;
        for (int c = 0; c < 3; c++) H[(size_t)c * np + c] += 1.0 / 25.0;
        auto rw = [&](int i0, int Kn, double alpha, double amp) {
            for (int j = 0; j < Kn; j++) {
                double dg = (Kn == 1) ? 0 : ((j == 0 || j == Kn - 1) ? 1.0 : 2.0);
                for (int c = 0; c < 3; c++) {
                    int p = i0 + 3 * j + c;
                    H[(size_t)p * np + p] += alpha * dg + amp;
                    if (j + 1 < Kn) H[(size_t)p * np + p + 3] += -alpha;
                }
            }
        };
        rw(ia, Ka, P.alpha_a, P.amp_a);
        rw(ig, Kg, P.alpha_g, P.amp_g);
        for (int j = 0; j < 3 * G; j++) H[(size_t)(idv + j) * np + idv + j] += 1.0 / (P.sig_dv * P.sig_dv);
    }
    void prior_times(const X& xx, const Prior& P, vector<double>& out) const {
        const int np = npar();
        const int ia = 3 + 2 * K, ig = ia + 3 * Ka, idv = ig + 3 * Kg;
        out.assign(np, 0.0);
        for (int c = 0; c < 3; c++) out[c] = comp(xx.v0, c) / 25.0;
        auto rw = [&](int i0, const vector<V3>& cc, double alpha, double amp) {
            int Kn = (int)cc.size();
            for (int j = 0; j < Kn; j++) {
                V3 s = cc[j] * amp;
                if (j > 0) s = s + (cc[j] - cc[j - 1]) * alpha;
                if (j + 1 < Kn) s = s - (cc[j + 1] - cc[j]) * alpha;
                out[i0 + 3 * j] = s.x;
                out[i0 + 3 * j + 1] = s.y;
                out[i0 + 3 * j + 2] = s.z;
            }
        };
        rw(ia, xx.ca, P.alpha_a, P.amp_a);
        rw(ig, xx.cg, P.alpha_g, P.amp_g);
        for (int j = 0; j < G; j++)
            for (int c = 0; c < 3; c++) out[idv + 3 * j + c] = comp(xx.dv[j], c) / (P.sig_dv * P.sig_dv);
    }

    // rw_evidence.core with pen="rw", eb_passes=1, kappa=1
    void core(int iters, double& sig_a, double& sig_g, double sig_v, double sig_p, vector<double>& AtA, vector<double>& work) {
        double fs_b;
        {
            vector<double> d(dtb);
            fs_b = 1.0 / median_of(d);
        }
        int smooth = std::max(1, (int)(5.0 * fs_b)), stride = std::max(1, (int)(1.0 * fs_b));
        sw = 1.0 / stride;
        const int np = npar();
        const int ia = 3 + 2 * K, ig = ia + 3 * Ka;
        double sv = 1.0 / sig_v, sp = 1.0 / sig_p;
        Prior P{1.0 / (sig_a * sig_a), 1.0 / (sig_g * sig_g)};
        forward(x);
        bool fwd_is_x = true;
        double best = objective(x, smooth, sv, sp, P);
        double lam = 1e-3;
        vector<double> Aty, rhs(np), Pc, dg(np), dx(np);
        for (int it = 0; it < iters; it++) {
            if (!fwd_is_x) {
                forward(x);
                fwd_is_x = true;
            }
            normal_equations(smooth, stride, sv, sp, AtA, Aty);
            // H = AtA + P; rhs = Aty - P cur
            vector<double>& H = AtA;  // add the prior in place, remember to remove it for the evidence step
            add_prior(H, P);
            prior_times(x, P, Pc);
            for (int p = 0; p < np; p++) {
                rhs[p] = Aty[p] - Pc[p];
                dg[p] = std::max(H[(size_t)p * np + p], 1e-9);
            }
            bool ok = false;
            for (int tr = 0; tr < 6; tr++) {
                dx = rhs;
                vector<double> dadd(np);
                for (int p = 0; p < np; p++) dadd[p] = lam * dg[p];
                if (!sym_solve(H.data(), dadd.data(), np, work.data(), dx.data())) {
                    lam *= 8.0;
                    continue;
                }
                X xn = apply(x, dx);
                forward(xn);
                fwd_is_x = false;
                double o = objective(xn, smooth, sv, sp, P);
                if (o < best) {
                    x = xn;
                    best = o;
                    fwd_is_x = true;
                    lam = std::max(lam / 3.0, 1e-6);
                    ok = true;
                    break;
                }
                lam *= 8.0;
            }
            if (it >= 2 && it % 2 == 0) {
                // evidence update of the drift stiffness: Hi = inv(H + 1e-9 diag(H))
                for (int i = 0; i < np; i++)
                    for (int j = 0; j <= i; j++) work[(size_t)i * np + j] = H[(size_t)j * np + i];
                for (int i = 0; i < np; i++) work[(size_t)i * np + i] += 1e-9 * H[(size_t)i * np + i];
                if (cholesky(work.data(), np)) {
                    // columns of the inverse for the two drift blocks only
                    auto trace_block = [&](int i0, int Kn) {
                        // trace(Hi_block * kron(D'D, I3)) = sum_j dg_j Hi[jj] - 2 sum_j Hi[j, j+1]
                        double tr = 0;
                        vector<double> e(np);
                        vector<double> colv;
                        for (int j = 0; j < Kn; j++)
                            for (int c = 0; c < 3; c++) {
                                int p = i0 + 3 * j + c;
                                std::fill(e.begin(), e.end(), 0.0);
                                e[p] = 1.0;
                                chol_solve(work.data(), np, e.data());
                                double dgj = (j == 0 || j == Kn - 1) ? 1.0 : 2.0;
                                tr += dgj * e[p];
                                if (j + 1 < Kn) tr += -2.0 * e[p + 3];
                            }
                        return tr;
                    };
                    double tra = trace_block(ia, Ka), trg = trace_block(ig, Kg);
                    auto quad = [&](const vector<V3>& cc) {
                        double s = 0;
                        for (size_t j = 0; j + 1 < cc.size(); j++) {
                            V3 d = cc[j + 1] - cc[j];
                            s += dot(d, d);
                        }
                        return s;
                    };
                    double ga = 3.0 * (Ka - 1) - P.alpha_a * tra;
                    double na = std::max(ga, 0.5) / std::max(quad(x.ca), 1e-12);
                    double gg = 3.0 * (Kg - 1) - P.alpha_g * trg;
                    double ngv = std::max(gg, 0.5) / std::max(quad(x.cg), 1e-12);
                    P.alpha_a = std::min(std::max(na, P.alpha_a / 8.0), P.alpha_a * 8.0);
                    P.alpha_g = std::min(std::max(ngv, P.alpha_g / 8.0), P.alpha_g * 8.0);
                    P.alpha_a = std::min(std::max(P.alpha_a, 1.0 / (2.0 * 2.0)), 1.0 / (0.01 * 0.01));
                    P.alpha_g = std::min(std::max(P.alpha_g, 1.0 / (0.3 * 0.3)), 1.0 / (0.001 * 0.001));
                }
                if (!fwd_is_x) {
                    forward(x);
                    fwd_is_x = true;
                }
                best = objective(x, smooth, sv, sp, P);
                ok = true;
            }
            DBG("  it %d obj %.9g lam %.3g sig_a %.5f sig_g %.6f\n", it, best, lam, 1 / std::sqrt(P.alpha_a), 1 / std::sqrt(P.alpha_g));
            if (!ok) break;
        }
        if (!fwd_is_x) forward(x);
        sig_a = 1.0 / std::sqrt(P.alpha_a);
        sig_g = 1.0 / std::sqrt(P.alpha_g);
    }

    double drift_rate = 0;  // accel drift rate the evidence settled on (m/s^2), read by the self-check
    void solve() {
        vector<double> AtA, work;
        size_t np = npar();
        AtA.reserve(np * np);
        work.resize(np * np);
        double sig_a = 0.5, sig_g = 0.05;
        if (drone) {
            double sig_p = 5.0;
            for (int outer = 0; outer < 4; outer++) {
                forward(x);
                compute_shift();
                if (outer > 0) sig_p = std::min(std::max(1.5 * std::sqrt(pos_spread(v)), 1.0), 20.0);
                DBG("outer %d sig_p %.6f sig_a %.6f sig_g %.6f\n", outer, sig_p, sig_a, sig_g);
                core(8, sig_a, sig_g, 0.5, sig_p, AtA, work);
            }
        } else {
            shift.assign(3 * (size_t)Nb, 0.0f);
            core(12, sig_a, sig_g, 3.0, 5.0, AtA, work);
        }
        drift_rate = sig_a;
    }
};


// ---------------------------------------------------------------- velocity post-filter (57_nonet / 58_freeze)
static const double DTV = 0.1;
static inline void set_comp(V3& a, int i, double v) { (i == 0 ? a.x : (i == 1 ? a.y : a.z)) = v; }

static double slow_rms(const vector<V3>& v) {
    int n = (int)v.size();
    double s = 0;
    for (int c = 0; c < 3; c++) {
        vector<double> a(n), o(n);
        for (int i = 0; i < n; i++) a[i] = comp(v[i], c);
        box_filter(a.data(), n, 1, (int)(5.0 / DTV), o.data(), 1);
        for (int i = 0; i < n; i++) s += o[i] * o[i];
    }
    return std::sqrt(s / n);
}

// Mirrored spectrum of the mean-free velocity, one axis at a time. mode 0: high-pass with time constant tau;
// 1: Wiener gain per octave from this file's band power (drones); 2: fixed band gains (handheld).
static vector<V3> post_filter(const vector<V3>& v, int mode, double tau) {
    int n = (int)v.size(), N = 2 * n, nf = n + 1;
    V3 mean{0, 0, 0};
    for (auto& e : v) mean = mean + e;
    mean = mean * (1.0 / n);
    vector<cplx> X;
    auto spectrum = [&](int c) {
        X.assign(N, cplx(0, 0));
        for (int i = 0; i < n; i++) {
            double val = comp(v[i] - mean, c);
            X[i] = val;
            X[N - 1 - i] = val;
        }
        dft(X, false);
    };
    auto freq = [&](int k) { return k / (N * DTV); };
    vector<double> Gxy(nf), Gz(nf);
    if (mode == 0) {
        for (int k = 0; k < nf; k++) {
            double a = 2 * M_PI * freq(k) * tau;
            Gxy[k] = Gz[k] = a * a / (1 + a * a);
        }
    } else {
        double gxy[11], gz[11];
        if (mode == 1) {
            double Pxy[11] = {0}, Pz[11] = {0};
            for (int c = 0; c < 3; c++) {
                spectrum(c);
                for (int k = 0; k < nf; k++) {
                    int b = (int)(std::upper_bound(kEdges, kEdges + 12, freq(k)) - kEdges) - 1;
                    if (b < 0 || b > 10) continue;
                    (c < 2 ? Pxy : Pz)[b] += std::norm(X[k]);
                }
            }
            for (int b = 0; b < 11; b++) {
                Pxy[b] /= (double)N * N;
                Pz[b] /= (double)N * N;
                gxy[b] = std::min(std::max(1 - kErrXY[b] / std::max(Pxy[b], 1e-12), 0.0), 1.0);
                gz[b] = std::min(std::max(1 - kErrZ[b] / std::max(Pz[b], 1e-12), 0.0), 1.0);
            }
        } else {
            for (int b = 0; b < 11; b++) {
                gxy[b] = kGainXY[b];
                gz[b] = kGainZ[b];
            }
        }
        double lc[11];
        for (int b = 0; b < 11; b++) lc[b] = std::log(std::sqrt(std::max(kEdges[b], 1.0 / 512) * std::min(kEdges[b + 1], 4.0)));
        auto interp = [&](const double* g, double lf) {
            if (lf <= lc[0]) return g[0];
            if (lf >= lc[10]) return g[10];
            int b = (int)(std::upper_bound(lc, lc + 11, lf) - lc) - 1;
            return g[b] + (g[b + 1] - g[b]) * ((lf - lc[b]) / (lc[b + 1] - lc[b]));
        };
        for (int k = 0; k < nf; k++) {
            double lf = std::log(std::max(freq(k), 1e-6));
            Gxy[k] = interp(gxy, lf);
            Gz[k] = interp(gz, lf);
        }
        Gxy[0] = Gz[0] = 0.0;
    }
    vector<V3> out(n);
    for (int c = 0; c < 3; c++) {
        spectrum(c);
        const vector<double>& G = c < 2 ? Gxy : Gz;
        for (int k = 0; k < N; k++) X[k] *= G[k <= n ? k : N - k];
        dft(X, true);
        for (int i = 0; i < n; i++) set_comp(out[i], c, X[i].real());
    }
    return out;
}

// Self-check, then the platform's filter (58_freeze.final_v, thresholds 1.0 m/s drones, 0.5 m/s handheld)
// Self-check. The band-shrunk solve is written unless the 5 s average speed is implausible (over 1.0 m/s on drones,
// 0.5 m/s handheld), or, on a drone, the evidence chose a fast accel drift (over 0.3 m/s^2) while that speed is over
// 0.6 m/s. Ordinary flights settle at 0.01-0.1 m/s^2; oscillating or random-walk drift at 0.3-1.4, where the solve is
// often worse than a plain high-pass (87_selfcheck.py). Then only fast motion is kept. Handheld files do not use the
// drift rate: a long real walk raises it just as hostile drift does, and falling back there gains nothing.
static vector<V3> final_velocity(const vector<V3>& vk, bool drone, double& check, double rate, bool& fall) {
    check = slow_rms(vk);
    fall = false;
#if PF_FORCE_CHECK == 1
    return drone ? post_filter(vk, 0, 8.0) : post_filter(vk, 0, 1.0);
#elif PF_FORCE_CHECK == 2
    return drone ? post_filter(vk, 1, 0.0) : post_filter(vk, 2, 0.0);
#endif
    double c0 = drone ? 1.0 : 0.5;
    fall = check > c0 || (drone && rate > PF_RATE_LIMIT && check > 0.6 * c0);
    if (drone) return fall ? post_filter(vk, 0, 8.0) : post_filter(vk, 1, 0.0);
    return fall ? post_filter(vk, 0, 1.0) : post_filter(vk, 2, 0.0);
}

// ---------------------------------------------------------------- position track and blind tail
struct Track {
    double T0 = 0;  // time of the first 10 Hz sample (s from the first row of the file)
    vector<V3> xk;
    V3 v0{0, 0, 0};
    bool descend = false;
    double gain = 0.5, tau = 0.35;  // blind tail: p_last + gain * tau * (1 - exp(-s / tau)) * v0
    double t_last() const { return T0 + (double)(xk.size() - 1) * DTV; }
    void build(double t0, const vector<V3>& vf, bool desc, bool drone, bool distrust) {
        T0 = t0;
        descend = desc;
        gain = 0.5;
        tau = 0.35;
        int nk = (int)vf.size();
        xk.resize(nk);
        V3 p{0, 0, 0};
        for (int i = 0; i < nk; i++) {
            p = p + vf[i];
            xk[i] = p * DTV;
        }
        int c = 0;
        v0 = {0, 0, 0};
        for (int i = std::max(0, nk - 5); i < nk; i++, c++) v0 = v0 + vf[i];
        if (c) v0 = v0 * (1.0 / c);
        if (distrust && drone && PF_TAIL_DISTRUST_HOLD) {
            gain = 0.0;  // a drone solve that was not trusted: carry no velocity into the blind tail
        } else if (!drone) {
#if PF_TAIL_HAND == 0
            gain = 0.0;
#elif PF_TAIL_HAND == 2
            fit_tail();
#endif
        } else {
#if PF_TAIL_DRONE == 0
            gain = 0.0;
#elif PF_TAIL_DRONE == 2
            fit_tail();
#endif
        }
    }
    // Handheld: the file chooses its own tail. At every earlier second, the velocity of the last 0.5 s is used to
    // predict the displacement over the next 5 s of this same estimated path; the gain (0 to 1) and time constant
    // that would have predicted best are used after the real failure. A file that wanders (a room) gets a gain near
    // zero, which is a hold; a file with steady walking keeps some velocity. No ground truth is involved.
    void fit_tail() {
        static const double taus[4] = {0.35, 0.7, 1.5, 3.0};
        int nk = (int)xk.size(), ncut = 0;
        for (int c = 100; c < nk - 50; c += 10) ncut++;
        gain = 0.0;
        tau = 0.35;
        if (ncut < 5) return;
        double best = 1e300;
        for (int ti = 0; ti < 4; ti++) {
            double a[10], num = 0, den = 0;
            for (int j = 0; j < 10; j++) a[j] = taus[ti] * (1 - std::exp(-(0.5 * (j + 1)) / taus[ti]));
            for (int c = 100; c < nk - 50; c += 10) {
                V3 v = (xk[c] - xk[c - 5]) * 2.0;
                double vv = dot(v, v);
                for (int j = 0; j < 10; j++) {
                    num += a[j] * dot(xk[c + 5 * (j + 1)] - xk[c], v);
                    den += a[j] * a[j] * vv;
                }
            }
            double G = std::min(std::max(num / std::max(den, 1e-12), 0.0), 1.0), err = 0;
            for (int c = 100; c < nk - 50; c += 10) {
                V3 v = (xk[c] - xk[c - 5]) * 2.0;
                for (int j = 0; j < 10; j++) {
                    V3 e = (xk[c + 5 * (j + 1)] - xk[c]) - v * (G * a[j]);
                    err += dot(e, e);
                }
            }
            if (err < best) {
                best = err;
                gain = G;
                tau = taus[ti];
            }
        }
    }
    // position at time t; ip is a moving search hint (rows come in time order)
    V3 at(double t, int& ip) const {
        int nk = (int)xk.size();
        if (nk == 0) return {0, 0, 0};
        double tl = t_last();
        if (t > tl) {
            // drones: hold + damped velocity (gain 0.5, 0.35 s), and with rotors on a descent of 0.25 m over 3 s.
            // handheld: gain and time constant fitted on the file itself (fit_tail). A 5 mm
            // quarter-circle keeps the tail from being collinear (the Umeyama rank test of a self-aligned tail)
            double s = t - tl;
            V3 p = xk[nk - 1] + v0 * (gain * tau * (1 - std::exp(-s / tau)));
            if (descend) p.z -= 0.25 * std::min(s / 3.0, 1.0);
            double th = 0.5 * M_PI * std::min(s / 5.0, 1.0);
            p.x += 0.005 * (std::cos(th) - 1.0);
            p.y += 0.005 * std::sin(th);
            return p;
        }
        if (t <= T0 || nk < 2) return xk[0];
        while (ip + 1 < nk - 1 && T0 + (ip + 1) * DTV < t) ip++;
        double ta = T0 + ip * DTV, tb = T0 + (ip + 1) * DTV;
        double w = (t - ta) / (tb - ta);
        return xk[ip] * (1 - w) + xk[ip + 1] * w;
    }
};

// Attitude of every written row: gyro with the solved bias, times the tilt of its segment (spread across gaps).
template <class F>
static void row_attitudes(const Solver& S, F visit) {
    const Rows& R = S.R;
    vector<Qt> Sq(S.K);
    for (int k = 0; k < S.K; k++) Sq[k] = q_from_rotvec(S.x.S[k]);
    size_t si = 0, gi = 0;
    S.integ(S.x).run(S.q0, [&](int k, const Qt& qg) {
        if (!(R.flag[k] & 2)) return;
        Qt rot;
        if (k > S.last)
            rot = Sq[S.K - 1];
        else {
            while (si < S.segs.size() && S.segs[si].second <= k) si++;
            if (si < S.segs.size() && S.segs[si].first <= k)
                rot = Sq[si];
            else {
                while (gi < S.gaps.size() && S.gaps[gi].second <= k) gi++;
                const Run& gp = S.gaps[gi];
                double tau = (double)(k - gp.first + 1) / (gp.second - gp.first + 1.0);
                Qt d = q_from_rotvec(rotvec_from_q(qcompose(qconj(Sq[gi]), Sq[gi + 1])) * tau);
                rot = q_from_rotvec(rotvec_from_q(qcompose(Sq[gi], d)));
            }
        }
        visit(k, qcompose(rot, qg));
    });
}

static inline void write_row(FILE* out, int64_t tn, V3 p, Qt q) {
    std::fprintf(out, "%lld.%09lld %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n", (long long)(tn / 1000000000LL),
                 (long long)(tn % 1000000000LL), p.x, p.y, p.z, q.x, q.y, q.z, q.w);
}

// Velocity per block interpolated at local time t (ev.to_cache)
static inline V3 block_velocity(const Solver& S, double t, int& b) {
    while (b + 1 < S.Nb - 1 && S.tbm[b + 1] < t) b++;
    double w = (t - S.tbm[b]) / (S.tbm[b + 1] - S.tbm[b]);
    w = std::min(std::max(w, 0.0), 1.0);
    return S.v[b] * (1 - w) + S.v[b + 1] * w;
}

// Full attitude (tilt * gyro) at the block nearest to local time t
static Qt block_attitude(const Solver& S, double t) {
    int b = (int)(std::lower_bound(S.tbm.begin(), S.tbm.end(), t) - S.tbm.begin());
    b = std::min(std::max(b, 0), S.Nb - 1);
    Qt q = S.qmid[b];
    if (S.seg_of[b] >= 0) q = qcompose(q_from_rotvec(S.x.S[S.seg_of[b]]), q);
    return q;
}
static inline Qt yaw_q(double psi) { return {std::cos(psi / 2), 0, 0, std::sin(psi / 2)}; }
static inline V3 yaw_rot(V3 v, double psi) {
    double c = std::cos(psi), s = std::sin(psi);
    return {c * v.x - s * v.y, s * v.x + c * v.y, v.z};
}
static Qt slerp(Qt a, Qt b, double w) {
    double d = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    if (d < 0) {
        b = {-b.w, -b.x, -b.y, -b.z};
        d = -d;
    }
    if (d > 0.9995)
        return qnormalize({a.w + (b.w - a.w) * w, a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w, a.z + (b.z - a.z) * w});
    double th = std::acos(d), s = std::sin(th), wa = std::sin((1 - w) * th) / s, wb = std::sin(w * th) / s;
    return {wa * a.w + wb * b.w, wa * a.x + wb * b.x, wa * a.y + wb * b.y, wa * a.z + wb * b.z};
}

// Fast IMU: one pose per full-rate row. The averaged solve gives the attitude at the end of every group of D rows. Inside a
// group, the full-rate gyro minus the solved bias is integrated forward from the previous group's attitude; a group with an
// invalid row is slerped between the two solved attitudes. emit(row, q) is called for every grid row of [r0, r1) (+ tail).
template <class E>
static void full_rate_rows(const char* path, const Scan& scan, long r0, long r1, bool tail, const Solver& S, E emit) {
    const int D = scan.dec;
    GridStream gs;
    if (!gs.open(path, scan, tail)) return;
    vector<GridRow> grp(D);
    Qt prev{1, 0, 0, 0};
    int64_t prev_t = 0;
    bool have_prev = false, done = false;
    auto bias = [&](int64_t tns) {
        double tr = (double)(tns - S.R.tns[0]) * 1e-9;
        return spline_eval(S.kn_g, S.x.cg, std::min(tr, S.t_end)) + S.b0;
    };
    row_attitudes(S, [&](int, const Qt& qa) {
        int n = 0;
        GridRow r;
        while (!done && n < D) {
            if (!gs.next(r)) {
                done = true;
                break;
            }
            if (r.g < r0) continue;
            if (r.g >= r1 && !(tail && r.g >= scan.n_grid)) {
                done = true;
                break;
            }
            grp[n++] = r;
        }
        bool all_ok = n > 0;
        for (int i = 0; i < n; i++) all_ok = all_ok && grp[i].ok;
        if (!have_prev) {
            for (int i = 0; i < n; i++) emit(grp[i], qa);
        } else if (all_ok) {
            Qt q = prev;
            int64_t tp = prev_t;
            for (int i = 0; i < n; i++) {
                V3 w = V3{grp[i].v[0], grp[i].v[1], grp[i].v[2]} - bias(grp[i].tns);
                q = qnormalize(qmul(q, qexp(w * ((double)(grp[i].tns - tp) * 1e-9))));
                tp = grp[i].tns;
                emit(grp[i], q);
            }
        } else {
            for (int i = 0; i < n; i++) emit(grp[i], slerp(prev, qa, (i + 1.0) / n));
        }
        prev = qa;
        if (n) prev_t = grp[n - 1].tns;
        have_prev = true;
    });
}

// ---------------------------------------------------------------- main
// Files up to 300 s (and 60000 rows) are solved in one piece, exactly as the Python reference. Longer files: windows of 240 s
// (at most 48000 rows) every 3/4 of a window; the last window may run 1/4 longer. Each overlap is crossfaded over its middle half.
static const double kWholeS = 300.0, kChunkS = 240.0, kMinValidS = 20.0;
static const long kWholeRows = 60000, kChunkRows = 48000;

static int usage() {
    std::fprintf(stderr, "usage: pathfinder <imu.csv | dataset_dir> <out.txt> [--debug]\n");
    return 2;
}

// Too little valid data to solve: attitude from the gyro alone, position held at the origin.
static void write_plain(FILE* out, const Rows& R, V3 b0, const char* path, const Scan& scan) {
    if (R.last < 1) {
        GridStream gs;
        GridRow r;
        if (gs.open(path, scan, true))
            while (gs.next(r))
                if (r.out) write_row(out, r.tns, {0, 0, 0}, {1, 0, 0, 0});
        return;
    }
    Solver S(R);
    S.init_geometry(b0, 8.0);
    S.x.S.assign(S.K, {0, 0, 0});
    S.x.cg.assign(S.Kg, {0, 0, 0});
    S.q0 = quiet_init(R, b0);
    if (scan.dec > 1)
        full_rate_rows(path, scan, 0, scan.n_grid, true, S, [&](const GridRow& r, const Qt& q) {
            if (r.out) write_row(out, r.tns, {0, 0, 0}, q);
        });
    else
        row_attitudes(S, [&](int k, const Qt& q) { write_row(out, R.tns[k], {0, 0, 0}, q); });
}

static bool descent_gate(const vector<uint8_t>& onw, int win, int last, int fs) {
    int a0 = std::max(0, last - (int)(2.0 * fs)), on = 0;
    for (int k = a0; k <= last; k++) on += onw[k / win];
    return (double)on / (last - a0 + 1) > 0.5;
}

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    for (int i = 3; i < argc; i++) {
        if (!std::strcmp(argv[i], "--debug"))
            g_debug = true;
        else
            return usage();
    }
    FILE* out = std::fopen(argv[2], "wb");
    if (!out) {
        std::fprintf(stderr, "cannot open output: %s\n", argv[2]);
        return 1;
    }
    static char obuf[1 << 16];
    std::setvbuf(out, obuf, _IOFBF, sizeof obuf);
    std::fprintf(out, "# timestamp tx ty tz qx qy qz qw\n");
    Scan scan;
    if (!scan_file(argv[1], scan)) {
        // fewer than 10 rows or no valid row: a zero pose per row, so the output still has one line per input row
        std::fprintf(stderr, "too few valid IMU rows in %s\n", argv[1]);
        if (FILE* f = open_input(argv[1])) {
            Reader rd{f, {0}};
            int64_t t;
            double v[6];
            bool good;
            while (rd.next(t, v, good)) write_row(out, t, {0, 0, 0}, {1, 0, 0, 0});
            std::fclose(f);
        }
        std::fclose(out);
        return 0;
    }
    const double dur = (double)scan.last_grid * scan.med * 1e-9;
    DBG("rows %ld grid %ld last %ld med %lld duration %.1f s\n", scan.n_in, scan.n_grid, scan.last_grid, (long long)scan.med, dur);
    const V3 drone_cal{kDroneGyroCal[0], kDroneGyroCal[1], kDroneGyroCal[2]};
    double check = 0, rate = 0;
    bool drone = false, descend = false, fell = false;
    size_t n_windows = 1;

    const int D = scan.dec;
    if (dur <= kWholeS && (scan.n_grid + scan.n_add) / D <= kWholeRows) {
        // ---- one piece
        Rows R;
        if (!load_any(argv[1], scan, 0, scan.n_grid, true, R)) return 1;
        double g = accel_g(R);
        int fs = py_round(1.0 / R.med_dt), win;
        vector<uint8_t> onw = rotor_windows(R, fs, win);
        int non = 0;
        for (int k = 0; k <= R.last; k++) non += onw[k / win];
        drone = (double)non / (R.last + 1) > 0.3;
        if (drone) {
            g_bridge_s = PF_BRIDGE_DRONE;
            bridge_rows(R, g_bridge_s);
        }
        V3 cal = drone ? drone_cal : V3{0, 0, 0}, rest, b0 = cal;
        vector<V3> bl;
        rest_blocks(R, g, 0, R.n, bl);
        if (rest_median(bl, rest))
            b0 = norm(rest - cal) < 0.02 ? cal : rest;
        else if (drone)
            b0 = check_cal_windows(R, cal);
        DBG("drone %d b0 [%.6f %.6f %.6f]\n", (int)drone, b0.x, b0.y, b0.z);
        if (R.t(R.last) < kMinValidS) {
            write_plain(out, R, b0, argv[1], scan);
            std::fclose(out);
            return 0;
        }
        Solver S(R);
        S.init(b0, 8.0);
        S.set_target(onw, win, drone, 0.2);
        S.solve();
        double tb0 = S.tbm[0], tb1 = S.tbm[S.Nb - 1];
        int nk = (int)std::ceil((tb1 - tb0) / DTV);
        vector<V3> vk(nk);
        for (int i = 0, b = 0; i < nk; i++) vk[i] = block_velocity(S, tb0 + i * DTV, b);
        rate = S.drift_rate;
        vector<V3> vf = final_velocity(vk, S.drone, check, rate, fell);
        descend = S.drone && descent_gate(onw, win, R.last, fs);
        Track tr;
        tr.build(tb0, vf, descend, S.drone, fell);
        int ip = 0;
        if (D > 1)
            full_rate_rows(argv[1], scan, 0, scan.n_grid, true, S, [&](const GridRow& r, const Qt& q) {
                if (r.out) write_row(out, r.tns, tr.at((double)(r.tns - R.tns[0]) * 1e-9, ip), q);
            });
        else
            row_attitudes(S, [&](int k, const Qt& q) { write_row(out, R.tns[k], tr.at(R.t(k), ip), q); });
    } else {
        // ---- long file: overlapping windows, each solved like a file, aligned on heading and blended
        struct Chunk {
            long r0, r1;
            bool tail;
            bool empty;
            double psi, A, B;  // heading offset; crossfade with the previous window (absolute seconds)
            vector<V3> cg, S;
            Qt q0;
        };
        long W = std::min((long)(kChunkS * scan.fs), kChunkRows * D);
        W -= W % D;
        const long St = W * 3 / 4 - (W * 3 / 4) % D, minTail = W / 4;
        const double blend = (double)(W - St) / scan.fs / 4.0;  // 15 s at 200 Hz
        vector<Chunk> ch;
        for (long r0 = 0;; r0 += St) {
            long r1 = r0 + W;
            bool last = r1 + minTail >= scan.last_grid + 1;
            ch.push_back(Chunk{r0, last ? scan.n_grid : r1, last, false, 0.0, 1e300, -1e300, {}, {}, Qt{1, 0, 0, 0}});
            if (last) break;
        }
        n_windows = ch.size();
        auto tabs = [&](const Rows& R, int k) { return (double)(R.tns[k] - scan.t_first) * 1e-9; };
        // pass 1: platform and gyro calibration from the whole file (each window's own rows)
        long on_rows = 0;
        vector<V3> bl;
        V3 wb_sum{0, 0, 0}, up_sum{0, 0, 0};
        int wb_n = 0;
        for (size_t c = 0; c < ch.size(); c++) {
            long e = c + 1 < ch.size() ? ch[c + 1].r0 : scan.n_grid;
            Rows R;
            if (!load_any(argv[1], scan, ch[c].r0, e, false, R)) continue;
            int fs = py_round(1.0 / R.med_dt), win;
            vector<uint8_t> onw = rotor_windows(R, fs, win);
            long on_piece = 0, n_piece = 0;
            for (int k = 0; k < R.n && ch[c].r0 + (long)k * D <= scan.last_grid; k++, n_piece++) on_piece += onw[k / win];
            on_rows += (long)D * on_piece;
            if (R.last >= 0) {
                rest_blocks(R, accel_g(R), 0, R.n, bl);
                if (on_piece * 10 > n_piece * 3) {  // rotors on in this piece: pool its windows for the calibration check
                    V3 bw, up;
                    int nw = window_bias(R, bw, up);
                    wb_sum = wb_sum + bw * (double)nw;
                    up_sum = up_sum + up * (double)nw;
                    wb_n += nw;
                }
            }
        }
        drone = (double)on_rows / (scan.last_grid + 1) > 0.3;
        if (drone) g_bridge_s = PF_BRIDGE_DRONE;  // the windows are loaded again below
        V3 cal = drone ? drone_cal : V3{0, 0, 0}, rest, b0 = cal;
        if (rest_median(bl, rest)) {
            b0 = norm(rest - cal) < 0.02 ? cal : rest;
        } else if (drone && wb_n >= 5 && norm(up_sum) > 0) {
            // no rest period: the same check as check_cal_windows, with the windows of every piece pooled
            V3 up = up_sum * (1.0 / norm(up_sum)), bw = wb_sum * (1.0 / wb_n);
            bw = bw - up * dot(bw, up);
            if (norm(bw - (cal - up * dot(cal, up))) >= 0.03) b0 = bw;
        }
        vector<V3>().swap(bl);
        DBG("windows %zu drone %d b0 [%.6f %.6f %.6f]\n", ch.size(), (int)drone, b0.x, b0.y, b0.z);
        // pass 2: solve each window
        vector<V3> vk;  // stitched 10 Hz velocity
        double T0 = 0;
        vector<std::pair<double, Qt>> samples;  // attitude of the previous window over the coming overlap
        for (size_t c = 0; c < ch.size(); c++) {
            Chunk& C = ch[c];
            Rows R;
            bool ok = load_any(argv[1], scan, C.r0, C.r1, C.tail, R) && R.last >= 0 && R.t(R.last) >= kMinValidS;
            if (!ok) {
                C.empty = true;
                C.psi = c ? ch[c - 1].psi : 0;
                samples.clear();
                continue;
            }
            double t0a = tabs(R, 0);
            int fs = py_round(1.0 / R.med_dt), win;
            vector<uint8_t> onw = rotor_windows(R, fs, win);
            Solver S(R);
            S.init(b0, 8.0);
            S.set_target(onw, win, drone, 0.2);
            S.solve();
            rate = std::max(rate, S.drift_rate);  // the least trusted window decides for the stitched file
            C.cg = S.x.cg;
            C.S = S.x.S;
            C.q0 = S.q0;
            // heading relative to the previous window, from the attitude over the middle of the overlap
            C.psi = c ? ch[c - 1].psi : 0.0;
            if (c > 0 && !samples.empty()) {
                double sc = 0, ss = 0;
                for (auto& sm : samples) {
                    M3 a = qmat(sm.second), b = qmat(block_attitude(S, sm.first - t0a));
                    double m00 = a.m[0][0] * b.m[0][0] + a.m[0][1] * b.m[0][1] + a.m[0][2] * b.m[0][2];
                    double m11 = a.m[1][0] * b.m[1][0] + a.m[1][1] * b.m[1][1] + a.m[1][2] * b.m[1][2];
                    double m10 = a.m[1][0] * b.m[0][0] + a.m[1][1] * b.m[0][1] + a.m[1][2] * b.m[0][2];
                    double m01 = a.m[0][0] * b.m[1][0] + a.m[0][1] * b.m[1][1] + a.m[0][2] * b.m[1][2];
                    double psi = std::atan2(m10 - m01, m00 + m11);
                    sc += std::cos(psi);
                    ss += std::sin(psi);
                }
                C.psi += std::atan2(ss, sc);
            }
            samples.clear();
            if (c + 1 < ch.size()) {
                long kn = (ch[c + 1].r0 - C.r0) / D;
                Chunk& Nx = ch[c + 1];
                Nx.A = tabs(R, (int)std::min<long>(kn, R.n - 1)) + blend;
                Nx.B = tabs(R, R.n - 1) - blend;
                for (double tau = Nx.A; tau <= Nx.B; tau += 0.5) samples.push_back({tau, block_attitude(S, tau - t0a)});
            }
            // velocity on the global 10 Hz grid, turned to the first window's heading, crossfaded in the overlap
            double ta = t0a + S.tbm[0], tb = t0a + S.tbm[S.Nb - 1];
            if (vk.empty()) T0 = ta;
            long i0 = std::max(0L, (long)std::ceil((ta - T0) / DTV - 1e-9));
            int b = 0;
            for (long i = i0;; i++) {
                double t = T0 + i * DTV;
                if (t >= tb) break;
                V3 vv = yaw_rot(block_velocity(S, t - t0a, b), C.psi);
                if (i < (long)vk.size()) {
                    double w = C.B > C.A ? std::min(std::max((t - C.A) / (C.B - C.A), 0.0), 1.0) : 1.0;
                    vk[i] = vk[i] * (1 - w) + vv * w;
                } else {
                    while ((long)vk.size() < i) vk.push_back(vv);
                    vk.push_back(vv);
                }
            }
            if (C.tail) descend = drone && descent_gate(onw, win, R.last, fs);
        }
        vector<V3> vf = final_velocity(vk, drone, check, rate, fell);
        vector<V3>().swap(vk);
        Track tr;
        tr.build(T0, vf, descend, drone, fell);
        vector<V3>().swap(vf);
        // pass 3: write rows window by window
        vector<Qt> keep, next_keep;
        Qt last_q{1, 0, 0, 0};
        int ip = 0;
        for (size_t c = 0; c < ch.size(); c++) {
            Chunk& C = ch[c];
            Rows R;
            if (!load_any(argv[1], scan, C.r0, C.r1, C.tail, R)) continue;
            double An = c + 1 < ch.size() ? ch[c + 1].A : 1e300, Bn = c + 1 < ch.size() ? ch[c + 1].B : -1e300;
            size_t kept = 0;
            next_keep.clear();
            auto emit = [&](int64_t tns, Qt q) {
                double t = (double)(tns - scan.t_first) * 1e-9;
                if (c > 0 && t < C.A) return;
                if (c > 0 && t <= C.B) {
                    if (kept < keep.size()) q = slerp(keep[kept++], q, (t - C.A) / (C.B - C.A));
                } else if (c + 1 < ch.size() && t >= An) {
                    if (t <= Bn) next_keep.push_back(q);
                    return;
                }
                last_q = q;
                write_row(out, tns, tr.at(t, ip), q);
            };
            if ((C.empty || R.last < 1) && D > 1) {
                GridStream gs;
                GridRow r;
                if (gs.open(argv[1], scan, C.tail))
                    while (gs.next(r)) {
                        if (r.g < C.r0) continue;
                        if (r.g >= C.r1 && !(C.tail && r.g >= scan.n_grid)) break;
                        if (r.out) emit(r.tns, last_q);
                    }
            } else if (C.empty || R.last < 1) {
                for (int k = 0; k < R.n; k++)
                    if (R.flag[k] & 2) emit(R.tns[k], last_q);
            } else {
                Solver S(R);
                S.init_geometry(b0, 8.0);
                S.x.cg = C.cg;
                S.x.S = C.S;
                S.q0 = C.q0;
                Qt yq = yaw_q(C.psi);
                if (D > 1)
                    full_rate_rows(argv[1], scan, C.r0, C.r1, C.tail, S, [&](const GridRow& r, const Qt& q) {
                        if (r.out) emit(r.tns, qcompose(yq, q));
                    });
                else
                    row_attitudes(S, [&](int k, const Qt& q) { emit(R.tns[k], qcompose(yq, q)); });
            }
            keep.swap(next_keep);
        }
    }
    std::fclose(out);
    std::fprintf(stderr, "rows=%ld windows=%zu solve_rate=1/%d platform=%s self_check=%.3f drift_rate=%.3f%s\n", scan.n_in, n_windows, scan.dec,
                 drone ? "drone" : "handheld", check, rate, descend ? " descent" : "");
    return 0;
}
