// Path Finder baseline: IMU-only 6-DoF trajectory from a EuRoC-format CSV.
//
// This is the BASELINE, not the final entry. It has no learned models yet:
//   - orientation: gyro integration, gyro bias from the quietest second, weak tilt correction
//   - position:    band-limited double integration, so it keeps fast motion and stays bounded
//   - gaps/blind:  coast on a decaying turn rate and a decaying velocity
//
// Memory is fixed: rows are streamed from disk and never stored, so peak RAM does not grow
// with the length of the file.
//
// Usage: pathfinder <imu.csv | dataset_dir> <out.txt> [options]   (run with no args for options)

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------- small maths helpers
struct V3 { double x, y, z; };
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator*(V3 a, double s) { return {a.x * s, a.y * s, a.z * s}; }
static double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static double norm(V3 a) { return std::sqrt(dot(a, a)); }

// Unit quaternion q_WB: rotates a vector from the body frame into the world frame.
struct Q { double w, x, y, z; };
static Q mul(Q a, Q b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}
static Q conj(Q q) { return {q.w, -q.x, -q.y, -q.z}; }
static Q normalized(Q q) {
    double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    return {q.w / n, q.x / n, q.y / n, q.z / n};
}
static V3 rotate(Q q, V3 v) {  // q * v * q^-1, written out for speed
    V3 u{q.x, q.y, q.z};
    V3 t = cross(u, v) * 2.0;
    return v + t * q.w + cross(u, t);
}
static Q exp_rotvec(V3 r) {  // rotation vector (radians) to quaternion
    double a = norm(r);
    if (a < 1e-12) return normalized({1.0, r.x * 0.5, r.y * 0.5, r.z * 0.5});
    double s = std::sin(a * 0.5) / a;
    return {std::cos(a * 0.5), r.x * s, r.y * s, r.z * s};
}
// Rotation that takes unit vector a onto unit vector b.
static Q from_two_vectors(V3 a, V3 b) {
    double d = dot(a, b);
    if (d < -0.999999) return {0.0, 1.0, 0.0, 0.0};  // opposite: any perpendicular axis works
    V3 c = cross(a, b);
    return normalized({1.0 + d, c.x, c.y, c.z});
}

// ---------------------------------------------------------------- CSV streaming
struct Row {
    int64_t t_ns;
    V3 gyro, accel;
    bool valid;  // false: NaN, all-zero, frozen or physically impossible sample
};

static double g_prev[6];
static bool g_have_prev = false;

// Reads the next data row. Returns false at end of file. Header lines and unparseable lines
// are skipped. A row with a good timestamp but bad values is still returned, marked invalid,
// because the spec wants one pose per IMU frame.
static bool next_row(FILE* f, Row& r) {
    char line[512];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char* p = line;
        char* end = nullptr;
        long long t = std::strtoll(p, &end, 10);
        if (end == p) continue;
        r.t_ns = t;
        double v[6];
        bool ok = true;
        p = end;
        for (int i = 0; i < 6; i++) {
            while (*p == ',' || *p == ' ') p++;
            v[i] = std::strtod(p, &end);
            if (end == p) { ok = false; break; }
            p = end;
        }
        if (ok) {
            bool zero = true, same = g_have_prev;
            for (int i = 0; i < 6; i++) {
                if (!std::isfinite(v[i])) ok = false;
                if (v[i] != 0.0) zero = false;
                if (v[i] != g_prev[i]) same = false;
            }
            if (zero || same) ok = false;
            std::memcpy(g_prev, v, sizeof v);
            g_have_prev = true;
        }
        r.gyro = {ok ? v[0] : 0.0, ok ? v[1] : 0.0, ok ? v[2] : 0.0};
        r.accel = {ok ? v[3] : 0.0, ok ? v[4] : 0.0, ok ? v[5] : 0.0};
        if (ok && (norm(r.gyro) > 35.0 || norm(r.accel) > 200.0)) ok = false;  // beyond sensor range
        r.valid = ok;
        return true;
    }
    return false;
}

static void rewind_rows(FILE* f) {
    std::rewind(f);
    g_have_prev = false;
}

// ---------------------------------------------------------------- pass 1: survey the file
struct Survey {
    double dt_nominal = 0.005;
    int64_t dt_nominal_ns = 5000000;
    V3 bias{0, 0, 0}, gravity_body{0, 0, 9.81};
    double g = 9.81;
    int64_t quiet_t_ns = 0;
    bool found_quiet = false;
    long rows = 0, invalid = 0;
    double quiet_gyro_std = 0;
    bool stationary = false;
};

// Per-axis gyro std (rad/s) below which we believe a block is truly still. Sensor noise alone is
// about 0.0024 on both datasets. A handheld "quiet" second is well above this, and its mean is
// real motion, so subtracting it as bias makes the heading drift badly.
static const double kStillGyroStd = 0.01;

// Finds the sample period (median dt) and the quietest one-second block. In that block the
// mean gyro is our bias estimate and the mean accel points along gravity. We search the whole
// file because a flight may not begin stationary and early rows may be moving.
static Survey survey(FILE* f) {
    Survey s;
    static int dt_hist[1000];  // 0.1 ms bins up to 100 ms
    static int64_t dt_sum[1000];
    Row r;
    int64_t last_t = -1, block_start = -1;
    V3 sw{0, 0, 0}, sa{0, 0, 0};
    double sww = 0, saa = 0;
    long n = 0, expected = 0;
    double best = 1e300;
    while (next_row(f, r)) {
        s.rows++;
        if (!r.valid) s.invalid++;
        if (last_t >= 0) {
            int64_t dt = r.t_ns - last_t;
            int bin = (int)(dt / 100000);
            if (dt > 0 && bin < 1000) { dt_hist[bin]++; dt_sum[bin] += dt; }
        }
        last_t = r.t_ns;
        if (block_start < 0) block_start = r.t_ns;
        expected++;
        if (r.t_ns - block_start >= 1000000000LL) {
            // Close this block. Require most samples present, so a gap cannot look "quiet".
            if (n >= 50 && n >= expected * 8 / 10) {
                V3 mw = sw * (1.0 / n), ma = sa * (1.0 / n);
                double var_w = sww / n - dot(mw, mw), var_a = saa / n - dot(ma, ma);
                double off_g = norm(ma) - 9.81;
                double score = var_w + 0.01 * var_a + 0.01 * off_g * off_g;
                if (score < best) {
                    best = score;
                    s.bias = mw;
                    s.gravity_body = ma;
                    s.g = norm(ma);
                    s.quiet_t_ns = block_start + 500000000LL;
                    s.found_quiet = true;
                    s.quiet_gyro_std = std::sqrt(std::fmax(var_w, 0.0) / 3.0);
                }
            }
            block_start = r.t_ns;
            sw = sa = {0, 0, 0};
            sww = saa = 0;
            n = expected = 0;
        }
        if (r.valid) {
            sw = sw + r.gyro;
            sa = sa + r.accel;
            sww += dot(r.gyro, r.gyro);
            saa += dot(r.accel, r.accel);
            n++;
        }
    }
    long total = 0, seen = 0;
    for (int c : dt_hist) total += c;
    for (int i = 0; i < 1000; i++) {
        seen += dt_hist[i];
        if (seen * 2 >= total && total > 0) {
            s.dt_nominal_ns = dt_sum[i] / dt_hist[i];  // average dt inside the median bin
            break;
        }
    }
    s.dt_nominal = s.dt_nominal_ns * 1e-9;
    s.stationary = s.found_quiet && s.quiet_gyro_std < kStillGyroStd;
    if (!s.stationary) s.bias = {0, 0, 0};  // the deck says inputs are already compensated
    return s;
}

// ---------------------------------------------------------------- pass 2: starting attitude
// Gravity tells us the attitude at the quiet block, not at the first row. So integrate the
// gyro from row 0 up to the quiet block, then undo that rotation to get the attitude at row 0.
static Q initial_attitude(FILE* f, const Survey& s) {
    Q q_up = from_two_vectors(s.gravity_body * (1.0 / norm(s.gravity_body)), {0, 0, 1});
    Q q_rel{1, 0, 0, 0};
    Row r;
    int64_t last_t = -1;
    while (s.found_quiet && next_row(f, r) && r.t_ns <= s.quiet_t_ns) {
        if (last_t >= 0 && r.valid) {
            double dt = (r.t_ns - last_t) * 1e-9;
            if (dt > 0) q_rel = normalized(mul(q_rel, exp_rotvec((r.gyro - s.bias) * dt)));
        }
        last_t = r.t_ns;
    }
    return normalized(mul(q_up, conj(q_rel)));
}

// ---------------------------------------------------------------- the estimator
struct Params {
    double tau_v = 1.0;        // s, velocity leak. Larger keeps more motion but drifts more.
    double tau_p = 4.0;        // s, position leak back to the start point.
    double kp = 0.2;           // tilt-correction gain. Steady tilt error is about bias / kp.
    double boost = 2.0;        // extra tilt gain right after a gap, so "up" recovers quickly.
    double tau_a = 3.0;        // s, slow mean removed from world accel, before integrating.
    double tau_rate = 0.5;     // s, how fast the turn rate fades while coasting.
    double tau_coast = 0.75;   // s, how fast velocity fades while coasting.
    double blind_s = 5.0;      // s of forecast to end with after the IMU dies.
    bool fill_gaps = false;    // also write poses for rows missing from the input.
};

struct State {
    Q q{1, 0, 0, 0};
    V3 v{0, 0, 0}, p{0, 0, 0};
    V3 rate_lp{0, 0, 0};  // smoothed bias-free turn rate, used only for coasting
    V3 a_mean{0, 0, 0};   // slow mean of world accel: tilt error and accel bias end up here
    double boost = 0;     // current extra tilt gain, decays after a gap
};

static void step_valid(State& s, const Row& r, double dt, const Survey& sv, const Params& pm) {
    V3 w = r.gyro - sv.bias;
    double a_mag = norm(r.accel);
    V3 correction{0, 0, 0};
    if (std::fabs(a_mag - sv.g) < 1.0) {  // trust accel for "up" only when near 1 g
        V3 up_pred = rotate(conj(s.q), {0, 0, 1});
        correction = cross(r.accel * (1.0 / a_mag), up_pred) * (pm.kp + s.boost);
    }
    s.boost *= std::exp(-dt / 1.0);
    s.q = normalized(mul(s.q, exp_rotvec((w + correction) * dt)));

    // A steady accel error e would settle at a position offset of e * tau_v * tau_p (16 m per
    // m/s^2 at tau 2 and 8). Removing the slow mean first stops steady errors reaching position.
    V3 a_world = rotate(s.q, r.accel) - V3{0, 0, sv.g};
    s.a_mean = s.a_mean + (a_world - s.a_mean) * (dt / pm.tau_a);
    a_world = a_world - s.a_mean;
    s.v = s.v + (a_world - s.v * (1.0 / pm.tau_v)) * dt;
    s.p = s.p + (s.v - s.p * (1.0 / pm.tau_p)) * dt;

    double k = dt / (0.1 + dt);
    s.rate_lp = s.rate_lp + (w - s.rate_lp) * k;
}

static void step_coast(State& s, double dt, const Params& pm) {
    s.boost = pm.boost;
    s.rate_lp = s.rate_lp * std::exp(-dt / pm.tau_rate);
    s.q = normalized(mul(s.q, exp_rotvec(s.rate_lp * dt)));
    s.v = s.v * std::exp(-dt / pm.tau_coast);
    s.p = s.p + (s.v - s.p * (1.0 / pm.tau_p)) * dt;
}

// Integer nanoseconds are printed directly, so no precision is lost to floating point.
static void write_pose(FILE* out, int64_t t_ns, const State& s) {
    long long sec = t_ns / 1000000000LL, ns = t_ns % 1000000000LL;
    std::fprintf(out, "%lld.%09lld %.6f %.6f %.6f %.9f %.9f %.9f %.9f\n", sec, ns, s.p.x, s.p.y, s.p.z,
                 s.q.x, s.q.y, s.q.z, s.q.w);
}

// ---------------------------------------------------------------- main
static int usage() {
    std::fprintf(stderr,
                 "usage: pathfinder <imu.csv | dataset_dir> <out.txt> [options]\n"
                 "  --tau-v S       velocity leak (default 1.0)\n"
                 "  --tau-p S       position leak (default 4.0)\n"
                 "  --tau-a S       slow accel mean removed before integrating (default 3.0)\n"
                 "  --kp X          tilt correction gain (default 0.2)\n"
                 "  --boost X       extra tilt gain after a gap (default 2.0)\n"
                 "  --blind-seconds S  forecast length after the last good row (default 5, 0 = off)\n"
                 "  --fill-gaps     also write poses for rows missing from the input\n");
    return 2;
}

static FILE* open_input(const char* path) {
    static char buf[1024];
    const char* tries[] = {"%s/imu0/data.csv", "%s/mav0/imu0/data.csv"};
    for (const char* t : tries) {
        std::snprintf(buf, sizeof buf, t, path);
        if (FILE* f = std::fopen(buf, "rb")) return f;
    }
    return std::fopen(path, "rb");
}

int main(int argc, char** argv) {
    if (argc < 3) return usage();
    Params pm;
    for (int i = 3; i < argc; i++) {
        bool more = i + 1 < argc;
        if (!std::strcmp(argv[i], "--fill-gaps")) pm.fill_gaps = true;
        else if (!std::strcmp(argv[i], "--tau-v") && more) pm.tau_v = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--tau-p") && more) pm.tau_p = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--tau-a") && more) pm.tau_a = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--boost") && more) pm.boost = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--kp") && more) pm.kp = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--blind-seconds") && more) pm.blind_s = std::atof(argv[++i]);
        else return usage();
    }
    FILE* in = open_input(argv[1]);
    if (!in) { std::fprintf(stderr, "cannot open input: %s\n", argv[1]); return 1; }
    FILE* out = std::fopen(argv[2], "wb");
    if (!out) { std::fprintf(stderr, "cannot open output: %s\n", argv[2]); return 1; }
    static char in_buf[65536], out_buf[65536];
    std::setvbuf(in, in_buf, _IOFBF, sizeof in_buf);
    std::setvbuf(out, out_buf, _IOFBF, sizeof out_buf);

    Survey sv = survey(in);
    if (sv.rows == 0) { std::fprintf(stderr, "no data rows found\n"); return 1; }
    rewind_rows(in);
    State st;
    st.q = initial_attitude(in, sv);
    rewind_rows(in);

    std::fprintf(out, "# timestamp tx ty tz qx qy qz qw\n");
    Row r;
    int64_t last_t = -1, last_good_t = -1;
    long written = 0, gaps = 0;
    const int64_t gap_ns = sv.dt_nominal_ns * 3 / 2;
    while (next_row(in, r)) {
        if (last_t >= 0) {
            int64_t dt_ns = r.t_ns - last_t;
            // A hole in the timestamps: coast across it at the normal sample rate.
            if (dt_ns > gap_ns) {
                gaps++;
                int64_t t = last_t + sv.dt_nominal_ns;
                for (; t < r.t_ns - sv.dt_nominal_ns / 2; t += sv.dt_nominal_ns) {
                    step_coast(st, sv.dt_nominal, pm);
                    if (pm.fill_gaps) { write_pose(out, t, st); written++; }
                    last_t = t;
                }
                dt_ns = r.t_ns - last_t;
            }
            double dt = dt_ns > 0 ? dt_ns * 1e-9 : 0.0;
            if (r.valid) step_valid(st, r, dt, sv, pm);
            else step_coast(st, dt, pm);
        }
        if (r.valid) last_good_t = r.t_ns;
        write_pose(out, r.t_ns, st);
        written++;
        last_t = r.t_ns;
    }

    // IMU has died. If the file already ends in dead rows, those count towards the blind time.
    long blind_rows = 0;
    if (pm.blind_s > 0 && last_good_t >= 0) {
        double still_needed = pm.blind_s - (last_t - last_good_t) * 1e-9;
        blind_rows = still_needed > 0 ? std::lround(still_needed / sv.dt_nominal) : 0;
        for (long i = 0; i < blind_rows; i++) {
            step_coast(st, sv.dt_nominal, pm);
            last_t += sv.dt_nominal_ns;
            write_pose(out, last_t, st);
            written++;
        }
    }
    std::fclose(out);
    std::fclose(in);
    std::fprintf(stderr,
                 "rows=%ld invalid=%ld gaps=%ld written=%ld blind_rows=%ld dt=%.4fms g=%.3f\n"
                 "quiet_block_t=%.3fs gyro_std=%.4f stationary=%d gyro_bias=[%.5f %.5f %.5f]\n",
                 sv.rows, sv.invalid, gaps, written, blind_rows, sv.dt_nominal * 1e3, sv.g,
                 sv.quiet_t_ns * 1e-9, sv.quiet_gyro_std, (int)sv.stationary, sv.bias.x, sv.bias.y, sv.bias.z);
    return 0;
}
