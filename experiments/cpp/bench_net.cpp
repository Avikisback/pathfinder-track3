// Forward pass of VelNet in plain C++: same layers and weights as PyTorch, checked against its output.
// Exact GELU (std::erf), single thread, no SIMD intrinsics. Usage: bench_net <weights.bin> [passes]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<float> g_blob;
static size_t g_pos = 0;
static const float* take(size_t n) { const float* p = g_blob.data() + g_pos; g_pos += n; return p; }

static inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678f)); }

struct Conv {
    int in, out, k, stride, pad;
    const float* w;
    const float* b;
    void load(int i, int o, int kk, int s, int p) {
        in = i; out = o; k = kk; stride = s; pad = p;
        w = take(size_t(o) * i * kk);
        b = take(o);
    }
    int out_len(int T) const { return (T + 2 * pad - k) / stride + 1; }
    // x: channels-first [in][T], y: [out][outT]
    void run(const float* x, int T, float* y) const {
        int L = out_len(T);
        for (int o = 0; o < out; o++) {
            const float* wo = w + size_t(o) * in * k;
            float* yo = y + size_t(o) * L;
            for (int t = 0; t < L; t++) yo[t] = b[o];
            for (int i = 0; i < in; i++) {
                const float* xi = x + size_t(i) * T;
                const float* wi = wo + size_t(i) * k;
                for (int kk = 0; kk < k; kk++) {
                    float c = wi[kk];
                    for (int t = 0; t < L; t++) {
                        int s = t * stride + kk - pad;
                        if (s >= 0 && s < T) yo[t] += c * xi[s];
                    }
                }
            }
        }
    }
};

struct Net {
    int T, n, ch[8], fc, outn, c_in;
    Conv down[8], res[8];
    const float* w1; const float* b1; const float* w2; const float* b2;
    int flat;
    std::vector<float> buf[3];
    void load(int T_, int c_in_, const int* c, int n_, int k, int fc_, int out_) {
        T = T_; c_in = c_in_; n = n_; fc = fc_; outn = out_;
        int a = c_in, L = T;
        for (int i = 0; i < n; i++) {
            ch[i] = c[i];
            down[i].load(a, c[i], k, 2, k / 2);
            res[i].load(c[i], c[i], 3, 1, 1);
            a = c[i];
            L = down[i].out_len(L);
        }
        flat = a * L;
        w1 = take(size_t(fc) * flat); b1 = take(fc);
        w2 = take(size_t(outn) * fc); b2 = take(outn);
        for (auto& v : buf) v.assign(size_t(64) * T, 0.f);
    }
    void forward(const float* x, float* out) {
        const float* cur = x;
        int L = T;
        for (int i = 0; i < n; i++) {
            int nl = down[i].out_len(L);
            down[i].run(cur, L, buf[0].data());
            size_t m = size_t(ch[i]) * nl;
            for (size_t j = 0; j < m; j++) buf[0][j] = gelu(buf[0][j]);
            res[i].run(buf[0].data(), nl, buf[1].data());
            for (size_t j = 0; j < m; j++) buf[2][j] = buf[0][j] + gelu(buf[1][j]);
            buf[0].swap(buf[2]);
            // result lives in buf[0]; copy to buf[2] so the next layer can write buf[0]
            for (size_t j = 0; j < m; j++) buf[2][j] = buf[0][j];
            cur = buf[2].data();
            L = nl;
        }
        std::vector<float>& h = buf[1];
        for (int o = 0; o < fc; o++) {
            float s = b1[o];
            const float* w = w1 + size_t(o) * flat;
            for (int j = 0; j < flat; j++) s += w[j] * cur[j];
            h[o] = gelu(s);
        }
        for (int o = 0; o < outn; o++) {
            float s = b2[o];
            const float* w = w2 + size_t(o) * fc;
            for (int j = 0; j < fc; j++) s += w[j] * h[j];
            out[o] = s;
        }
    }
};

int main(int argc, char** argv) {
    FILE* f = std::fopen(argv[1], "rb");
    if (!f) return 1;
    int hdr[16];
    if (std::fread(hdr, 4, 16, f) != 16) return 1;
    int T = hdr[0], c_in = hdr[1], n = hdr[2], k = hdr[3], fc = hdr[4], outn = hdr[5];
    int ch[8];
    for (int i = 0; i < n; i++) ch[i] = hdr[6 + i];
    int nw = hdr[14];
    std::vector<float> test_in(size_t(c_in) * T), test_out(outn);
    if (std::fread(test_in.data(), 4, test_in.size(), f) != test_in.size()) return 1;
    if (std::fread(test_out.data(), 4, outn, f) != size_t(outn)) return 1;
    g_blob.resize(nw);
    if (std::fread(g_blob.data(), 4, nw, f) != size_t(nw)) return 1;
    std::fclose(f);
    Net net;
    net.load(T, c_in, ch, n, k, fc, outn);
    std::vector<float> out(outn);
    net.forward(test_in.data(), out.data());
    double md = 0;
    for (int i = 0; i < outn; i++) md = std::fmax(md, std::fabs(out[i] - test_out[i]));
    std::printf("max difference from PyTorch: %.2e\n", md);
    int passes = argc > 2 ? std::atoi(argv[2]) : 2000;
    auto t0 = std::chrono::steady_clock::now();
    float sink = 0;
    for (int i = 0; i < passes; i++) {
        test_in[i % test_in.size()] += 1e-4f;
        net.forward(test_in.data(), out.data());
        sink += out[0];
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("forward pass %.1f us; at 10 Hz output that is %.2f ms per 1 s of IMU (limit 20 ms). sink %g\n",
                s / passes * 1e6, s / passes * 10 * 1e3, sink);
    return 0;
}
