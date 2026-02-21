/**
 * inference.cpp -- ONNX Runtime benchmark for all four scratch-ml models
 *
 * Loads each ONNX model, feeds it the holdout windows exported by
 * src/train.py, and measures single-window inference latency on CPU
 * (and CUDA if available).
 *
 * Usage:
 *   ./inference <path-to-artifacts/onnx>
 *   ./inference                          # defaults to ../artifacts/onnx
 *
 * Outputs:
 *   <onnx_dir>/cpp_metrics.json  -- latency stats, throughput, RMSE per (model, provider)
 *
 * Build:
 *   mkdir -p build && cd build
 *   cmake .. -DORT_ROOT=~/onnxruntime -DCMAKE_BUILD_TYPE=Release
 *   cmake --build . --parallel
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/resource.h>  // getrusage() for peak RSS

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

// =============================================================================
// Constants
// =============================================================================

static constexpr int LOOKBACK   = 256;
static constexpr int N_FEATURES = 3;
static constexpr int N_WARMUP   = 10;  // passes excluded from timing stats

// =============================================================================
// Data types
// =============================================================================

struct HoldoutMeta {
    int n_windows;
    int lookback;
    int n_features;
};

struct ModelMetrics {
    std::string model_name;
    std::string provider;
    double      load_time_ms;
    double      mean_latency_ms;
    double      min_latency_ms;
    double      max_latency_ms;
    double      std_latency_ms;
    double      p50_latency_ms;
    double      p95_latency_ms;
    double      p99_latency_ms;
    double      throughput_windows_per_sec;
    double      rmse;
    double      peak_rss_mb;
    int         n_windows;
};

// =============================================================================
// Helpers
// =============================================================================

/**
 * Read count float32 values from a raw binary file into a vector.
 * Throws std::runtime_error if the file cannot be opened or the read is short.
 */
std::vector<float> load_binary_floats(const std::string& path, std::size_t count) {
    std::vector<float> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("Cannot open: " + path);
    }
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) {
        throw std::runtime_error("Short read from: " + path);
    }
    return data;
}

/**
 * Parse holdout_meta.json and return the three integer fields.
 */
HoldoutMeta load_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Cannot open: " + path);
    }
    auto j = nlohmann::json::parse(f);
    return {
        j.at("n_windows").get<int>(),
        j.at("lookback").get<int>(),
        j.at("n_features").get<int>(),
    };
}

/**
 * Linear-interpolation percentile on a pre-sorted vector.
 * p must be in [0.0, 1.0].
 */
double percentile(const std::vector<double>& sorted_v, double p) {
    if (sorted_v.empty()) return 0.0;
    double idx = p * static_cast<double>(sorted_v.size() - 1);
    auto   lo  = static_cast<std::size_t>(idx);
    auto   hi  = lo + 1;
    if (hi >= sorted_v.size()) return sorted_v.back();
    double frac = idx - static_cast<double>(lo);
    return sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac;
}

/**
 * Return the process peak resident set size in megabytes.
 * Uses getrusage(RUSAGE_SELF) -- available on Linux and macOS.
 * Linux reports ru_maxrss in kilobytes; macOS reports it in bytes.
 */
double peak_rss_mb() {
    struct rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return static_cast<double>(usage.ru_maxrss) / 1024.0 / 1024.0;
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

// =============================================================================
// Benchmark
// =============================================================================

/**
 * Create an ORT session for one model and one execution provider, run
 * N_WARMUP warmup passes (not timed), then benchmark every holdout window
 * one at a time (batch_size=1).
 *
 * IntraOpNumThreads and InterOpNumThreads are both set to 1 so that each
 * window is processed serially -- this gives the cleanest single-window
 * latency measurement without interference from thread-pool scheduling.
 *
 * The input tensor is created with Ort::Value::CreateTensor which borrows
 * the caller's memory (zero-copy).  input_data must remain alive for the
 * duration of this function.
 */
ModelMetrics run_benchmark(
    const std::string&        model_name,
    const std::string&        onnx_path,
    const std::vector<float>& input_data,   // n_windows * LOOKBACK * N_FEATURES floats
    const std::vector<float>& targets,      // n_windows floats
    int                       n_windows,
    const std::string&        provider
) {
    // ── Session setup ─────────────────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, model_name.c_str());

    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetInterOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    if (provider == "CUDAExecutionProvider") {
        OrtCUDAProviderOptions cuda_opts{};
        opts.AppendExecutionProvider_CUDA(cuda_opts);
    }

    auto t_load_start = std::chrono::steady_clock::now();
    Ort::Session session(env, onnx_path.c_str(), opts);
    auto t_load_end   = std::chrono::steady_clock::now();
    double load_time_ms = std::chrono::duration<double, std::milli>(
        t_load_end - t_load_start).count();

    // ── Input/output name arrays (C strings required by the ORT C++ API) ─────
    const char* input_names[]  = {"input"};
    const char* output_names[] = {"output"};

    // Shape is (1, LOOKBACK, N_FEATURES) -- one window at a time
    const std::array<std::int64_t, 3> shape{1, LOOKBACK, N_FEATURES};
    static const std::size_t window_floats = LOOKBACK * N_FEATURES;

    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ── Warmup ────────────────────────────────────────────────────────────────
    int warmup_count = std::min(N_WARMUP, n_windows);
    for (int i = 0; i < warmup_count; ++i) {
        // const_cast is safe: ORT does not modify input data
        float* ptr = const_cast<float*>(input_data.data() + i * window_floats);
        auto tensor = Ort::Value::CreateTensor<float>(
            mem_info, ptr, window_floats, shape.data(), shape.size());
        session.Run(Ort::RunOptions{nullptr},
                    input_names, &tensor, 1,
                    output_names, 1);
    }

    // ── Benchmark ─────────────────────────────────────────────────────────────
    std::vector<double> latencies(n_windows);
    std::vector<float>  predictions(n_windows);

    for (int i = 0; i < n_windows; ++i) {
        float* ptr = const_cast<float*>(input_data.data() + i * window_floats);
        auto tensor = Ort::Value::CreateTensor<float>(
            mem_info, ptr, window_floats, shape.data(), shape.size());

        auto t0 = std::chrono::steady_clock::now();
        auto outputs = session.Run(Ort::RunOptions{nullptr},
                                   input_names, &tensor, 1,
                                   output_names, 1);
        auto t1 = std::chrono::steady_clock::now();

        latencies[i]   = std::chrono::duration<double, std::milli>(t1 - t0).count();
        predictions[i] = *outputs[0].GetTensorData<float>();
    }

    // ── RMSE ──────────────────────────────────────────────────────────────────
    double sum_sq = 0.0;
    for (int i = 0; i < n_windows; ++i) {
        double diff = static_cast<double>(predictions[i])
                    - static_cast<double>(targets[i]);
        sum_sq += diff * diff;
    }
    double rmse = std::sqrt(sum_sq / static_cast<double>(n_windows));

    // ── Latency statistics ────────────────────────────────────────────────────
    double n = static_cast<double>(n_windows);
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / n;

    double variance = 0.0;
    for (double v : latencies) variance += (v - mean) * (v - mean);
    double std_dev = std::sqrt(variance / n);

    std::vector<double> sorted_lat = latencies;
    std::sort(sorted_lat.begin(), sorted_lat.end());

    ModelMetrics m;
    m.model_name                 = model_name;
    m.provider                   = provider;
    m.load_time_ms               = load_time_ms;
    m.mean_latency_ms            = mean;
    m.min_latency_ms             = sorted_lat.front();
    m.max_latency_ms             = sorted_lat.back();
    m.std_latency_ms             = std_dev;
    m.p50_latency_ms             = percentile(sorted_lat, 0.50);
    m.p95_latency_ms             = percentile(sorted_lat, 0.95);
    m.p99_latency_ms             = percentile(sorted_lat, 0.99);
    m.throughput_windows_per_sec = (mean > 0.0) ? 1000.0 / mean : 0.0;
    m.rmse                       = rmse;
    m.peak_rss_mb                = peak_rss_mb();
    m.n_windows                  = n_windows;

    return m;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    const std::string onnx_dir = (argc > 1) ? argv[1] : "../artifacts/onnx";
    std::cout << "ONNX dir: " << onnx_dir << "\n";

    // ── Load holdout data ─────────────────────────────────────────────────────
    HoldoutMeta meta = load_meta(onnx_dir + "/holdout_meta.json");
    std::cout << "Holdout: " << meta.n_windows << " windows x "
              << meta.lookback   << " timesteps x "
              << meta.n_features << " features\n";

    if (meta.lookback != LOOKBACK || meta.n_features != N_FEATURES) {
        std::cerr << "ERROR: meta mismatch -- expected lookback="
                  << LOOKBACK << " n_features=" << N_FEATURES << "\n";
        return 1;
    }

    auto input_data = load_binary_floats(
        onnx_dir + "/holdout_input.bin",
        static_cast<std::size_t>(meta.n_windows) * meta.lookback * meta.n_features);
    auto targets = load_binary_floats(
        onnx_dir + "/holdout_target.bin",
        static_cast<std::size_t>(meta.n_windows));

    // ── Model list (same order as Python) ─────────────────────────────────────
    const std::vector<std::string> model_names{
        "NaiveLSTM", "ImprovedLSTM", "Transformer", "TCN"
    };

    // ── Provider list: CPU always; CUDA only if the EP is compiled in ─────────
    std::vector<std::string> ep_list{"CPUExecutionProvider"};
    {
        auto avail = Ort::GetAvailableProviders();
        bool cuda_ok = std::find(avail.begin(), avail.end(),
                                 "CUDAExecutionProvider") != avail.end();
        if (cuda_ok) {
            ep_list.emplace_back("CUDAExecutionProvider");
        } else {
            std::cerr << "CUDAExecutionProvider not available; "
                         "skipping GPU benchmarks.\n";
        }
    }

    // ── Benchmark loop: outer = providers, inner = models ─────────────────────
    std::vector<ModelMetrics> all_metrics;

    for (const auto& provider : ep_list) {
        std::cout << "\n=== Provider: " << provider << " ===\n";
        for (const auto& model_name : model_names) {
            const std::string onnx_path = onnx_dir + "/" + model_name + ".onnx";
            std::cout << "  " << model_name << " ... " << std::flush;
            try {
                ModelMetrics m = run_benchmark(
                    model_name, onnx_path,
                    input_data, targets,
                    meta.n_windows, provider);
                all_metrics.push_back(m);
                std::cout << "mean=" << m.mean_latency_ms       << " ms  "
                          << "p99="  << m.p99_latency_ms        << " ms  "
                          << "rmse=" << m.rmse                  << "\n";
            } catch (const Ort::Exception& e) {
                std::cerr << "ORT error: " << e.what() << "\n";
            } catch (const std::exception& e) {
                std::cerr << "Error: " << e.what() << "\n";
            }
        }
    }

    // ── Serialise results ─────────────────────────────────────────────────────
    nlohmann::json result = nlohmann::json::array();
    for (const auto& m : all_metrics) {
        result.push_back({
            {"model",    m.model_name},
            {"provider", m.provider},
            {"load_time_ms", m.load_time_ms},
            {"latency_ms", {
                {"mean", m.mean_latency_ms},
                {"min",  m.min_latency_ms},
                {"max",  m.max_latency_ms},
                {"std",  m.std_latency_ms},
                {"p50",  m.p50_latency_ms},
                {"p95",  m.p95_latency_ms},
                {"p99",  m.p99_latency_ms},
            }},
            {"throughput_windows_per_sec", m.throughput_windows_per_sec},
            {"rmse",        m.rmse},
            {"peak_rss_mb", m.peak_rss_mb},
            {"n_windows",   m.n_windows},
        });
    }

    const std::string out_path = onnx_dir + "/cpp_metrics.json";
    std::ofstream out_file(out_path);
    out_file << result.dump(2) << "\n";
    std::cout << "\nSaved " << out_path << "\n";

    return 0;
}
