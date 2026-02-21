/**
 * trt_inference.cpp -- TensorRT GPU benchmark for all four scratch-ml models
 *
 * Loads each ONNX model, builds a TensorRT FP32 engine (or loads a cached
 * .engine file from disk), and measures single-window inference latency on GPU
 * using CUDA events.
 *
 * TensorRT is GPU-only; there is no CPU execution path.  The output JSON uses
 * the same schema as cpp_metrics.json so benchmark.py can merge both files into
 * a single comparison table.
 *
 * Usage:
 *   ./trt_inference <path-to-artifacts/onnx>
 *   ./trt_inference                           # defaults to ../artifacts/onnx
 *
 * Outputs:
 *   <onnx_dir>/trt_metrics.json      -- latency stats per model (GPU only)
 *   <onnx_dir>/{ModelName}.engine    -- cached TRT engine (reused on next run)
 *
 * Build (requires -DTRT_ROOT in CMake):
 *   mkdir -p cpp/build && cd cpp/build
 *   cmake .. \
 *     -DORT_ROOT=~/onnxruntime \
 *     -DTRT_ROOT=~/TensorRT \
 *     -DCMAKE_BUILD_TYPE=Release
 *   cmake --build . --parallel
 *
 * First run builds the engine (30-120s per model); subsequent runs load the
 * cached .engine file and complete in seconds.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <nlohmann/json.hpp>

// =============================================================================
// Constants
// =============================================================================

static constexpr int LOOKBACK   = 256;
static constexpr int N_FEATURES = 3;
static constexpr int N_WARMUP   = 10;

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
    int         n_windows;
};

// =============================================================================
// TRT logger
// =============================================================================

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress info/verbose; only print warnings and errors
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << "\n";
        }
    }
};

// =============================================================================
// Helpers
// =============================================================================

std::vector<float> load_binary_floats(const std::string& path, std::size_t count) {
    std::vector<float> data(count);
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    f.read(reinterpret_cast<char*>(data.data()),
           static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) throw std::runtime_error("Short read from: " + path);
    return data;
}

HoldoutMeta load_meta(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    auto j = nlohmann::json::parse(f);
    return {
        j.at("n_windows").get<int>(),
        j.at("lookback").get<int>(),
        j.at("n_features").get<int>(),
    };
}

double percentile(const std::vector<double>& sorted_v, double p) {
    if (sorted_v.empty()) return 0.0;
    double idx = p * static_cast<double>(sorted_v.size() - 1);
    auto   lo  = static_cast<std::size_t>(idx);
    auto   hi  = lo + 1;
    if (hi >= sorted_v.size()) return sorted_v.back();
    double frac = idx - static_cast<double>(lo);
    return sorted_v[lo] * (1.0 - frac) + sorted_v[hi] * frac;
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            throw std::runtime_error(std::string("CUDA error: ")              \
                + cudaGetErrorString(err));                                    \
        }                                                                      \
    } while (0)

// =============================================================================
// Engine build / load
// =============================================================================

/**
 * Build a TensorRT engine from an ONNX file, or load a cached .engine file.
 *
 * On first call the ONNX model is parsed via nvonnxparser, an FP32 engine is
 * built (this is slow -- several minutes for large models), serialized, and
 * written to engine_path.  On subsequent calls the cached file is loaded
 * directly, which is fast.
 *
 * Returns a pair of (runtime, cuda_engine) that the caller must manage.  Both
 * use raw TRT pointers; they are wrapped in unique_ptrs with a custom deleter.
 */
std::pair<
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)>,
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>
>
load_or_build_engine(
    const std::string& onnx_path,
    const std::string& engine_path,
    Logger& logger
) {
    auto rt_deleter = [](nvinfer1::IRuntime* p)     { delete p; };
    auto en_deleter = [](nvinfer1::ICudaEngine* p)  { delete p; };

    using RuntimePtr = std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)>;
    using EnginePtr  = std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)>;

    RuntimePtr runtime(nvinfer1::createInferRuntime(logger), rt_deleter);
    if (!runtime) throw std::runtime_error("Failed to create TRT InferRuntime");

    // --- Try loading cached engine ---
    {
        std::ifstream cache(engine_path, std::ios::binary | std::ios::ate);
        if (cache.good()) {
            std::streamsize size = cache.tellg();
            cache.seekg(0, std::ios::beg);
            std::vector<char> buf(size);
            cache.read(buf.data(), size);
            if (cache) {
                EnginePtr engine(
                    runtime->deserializeCudaEngine(buf.data(), static_cast<std::size_t>(size)),
                    en_deleter);
                if (engine) {
                    std::cout << "    (loaded cached engine from " << engine_path << ")\n";
                    return {std::move(runtime), std::move(engine)};
                }
            }
        }
    }

    // --- Build engine from ONNX ---
    std::cout << "    Building TRT engine from " << onnx_path
              << " (this may take a while)...\n" << std::flush;

    auto builder_deleter = [](nvinfer1::IBuilder* p) { delete p; };
    std::unique_ptr<nvinfer1::IBuilder, decltype(builder_deleter)> builder(
        nvinfer1::createInferBuilder(logger), builder_deleter);
    if (!builder) throw std::runtime_error("Failed to create TRT IBuilder");

    // TRT 10: explicit batch is the only network mode; pass 0 for flags.
    auto net_deleter = [](nvinfer1::INetworkDefinition* p) { delete p; };
    std::unique_ptr<nvinfer1::INetworkDefinition, decltype(net_deleter)> network(
        builder->createNetworkV2(0U), net_deleter);
    if (!network) throw std::runtime_error("Failed to create TRT INetworkDefinition");

    auto parser_deleter = [](nvonnxparser::IParser* p) { delete p; };
    std::unique_ptr<nvonnxparser::IParser, decltype(parser_deleter)> parser(
        nvonnxparser::createParser(*network, logger), parser_deleter);
    if (!parser) throw std::runtime_error("Failed to create ONNX parser");

    if (!parser->parseFromFile(onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        for (int i = 0; i < parser->getNbErrors(); ++i) {
            std::cerr << "  Parse error: " << parser->getError(i)->desc() << "\n";
        }
        throw std::runtime_error("ONNX parse failed: " + onnx_path);
    }

    auto cfg_deleter = [](nvinfer1::IBuilderConfig* p) { delete p; };
    std::unique_ptr<nvinfer1::IBuilderConfig, decltype(cfg_deleter)> config(
        builder->createBuilderConfig(), cfg_deleter);
    if (!config) throw std::runtime_error("Failed to create TRT IBuilderConfig");

    // 1 GiB workspace -- generous for all four model sizes
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Build serialized engine
    auto plan_deleter = [](nvinfer1::IHostMemory* p) { delete p; };
    std::unique_ptr<nvinfer1::IHostMemory, decltype(plan_deleter)> serialized(
        builder->buildSerializedNetwork(*network, *config), plan_deleter);
    if (!serialized) throw std::runtime_error("buildSerializedNetwork failed");

    // Cache to disk
    {
        std::ofstream out(engine_path, std::ios::binary);
        out.write(static_cast<const char*>(serialized->data()),
                  static_cast<std::streamsize>(serialized->size()));
        std::cout << "    Saved engine -> " << engine_path << "\n";
    }

    EnginePtr engine(
        runtime->deserializeCudaEngine(serialized->data(), serialized->size()),
        en_deleter);
    if (!engine) throw std::runtime_error("deserializeCudaEngine failed");

    return {std::move(runtime), std::move(engine)};
}

// =============================================================================
// Benchmark
// =============================================================================

ModelMetrics run_benchmark(
    const std::string&        model_name,
    const std::string&        onnx_path,
    const std::string&        engine_path,
    const std::vector<float>& input_data,
    const std::vector<float>& targets,
    int                       n_windows,
    Logger&                   logger
) {
    const std::size_t window_floats = static_cast<std::size_t>(LOOKBACK * N_FEATURES);

    // ── Load / build engine ───────────────────────────────────────────────────
    auto t_load_start = std::chrono::steady_clock::now();
    auto [runtime, engine] = load_or_build_engine(onnx_path, engine_path, logger);
    auto t_load_end   = std::chrono::steady_clock::now();
    double load_time_ms = std::chrono::duration<double, std::milli>(
        t_load_end - t_load_start).count();

    // ── Execution context ─────────────────────────────────────────────────────
    auto ctx_deleter = [](nvinfer1::IExecutionContext* p) { delete p; };
    std::unique_ptr<nvinfer1::IExecutionContext, decltype(ctx_deleter)> context(
        engine->createExecutionContext(), ctx_deleter);
    if (!context) throw std::runtime_error("Failed to create TRT execution context");

    // ── GPU buffers ───────────────────────────────────────────────────────────
    // Input:  (1, LOOKBACK, N_FEATURES)  float32
    // Output: (1, 1)                     float32
    const std::size_t input_bytes  = window_floats * sizeof(float);
    const std::size_t output_bytes = sizeof(float);

    float* d_input  = nullptr;
    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input,  input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    // Bind input/output by name (TRT 8.x+ API)
    context->setTensorAddress("input",  d_input);
    context->setTensorAddress("output", d_output);

    // ── CUDA timing events ────────────────────────────────────────────────────
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Warmup ────────────────────────────────────────────────────────────────
    int warmup_count = std::min(N_WARMUP, n_windows);
    for (int i = 0; i < warmup_count; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_input,
            input_data.data() + static_cast<std::size_t>(i) * window_floats,
            input_bytes,
            cudaMemcpyHostToDevice,
            stream));
        context->enqueueV3(stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ── Benchmark ─────────────────────────────────────────────────────────────
    std::vector<double> latencies(n_windows);
    std::vector<float>  predictions(n_windows);

    for (int i = 0; i < n_windows; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(
            d_input,
            input_data.data() + static_cast<std::size_t>(i) * window_floats,
            input_bytes,
            cudaMemcpyHostToDevice,
            stream));

        CUDA_CHECK(cudaEventRecord(ev_start, stream));
        context->enqueueV3(stream);
        CUDA_CHECK(cudaEventRecord(ev_end, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
        latencies[i] = static_cast<double>(ms);

        // Copy scalar output back to host
        CUDA_CHECK(cudaMemcpy(
            &predictions[i], d_output, output_bytes, cudaMemcpyDeviceToHost));
    }

    // ── RMSE ──────────────────────────────────────────────────────────────────
    double sum_sq = 0.0;
    for (int i = 0; i < n_windows; ++i) {
        double diff = static_cast<double>(predictions[i])
                    - static_cast<double>(targets[i]);
        sum_sq += diff * diff;
    }
    double rmse = std::sqrt(sum_sq / static_cast<double>(n_windows));

    // ── Statistics ────────────────────────────────────────────────────────────
    double n    = static_cast<double>(n_windows);
    double mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / n;
    double variance = 0.0;
    for (double v : latencies) variance += (v - mean) * (v - mean);
    double std_dev = std::sqrt(variance / n);

    std::vector<double> sorted_lat = latencies;
    std::sort(sorted_lat.begin(), sorted_lat.end());

    // ── Cleanup ───────────────────────────────────────────────────────────────
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);

    ModelMetrics m;
    m.model_name                 = model_name;
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

    // ── Check CUDA device ─────────────────────────────────────────────────────
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    if (n_devices == 0) {
        std::cerr << "No CUDA-capable device found. TensorRT requires a GPU.\n";
        return 1;
    }
    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, 0);
    std::cout << "GPU: " << props.name << "\n\n";

    // ── Model list ────────────────────────────────────────────────────────────
    const std::vector<std::string> model_names{
        "NaiveLSTM", "ImprovedLSTM", "Transformer", "TCN"
    };

    Logger logger;
    std::vector<ModelMetrics> all_metrics;

    for (const auto& model_name : model_names) {
        const std::string onnx_path   = onnx_dir + "/" + model_name + ".onnx";
        const std::string engine_path = onnx_dir + "/" + model_name + ".engine";

        std::cout << model_name << " (GPU) ...\n" << std::flush;
        try {
            ModelMetrics m = run_benchmark(
                model_name, onnx_path, engine_path,
                input_data, targets, meta.n_windows, logger);
            all_metrics.push_back(m);
            std::cout << "  mean=" << m.mean_latency_ms       << " ms  "
                      << "p99="   << m.p99_latency_ms         << " ms  "
                      << "rmse="  << m.rmse                   << "\n";
        } catch (const std::exception& e) {
            std::cerr << "  Error: " << e.what() << "\n";
        }
    }

    // ── Serialise results ─────────────────────────────────────────────────────
    nlohmann::json result = nlohmann::json::array();
    for (const auto& m : all_metrics) {
        result.push_back({
            {"model",    m.model_name},
            {"provider", "TensorRTExecutionProvider"},
            {"device",   "GPU"},
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
            {"rmse",      m.rmse},
            {"n_windows", m.n_windows},
        });
    }

    const std::string out_path = onnx_dir + "/trt_metrics.json";
    std::ofstream out_file(out_path);
    out_file << result.dump(2) << "\n";
    std::cout << "\nSaved " << out_path << "\n";

    return 0;
}
