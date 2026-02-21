cmake_minimum_required(VERSION 3.20)
project(scratch_ml_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# ONNX Runtime
# =============================================================================
#
# Two installation paths are supported -- CMake detects which one to use:
#
#   apt deb  (CPU only, Ubuntu 24.10+):
#     sudo apt install libonnxruntime libonnxruntime-dev
#     cmake .. -DCMAKE_BUILD_TYPE=Release          # no ORT_ROOT needed
#
#   tar.gz   (CPU or GPU, any Ubuntu):
#     # Download from https://github.com/microsoft/onnxruntime/releases
#     cmake .. -DORT_ROOT=~/onnxruntime -DCMAKE_BUILD_TYPE=Release
#
# When ORT_ROOT is provided it takes priority and the system paths are not
# searched.  When omitted, find_library / find_path probe the default system
# paths for a deb-installed library.

if(DEFINED ORT_ROOT)
    # --- tar.gz install: user-supplied prefix ---
    string(REGEX REPLACE "^~" "$ENV{HOME}" ORT_ROOT "${ORT_ROOT}")
    message(STATUS "ORT_ROOT provided -- using tar.gz install at ${ORT_ROOT}")

    find_library(ORT_LIB
        NAMES onnxruntime
        HINTS "${ORT_ROOT}/lib"
        NO_DEFAULT_PATH
        DOC "ONNX Runtime shared library"
    )
    find_path(ORT_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        HINTS "${ORT_ROOT}/include"
        NO_DEFAULT_PATH
        DOC "ONNX Runtime include directory"
    )
else()
    # --- apt deb install: search standard system paths ---
    message(STATUS "ORT_ROOT not set -- searching system paths (apt deb install)")

    find_library(ORT_LIB
        NAMES onnxruntime
        DOC "ONNX Runtime shared library"
    )
    # apt deb places headers under /usr/include/onnxruntime/
    find_path(ORT_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATH_SUFFIXES onnxruntime
        DOC "ONNX Runtime include directory"
    )
endif()

if(NOT ORT_LIB OR NOT ORT_INCLUDE_DIR)
    message(FATAL_ERROR
        "ONNX Runtime not found.\n"
        "  apt deb (CPU, Ubuntu 24.10+): sudo apt install libonnxruntime libonnxruntime-dev\n"
        "  tar.gz  (CPU or GPU):         cmake .. -DORT_ROOT=/path/to/onnxruntime\n"
        "  Downloads: https://github.com/microsoft/onnxruntime/releases\n"
    )
endif()

message(STATUS "ORT library : ${ORT_LIB}")
message(STATUS "ORT include : ${ORT_INCLUDE_DIR}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
)

# =============================================================================
# nlohmann/json  (header-only, pinned for reproducibility)
# =============================================================================
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# =============================================================================
# Binary
# =============================================================================
add_executable(inference inference.cpp)
target_link_libraries(inference PRIVATE
    onnxruntime
    nlohmann_json::nlohmann_json
)

# For tar.gz installs copy ORT runtime libs next to the binary so provider
# plugins (CUDA/TensorRT EP) are discoverable without LD_LIBRARY_PATH.
# apt deb installs are already in the system linker path; no copy needed.
if(DEFINED ORT_ROOT)
    file(GLOB ORT_RUNTIME_LIBS
        "${ORT_ROOT}/lib/libonnxruntime.so*"
        "${ORT_ROOT}/lib/libonnxruntime_providers*.so*"
    )
    list(REMOVE_DUPLICATES ORT_RUNTIME_LIBS)

    add_custom_command(TARGET inference POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ORT_RUNTIME_LIBS}
            "$<TARGET_FILE_DIR:inference>"
    )
endif()

# =============================================================================
# TensorRT inference binary (auto-detected via find_library / find_path)
# =============================================================================
#
# Works automatically with a TensorRT .deb local install, which places headers
# and libraries into standard system paths:
#   /usr/include/x86_64-linux-gnu/   (NvInfer.h, NvOnnxParser.h, ...)
#   /usr/lib/x86_64-linux-gnu/       (libnvinfer.so, libnvonnxparser.so, ...)
#
# No TRT_ROOT needed -- just run:
#   cmake .. -DORT_ROOT=$HOME/onnxruntime -DCMAKE_BUILD_TYPE=Release
#
# If TensorRT is not found the target is silently skipped.

find_path(TRT_INCLUDE_DIR
    NAMES NvInfer.h
    DOC "TensorRT include directory"
)

find_library(TRT_LIB_NVINFER
    NAMES nvinfer
    DOC "TensorRT libnvinfer"
)

find_library(TRT_LIB_NVONNXPARSER
    NAMES nvonnxparser
    DOC "TensorRT libnvonnxparser"
)

find_library(TRT_LIB_NVINFER_PLUGIN
    NAMES nvinfer_plugin
    DOC "TensorRT libnvinfer_plugin"
)

if(TRT_INCLUDE_DIR AND TRT_LIB_NVINFER AND TRT_LIB_NVONNXPARSER AND TRT_LIB_NVINFER_PLUGIN)
    message(STATUS "TensorRT found -- trt_inference target enabled")
    message(STATUS "  include      : ${TRT_INCLUDE_DIR}")
    message(STATUS "  nvinfer      : ${TRT_LIB_NVINFER}")
    message(STATUS "  nvonnxparser : ${TRT_LIB_NVONNXPARSER}")
    message(STATUS "  nvinfer_plugin : ${TRT_LIB_NVINFER_PLUGIN}")

    # CUDAToolkit (CMake 3.17+) provides the CUDA::cudart imported target and
    # handles include dirs automatically -- preferred over the legacy find_package(CUDA).
    find_package(CUDAToolkit REQUIRED)

    add_executable(trt_inference trt_inference.cpp)

    target_include_directories(trt_inference PRIVATE
        "${TRT_INCLUDE_DIR}"
    )

    target_link_libraries(trt_inference PRIVATE
        "${TRT_LIB_NVINFER}"
        "${TRT_LIB_NVONNXPARSER}"
        "${TRT_LIB_NVINFER_PLUGIN}"
        CUDA::cudart
        nlohmann_json::nlohmann_json
    )
    # .deb installs are already in the system linker path; no RPATH copy needed.
else()
    message(STATUS "TensorRT not found in system paths -- skipping trt_inference target")
    if(NOT TRT_INCLUDE_DIR)
        message(STATUS "  Missing: NvInfer.h")
    endif()
    if(NOT TRT_LIB_NVINFER)
        message(STATUS "  Missing: libnvinfer")
    endif()
    if(NOT TRT_LIB_NVONNXPARSER)
        message(STATUS "  Missing: libnvonnxparser")
    endif()
    if(NOT TRT_LIB_NVINFER_PLUGIN)
        message(STATUS "  Missing: libnvinfer_plugin")
    endif()
    message(STATUS "  Install: sudo apt install tensorrt  (requires NVIDIA CUDA .deb repo)")
    message(STATUS "  Or download from: https://developer.nvidia.com/tensorrt-download")
endif()
