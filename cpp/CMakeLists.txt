cmake_minimum_required(VERSION 3.20)
project(scratch_ml_inference CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# =============================================================================
# ONNX Runtime
# =============================================================================
# Download a prebuilt release from https://github.com/microsoft/onnxruntime/releases
# e.g. onnxruntime-linux-x64-1.20.1.tgz, extract it, then pass the path here:
#
#   cmake .. -DORT_ROOT=~/onnxruntime -DCMAKE_BUILD_TYPE=Release
#
if(NOT DEFINED ORT_ROOT)
    message(FATAL_ERROR
        "ORT_ROOT not set.\n"
        "Download a prebuilt ONNX Runtime release from:\n"
        "  https://github.com/microsoft/onnxruntime/releases\n"
        "Extract it and pass: -DORT_ROOT=/path/to/onnxruntime"
    )
endif()

# CMake does not expand ~ -- normalise it to $HOME so both forms work
string(REGEX REPLACE "^~" "$ENV{HOME}" ORT_ROOT "${ORT_ROOT}")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION             "${ORT_ROOT}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
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

# Copy the shared library next to the binary so it is found at runtime
# without needing to set LD_LIBRARY_PATH.
add_custom_command(TARGET inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_ROOT}/lib/libonnxruntime.so"
        "$<TARGET_FILE_DIR:inference>"
)
