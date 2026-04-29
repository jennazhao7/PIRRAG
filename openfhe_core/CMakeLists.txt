cmake_minimum_required(VERSION 3.16)
project(openfhe_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenFHE CONFIG REQUIRED)
find_package(OpenMP)

function(target_use_openmp target_name)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target_name} PUBLIC OpenMP::OpenMP_CXX)
  endif()
endfunction()

add_library(openfhe_common STATIC
    src/io_utils.cpp
)
target_include_directories(openfhe_common PUBLIC include)
target_use_openmp(openfhe_common)

# OpenFHE installs are not always exported with include dirs in targets,
# so add robust include hints from common prefixes.
if(OpenFHE_DIR)
  get_filename_component(_openfhe_lib_dir "${OpenFHE_DIR}" DIRECTORY)
  get_filename_component(_openfhe_prefix "${_openfhe_lib_dir}" DIRECTORY)
  if(EXISTS "${_openfhe_prefix}/include")
    target_include_directories(openfhe_common PUBLIC "${_openfhe_prefix}/include")
  endif()
  if(EXISTS "${_openfhe_prefix}/include/openfhe")
    target_include_directories(openfhe_common PUBLIC "${_openfhe_prefix}/include/openfhe")
  endif()
endif()

if(DEFINED ENV{CONDA_PREFIX})
  if(EXISTS "$ENV{CONDA_PREFIX}/include")
    target_include_directories(openfhe_common PUBLIC "$ENV{CONDA_PREFIX}/include")
  endif()
  if(EXISTS "$ENV{CONDA_PREFIX}/include/openfhe")
    target_include_directories(openfhe_common PUBLIC "$ENV{CONDA_PREFIX}/include/openfhe")
  endif()
  if(EXISTS "$ENV{CONDA_PREFIX}/include/openfhe/core")
    target_include_directories(openfhe_common PUBLIC "$ENV{CONDA_PREFIX}/include/openfhe/core")
  endif()
  if(EXISTS "$ENV{CONDA_PREFIX}/include/openfhe/pke")
    target_include_directories(openfhe_common PUBLIC "$ENV{CONDA_PREFIX}/include/openfhe/pke")
  endif()
  if(EXISTS "$ENV{CONDA_PREFIX}/include/openfhe/binfhe")
    target_include_directories(openfhe_common PUBLIC "$ENV{CONDA_PREFIX}/include/openfhe/binfhe")
  endif()
endif()

if(EXISTS "/usr/local/include")
  target_include_directories(openfhe_common PUBLIC "/usr/local/include")
endif()
if(EXISTS "/usr/local/include/openfhe")
  target_include_directories(openfhe_common PUBLIC "/usr/local/include/openfhe")
endif()
if(EXISTS "/usr/local/include/openfhe/core")
  target_include_directories(openfhe_common PUBLIC "/usr/local/include/openfhe/core")
endif()
if(EXISTS "/usr/local/include/openfhe/pke")
  target_include_directories(openfhe_common PUBLIC "/usr/local/include/openfhe/pke")
endif()
if(EXISTS "/usr/local/include/openfhe/binfhe")
  target_include_directories(openfhe_common PUBLIC "/usr/local/include/openfhe/binfhe")
endif()
if(EXISTS "/usr/include")
  target_include_directories(openfhe_common PUBLIC "/usr/include")
endif()
if(EXISTS "/usr/include/openfhe")
  target_include_directories(openfhe_common PUBLIC "/usr/include/openfhe")
endif()
if(EXISTS "/usr/include/openfhe/core")
  target_include_directories(openfhe_common PUBLIC "/usr/include/openfhe/core")
endif()
if(EXISTS "/usr/include/openfhe/pke")
  target_include_directories(openfhe_common PUBLIC "/usr/include/openfhe/pke")
endif()
if(EXISTS "/usr/include/openfhe/binfhe")
  target_include_directories(openfhe_common PUBLIC "/usr/include/openfhe/binfhe")
endif()
target_link_libraries(openfhe_common PUBLIC ${OpenFHE_LIBRARIES})

add_executable(openfhe_keygen src/openfhe_keygen.cpp)
target_link_libraries(openfhe_keygen PRIVATE openfhe_common)

add_executable(openfhe_encrypt_query src/openfhe_encrypt_query.cpp)
target_link_libraries(openfhe_encrypt_query PRIVATE openfhe_common)

add_executable(openfhe_encrypt_query_centroid_batched src/openfhe_encrypt_query_centroid_batched.cpp)
target_link_libraries(openfhe_encrypt_query_centroid_batched PRIVATE openfhe_common)

add_executable(openfhe_encrypt_queries_batched src/openfhe_encrypt_queries_batched.cpp)
target_link_libraries(openfhe_encrypt_queries_batched PRIVATE openfhe_common)

add_executable(openfhe_encrypt_queries_centroid_batched src/openfhe_encrypt_queries_centroid_batched.cpp)
target_link_libraries(openfhe_encrypt_queries_centroid_batched PRIVATE openfhe_common)

add_executable(openfhe_compute_distances src/openfhe_compute_distances.cpp)
target_link_libraries(openfhe_compute_distances PRIVATE openfhe_common)
target_use_openmp(openfhe_compute_distances)

add_executable(openfhe_compute_distances_serial src/openfhe_compute_distances_serial.cpp)
target_link_libraries(openfhe_compute_distances_serial PRIVATE openfhe_common)
target_use_openmp(openfhe_compute_distances_serial)

add_executable(openfhe_compute_distances_centroid_batched src/openfhe_compute_distances_centroid_batched.cpp)
target_link_libraries(openfhe_compute_distances_centroid_batched PRIVATE openfhe_common)
target_use_openmp(openfhe_compute_distances_centroid_batched)

add_executable(openfhe_compute_distances_centroid_prepacked src/openfhe_compute_distances_centroid_prepacked.cpp)
target_link_libraries(openfhe_compute_distances_centroid_prepacked PRIVATE openfhe_common)
target_use_openmp(openfhe_compute_distances_centroid_prepacked)

add_executable(openfhe_compute_distances_query_centroid_batched src/openfhe_compute_distances_query_centroid_batched.cpp)
target_link_libraries(openfhe_compute_distances_query_centroid_batched PRIVATE openfhe_common)
target_use_openmp(openfhe_compute_distances_query_centroid_batched)

add_executable(openfhe_compute_distances_batched src/openfhe_compute_distances_batched.cpp)
target_link_libraries(openfhe_compute_distances_batched PRIVATE openfhe_common)
target_use_openmp(openfhe_compute_distances_batched)

add_executable(openfhe_decrypt_topk src/openfhe_decrypt_topk.cpp)
target_link_libraries(openfhe_decrypt_topk PRIVATE openfhe_common)

add_executable(openfhe_decrypt_topk_centroid_batched src/openfhe_decrypt_topk_centroid_batched.cpp)
target_link_libraries(openfhe_decrypt_topk_centroid_batched PRIVATE openfhe_common)

add_executable(openfhe_decrypt_topk_query_centroid_batched src/openfhe_decrypt_topk_query_centroid_batched.cpp)
target_link_libraries(openfhe_decrypt_topk_query_centroid_batched PRIVATE openfhe_common)

set_target_properties(
    openfhe_keygen
    openfhe_encrypt_query
    openfhe_encrypt_query_centroid_batched
    openfhe_encrypt_queries_batched
    openfhe_encrypt_queries_centroid_batched
    openfhe_compute_distances
    openfhe_compute_distances_serial
    openfhe_compute_distances_centroid_batched
    openfhe_compute_distances_centroid_prepacked
    openfhe_compute_distances_query_centroid_batched
    openfhe_compute_distances_batched
    openfhe_decrypt_topk
    openfhe_decrypt_topk_centroid_batched
    openfhe_decrypt_topk_query_centroid_batched
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

