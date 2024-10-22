cmake_minimum_required(VERSION 3.15)



option(LITENN_ENABLE_VULKAN "Include GPU support for neural networks with Kernel Slicer" ON)

project(LiteNN CXX)


find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

add_library(litenn_project_compile_options INTERFACE)
target_compile_features(litenn_project_compile_options INTERFACE cxx_std_17)

set(LITENN_PROJECT_ROOT ${PROJECT_SOURCE_DIR})
set(LITENN_SRC_DIR ${LITENN_PROJECT_ROOT}/src)
set(LITENN_TEST_DIR ${LITENN_PROJECT_ROOT}/tests)
set(LITENN_RESOURCES_DIR ${LITENN_PROJECT_ROOT}/resources)

set(LITENN_EXT_DEPS ${LITENN_PROJECT_ROOT}/3rd_party/ext)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})


if(LITENN_ENABLE_VULKAN)

  find_package(Vulkan REQUIRED)
  find_package(LLVM 17 REQUIRED CONFIG)
  find_package(Clang 17 REQUIRED)

  include(cmake/kernel_slicer.cmake)

endif()

include(cmake/includes.cmake)

add_subdirectory(3rd_party)


add_subdirectory(${LITENN_SRC_DIR})
add_subdirectory(${LITENN_TEST_DIR})

