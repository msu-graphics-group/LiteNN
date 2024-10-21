cmake_minimum_required(VERSION 3.5)

project(nn_test)
set(CMAKE_CXX_STANDARD 17)
option(MODULE_VULKAN "include GPU support for neural networks" ON)

set(MODULES_SRC "")
set(MODULES_INCLUDE_DIRS "modules")
set(MODULES_LIBRARIES "")
set(MODULES_COMPILE_FLAGS "")

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

include(FetchContent)
FetchContent_Declare(
  LiteMath
  URL      https://github.com/msu-graphics-group/LiteMath/archive/refs/heads/main.zip
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/LiteMath
)

FetchContent_MakeAvailable(LiteMath)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/LiteMath)

if (MODULE_VULKAN)

  set(VKGEN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/tensor_processor_impl_gpu.cpp 
                   ${CMAKE_CURRENT_SOURCE_DIR}/tensor_processor_impl_gpu_ds.cpp
                   ${CMAKE_CURRENT_SOURCE_DIR}/tensor_processor_impl_gpu_init.cpp) 

  include(cmake/kernel_slicer.cmake)

  add_custom_command(OUTPUT ${VKGEN_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu
                     COMMAND ${KERNEL_SLICER} ${CMAKE_CURRENT_SOURCE_DIR}/tensor_processor_impl.cpp 
                                -mainClass TensorProcessorImpl 
                                -stdlibfolder ${KERNEL_SLICER_DIR}/TINYSTL 
                                -pattern ipv 
                                -reorderLoops YX 
                                -I${KERNEL_SLICER_DIR}/apps/LiteMath ignore 
                                -I${KERNEL_SLICER_DIR}/apps/LiteMathAux ignore 
                                -I${KERNEL_SLICER_DIR}/TINYSTL ignore 
                                -shaderCC glsl 
                                -suffix _GPU 
                                -DKERNEL_SLICER -v
                     WORKING_DIRECTORY ${KERNEL_SLICER_DIR}
                     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tensor_processor_impl.cpp)

  add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu/shaders.stamp
                    COMMAND chmod ug+x ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu/build.sh
                    COMMAND /usr/bin/bash -c ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu/build.sh
                    COMMAND ${CMAKE_COMMAND} -E touch my_file.stamp
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu)

  add_custom_target(shaders ALL
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu/shaders.stamp)

  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

  find_package(Vulkan)
  FetchContent_Declare(
    volk
    URL      https://github.com/zeux/volk/archive/refs/heads/master.zip
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/volk
  )
  FetchContent_Declare(
    vkutils
    URL      https://github.com/msu-graphics-group/vk-utils/archive/refs/heads/master.zip
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils
  )
  FetchContent_MakeAvailable(volk vkutils)

  set(MODULES_INCLUDE_DIRS ${MODULES_INCLUDE_DIRS} ${Vulkan_INCLUDE_DIR}
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils
                                                   ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/volk)

  link_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/volk)

  add_compile_definitions(USE_GPU)
  add_compile_definitions(USE_VOLK)

  set(VKUTILS_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils/vk_utils.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils/vk_copy.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils/vk_buffers.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils/vk_images.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils/vk_context.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils/vk_alloc_simple.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils/vk_pipeline.cpp
                     ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/vkutils/vk_descriptor_sets.cpp)   


  set(MODULES_SRC ${MODULES_SRC} ${VKGEN_SOURCE} ${VKUTILS_SOURCE})

  set(MODULES_LIBRARIES ${MODULES_LIBRARIES} ${Vulkan_LIBRARY} OpenMP::OpenMP_CXX volk dl)
endif()

set(SRC
      direct/tensors.cpp
      direct/neural_network.cpp
      direct/siren.cpp
      direct/nnd_tests.cpp
      
      tensor_processor.cpp
      tensor_processor_impl.cpp
      tensor_compiler.cpp
      neural_network.cpp
      tensor_token.cpp
      siren.cpp
      nn_tests.cpp
      dataset.cpp
      nn_tests_performance.cpp
      
      standalone_tests.cpp)  

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++17 -g -O3 -W -Wno-attributes -Wno-sign-compare -Wno-unused-parameter ")
add_executable(nn_test ${MODULES_SRC} ${SRC})
target_include_directories(nn_test PUBLIC ${MODULES_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nn_test PRIVATE ${MODULES_LIBRARIES})
target_link_libraries(nn_test PUBLIC OpenMP::OpenMP_CXX)