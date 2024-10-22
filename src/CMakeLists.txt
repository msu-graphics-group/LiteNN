enable_language(CXX)
enable_language(C)

add_library(litenn_lib_compile_options INTERFACE)
target_link_libraries(litenn_lib_compile_options INTERFACE litenn_project_compile_options)
target_compile_options(litenn_lib_compile_options INTERFACE -m64)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(litenn_lib_compile_options INTERFACE -O0 -g)
else()
    target_compile_options(litenn_lib_compile_options INTERFACE -O1)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(litenn_lib_compile_options INTERFACE -g)
endif()

include(properties.cmake)

if(LITENN_ENABLE_VULKAN)

    add_custom_command(OUTPUT ${MODULE_GENERATED_SOURCES} ${LITENN_SRC_DIR}/shaders_gpu
                       COMMAND ${KERNEL_SLICER} ${LITENN_SRC_DIR}/tensor_processor_impl.cpp 
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

    add_custom_command(OUTPUT ${LITENN_SRC_DIR}/shaders_gpu/shaders.stamp
                       COMMAND chmod ug+x ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu/build.sh
                       COMMAND /usr/bin/bash -c ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu/build.sh
                       COMMAND ${CMAKE_COMMAND} -E touch my_file.stamp
                       WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu
                       DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders_gpu)

    add_custom_target(shaders ALL
                      DEPENDS ${LITENN_SRC_DIR}/shaders_gpu/shaders.stamp)


    set(MODULE_SOURCES ${MODULE_SOURCES} ${MODULE_GENERATED_SOURCES})

    link_directories(${LITENN_EXT_DEPS}/volk)

    target_compile_definitions(litenn_lib_compile_options INTERFACE USE_GPU)
    target_compile_definitions(litenn_lib_compile_options INTERFACE USE_VOLK)

endif()

add_library(${MODULE_NAME}
    ${MODULE_SOURCES}
)

target_compile_options(${MODULE_NAME} PRIVATE -Wno-error=unused-variable)

target_link_libraries(${MODULE_NAME} PRIVATE
    ${MODULE_LIBS} litenn_lib_compile_options 
)
