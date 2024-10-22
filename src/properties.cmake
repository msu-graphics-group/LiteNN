set(MODULE_NAME LiteNN)


set(MODULE_SOURCES
    direct/tensors.cpp
    direct/neural_network.cpp
    direct/siren.cpp

    tensor_processor.cpp
    tensor_processor_impl.cpp
    tensor_compiler.cpp
    neural_network.cpp
    tensor_token.cpp
    siren.cpp
)

set(MODULE_GENERATED_SOURCES
    ${LITENN_SRC_DIR}/tensor_processor_impl_gpu.cpp
    ${LITENN_SRC_DIR}/tensor_processor_impl_gpu_ds.cpp
    ${LITENN_SRC_DIR}/tensor_processor_impl_gpu_init.cpp
)

set(MODULE_LIBS
    OpenMP::OpenMP_CXX
    litenn_ext
)

if(LITENN_ENABLE_VULKAN)

    set(MODULE_LIBS ${MODULE_LIBS}
        volk 
        dl
        Vulkan::Vulkan
    )

endif()
