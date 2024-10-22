set(MODULE_NAME litenn_ext)


set(MODULE_SOURCES

)

set(MODULE_LIBS

)


if(LITENN_ENABLE_VULKAN)
    link_directories(${LITENN_EXT_DEPS}/volk)

    set(MODULE_SOURCES ${MODULE_SOURCES}
        ext/vkutils/vk_utils.cpp
        ext/vkutils/vk_copy.cpp
        ext/vkutils/vk_buffers.cpp
        ext/vkutils/vk_images.cpp
        ext/vkutils/vk_context.cpp
        ext/vkutils/vk_alloc_simple.cpp
        ext/vkutils/vk_pipeline.cpp
        ext/vkutils/vk_descriptor_sets.cpp
        ext/volk/volk.c
    )
    
    set(MODULE_LIBS ${MODULE_LIBS}
        Vulkan::Vulkan volk
    )
endif()
