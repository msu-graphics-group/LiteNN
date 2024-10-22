include_directories(${LITENN_EXT_DEPS}/LiteMath)
include_directories(3rd_party)
include_directories(3rd_party/stb)

if(LITENN_ENABLE_VULKAN)
    include_directories(${LITENN_EXT_DEPS}/vkutils)
    include_directories(${LITENN_EXT_DEPS}/volk)
endif()

include_directories(src)