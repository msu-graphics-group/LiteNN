include_directories(${LITENN_EXT_DEPS}/ext/LiteMath)
include_directories(3rd_party)
include_directories(3rd_party/stb)

if(LITENN_ENABLE_VULKAN)
    include_directories(${LITENN_EXT_DEPS}/ext/vkutils)
    include_directories(${LITENN_EXT_DEPS}/ext/volk)
endif()

include_directories(src)
include_directories(include/LiteNN)