enable_language(CXX)
enable_language(C)

add_library(litenn_ext_compile_options INTERFACE)
target_compile_options(litenn_ext_compile_options INTERFACE -m64)

target_compile_definitions(litenn_ext_compile_options INTERFACE USE_GPU)
target_compile_definitions(litenn_ext_compile_options INTERFACE USE_VOLK)

#FETCHING LIBRARIES START
include(FetchContent)

FetchContent_Declare(
    LiteMath
    URL      https://github.com/msu-graphics-group/LiteMath/archive/refs/heads/main.zip
    SOURCE_DIR ${LITENN_EXT_DEPS}/ext/LiteMath
)
FetchContent_MakeAvailable(LiteMath)

if(LITENN_ENABLE_VULKAN)
    FetchContent_Declare(
        volk
        URL      https://github.com/zeux/volk/archive/refs/heads/master.zip
        SOURCE_DIR ${LITENN_EXT_DEPS}/ext/volk
    )

    FetchContent_Declare(
        vkutils
        URL      https://github.com/msu-graphics-group/vk-utils/archive/refs/heads/master.zip
        SOURCE_DIR ${LITENN_EXT_DEPS}/ext/vkutils
    )

    FetchContent_MakeAvailable(volk vkutils)

#FETCHING LIBRARIES END



    include(properties.cmake)


    add_library(${MODULE_NAME}
        ${MODULE_SOURCES}
    )

    target_compile_options(${MODULE_NAME} PRIVATE -Wno-error=unused-variable)

    target_link_libraries(${MODULE_NAME} 
        PUBLIC
        ${MODULE_LIBS}
        PRIVATE litenn_ext_compile_options 
    )
endif()
