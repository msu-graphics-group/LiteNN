enable_language(CXX)
enable_language(C)

add_library(litenn_test_compile_options INTERFACE)
target_link_libraries(litenn_test_compile_options INTERFACE litenn_project_compile_options)
target_compile_options(litenn_test_compile_options INTERFACE -m64)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(litenn_test_compile_options INTERFACE -O0 -g)
else()
    target_compile_options(litenn_test_compile_options INTERFACE -O1)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(litenn_test_compile_options INTERFACE -g)
endif()

include(properties.cmake)

add_executable(${MODULE_NAME}
    ${MODULE_SOURCES}
)

target_compile_definitions(litenn_test_compile_options INTERFACE USE_GPU)
target_compile_definitions(litenn_test_compile_options INTERFACE USE_VOLK)


target_compile_options(${MODULE_NAME} PRIVATE -Wno-error=unused-variable)

target_link_libraries(${MODULE_NAME}
    ${MODULE_LIBS} litenn_test_compile_options 
)
