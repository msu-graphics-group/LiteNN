set(MODULE_NAME litenn_test)


set(MODULE_SOURCES
    nn_tests.cpp
    nn_tests_performance.cpp
    standalone_tests.cpp
    direct/nnd_tests.cpp
    dataset.cpp
)

set(MODULE_LIBS
    LiteNN stb
)

if(LITENN_ENABLE_VULKAN)
    set(MODULE_LIBS ${MODULE_LIBS}
        litenn_ext
    )
endif()